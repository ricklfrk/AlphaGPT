"""
Strategy Backtester V2 - 正確的回測邏輯

核心改進：
- 每天收盤後用當天數據計算信號
- 信號不變則持倉不變（不強制每天換倉）
- 信號變化時才在次日開盤換倉
- 完全模擬實際交易流程

交易邏輯：
  Day T 收盤: 計算信號 (使用 T 及之前的數據)
  Day T+1 開盤: 根據 T 的信號調整倉位
  
  如果 T 的信號是 BUY:
    - T+1 開盤買入（如果之前不是多頭）
    - 持有直到信號變為 HOLD 或 SELL
  
  如果 T 的信號是 SELL:
    - T+1 開盤做空（如果之前不是空頭）
    - 持有直到信號變為 HOLD 或 BUY
  
  如果 T 的信號是 HOLD:
    - T+1 開盤平倉（如果有倉位）

Usage:
    python backtest_strategy_v2.py --strategy output/NVDA_T1OT2O_xxx/best_strategy.json --tickers NVDA --period 3y
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yfinance as yf
import matplotlib.pyplot as plt

# Windows 編碼修復
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ==================== 輸出目錄 ====================
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_backtest_folder(strategy_name, tickers):
    """創建帶時間戳的回測輸出文件夾"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    ticker_str = "_".join(tickers[:3])
    if len(tickers) > 3:
        ticker_str += f"_+{len(tickers)-3}"
    folder_name = f"backtest_{strategy_name}_{ticker_str}_{timestamp}"
    folder_path = os.path.join(OUTPUT_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

# ==================== 信號閾值配置 ====================
DEFAULT_SIGNAL_THRESHOLD = 0.1
DEFAULT_HYSTERESIS = 0.05         # 死區/滯後閾值，避免頻繁交易

# ==================== 交易成本 ====================
COST_RATE = 0.0010  # 單邊 0.10%（10 bps，適用於個股；SPY 可降至 5 bps）

# ==================== 算子定義 ====================
@torch.jit.script
def _ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    if d == 0: return x
    pad = torch.zeros((x.shape[0], d), device=x.device)
    return torch.cat([pad, x[:, :-d]], dim=1)

@torch.jit.script
def _ts_delta(x: torch.Tensor, d: int) -> torch.Tensor:
    return x - _ts_delay(x, d)

@torch.jit.script
def _ts_zscore(x: torch.Tensor, d: int) -> torch.Tensor:
    if d <= 1: return torch.zeros_like(x)
    B, T = x.shape
    pad = torch.zeros((B, d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    mean = windows.mean(dim=-1)
    std = windows.std(dim=-1) + 1e-6
    return (x - mean) / std

@torch.jit.script
def _ts_decay_linear(x: torch.Tensor, d: int) -> torch.Tensor:
    if d <= 1: return x
    B, T = x.shape
    pad = torch.zeros((B, d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    w = torch.arange(1, d + 1, device=x.device, dtype=x.dtype)
    w = w / w.sum()
    return (windows * w).sum(dim=-1)

@torch.jit.script
def _ts_max(x: torch.Tensor, d: int) -> torch.Tensor:
    if d <= 1: return x
    B, T = x.shape
    pad = torch.full((B, d - 1), float('-inf'), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    return windows.max(dim=-1)[0]

@torch.jit.script
def _ts_min(x: torch.Tensor, d: int) -> torch.Tensor:
    if d <= 1: return x
    B, T = x.shape
    pad = torch.full((B, d - 1), float('inf'), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    return windows.min(dim=-1)[0]

@torch.jit.script
def _ts_rank(x: torch.Tensor, d: int) -> torch.Tensor:
    """計算滾動窗口內的百分位排名"""
    if d <= 1: return torch.zeros_like(x)
    B, T = x.shape
    pad = torch.zeros((B, d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    # 排名：當前值在窗口中的位置 / 窗口大小
    current_val = x.unsqueeze(-1)
    rank = (windows < current_val).float().sum(dim=-1) / d
    return rank

# ==================== 配置 ====================
# V2 因子列表（與 times_us_v2.py 保持一致）
FEATURES = [
    'RET',      # 日收益率
    'RET5',     # 5日收益率
    'VOL_CHG',  # 成交量變化
    'V_RET',    # 量價收益
    'TREND',    # 趨勢（相對 MA60）
    'ATR',      # 平均真實波幅（波動率）
    'RSI',      # 相對強弱指數
    'CLV',      # Close Location Value
    'RS',       # 相對強度（vs SPY）
    'MOM',      # 動量（20日）
    'VIX',      # VIX 恐慌指數
]

# V1 因子列表（兼容舊策略）
FEATURES_V1 = ['RET', 'RET5', 'VOL_CHG', 'V_RET', 'TREND']

# ==================== 算子配置（必須與 times_us_v2.py 完全一致！）====================
OPS_CONFIG = [
    ('ADD', lambda x, y: x + y, 2),
    ('SUB', lambda x, y: x - y, 2),
    ('MUL', lambda x, y: x * y, 2),
    ('DIV', lambda x, y: x / (y + 1e-6 * torch.sign(y + 1e-9)), 2),
    ('NEG', lambda x: -x, 1),
    ('ABS', lambda x: torch.abs(x), 1),
    ('SIGN', lambda x: torch.sign(x), 1),
    ('DELTA5', lambda x: _ts_delta(x, 5), 1),
    ('MA5', lambda x: _ts_decay_linear(x, 5), 1),    # Token 19
    ('MA10', lambda x: _ts_decay_linear(x, 10), 1),  # Token 20
    ('STD10', lambda x: _ts_zscore(x, 10), 1),       # Token 21
    ('RANK10', lambda x: _ts_rank(x, 10), 1),        # Token 22
]

VOCAB = FEATURES + [cfg[0] for cfg in OPS_CONFIG]
OP_FUNC_MAP = {i + len(FEATURES): cfg[1] for i, cfg in enumerate(OPS_CONFIG)}
OP_ARITY_MAP = {i + len(FEATURES): cfg[2] for i, cfg in enumerate(OPS_CONFIG)}


class RealisticBacktester:
    """
    現實回測器 - 模擬真實交易流程
    
    交易邏輯：
    - T日收盤信號 → T+1開盤建倉 → 持有直到信號改變
    - 收益計算: position * (Close[T] - Close[T-1]) / Close[T-1]
    
    關鍵邏輯：
    1. 每天收盤後，用當天及之前的數據計算信號
    2. 次日開盤時根據信號調整倉位
    3. 持有直到信號改變
    
    支持 V1 (5因子) 和 V2 (11因子) 策略
    """
    
    def __init__(self, strategy_file: str, override_threshold: float = None, override_hysteresis: float = None):
        """初始化回測器"""
        with open(strategy_file, 'r') as f:
            strategy = json.load(f)
        
        # 檢測策略類型（單一公式 vs 集成策略）
        self.is_ensemble = strategy.get('type') == 'ensemble'
        
        if self.is_ensemble:
            # 集成策略：多個公式的加權組合
            self.formula_tokens_list = strategy['formula_tokens_list']
            self.formula_tokens = self.formula_tokens_list[0]  # 用第一個作為默認（用於兼容性）
            # 獲取每個組件的權重
            self.ensemble_weights = []
            if 'component_formulas' in strategy:
                for comp in strategy['component_formulas']:
                    self.ensemble_weights.append(comp.get('weight', 1.0 / len(self.formula_tokens_list)))
            else:
                # 默認均等權重
                self.ensemble_weights = [1.0 / len(self.formula_tokens_list)] * len(self.formula_tokens_list)
            self.ensemble_count = len(self.formula_tokens_list)
        else:
            # 單一公式策略
            self.formula_tokens = strategy['formula_tokens']
            self.is_ensemble = False
        
        self.formula_readable = strategy.get('formula_readable', self._decode_formula())
        self.strategy_file = strategy_file
        self.strategy_name = os.path.basename(os.path.dirname(strategy_file))
        
        # 檢測策略版本
        self.version = strategy.get('version', '1.0')
        self.is_v2 = self.version.startswith('2')
        
        # 根據版本選擇因子列表
        self.features = FEATURES if self.is_v2 else FEATURES_V1
        
        # 歸一化參數
        self.norm_params = strategy.get('norm_params', None)
        
        # 閾值
        self.strategy_threshold = strategy.get('signal_threshold', DEFAULT_SIGNAL_THRESHOLD)
        self.signal_threshold = override_threshold if override_threshold is not None else self.strategy_threshold
        
        # 滯後閾值（死區）
        self.strategy_hysteresis = strategy.get('hysteresis', DEFAULT_HYSTERESIS)
        self.hysteresis = override_hysteresis if override_hysteresis is not None else self.strategy_hysteresis
        
        # SPY 和 VIX 數據（V2 策略需要）
        self.spy_data = None
        self.vix_data = None
        
        # 打印策略信息
        if self.is_ensemble:
            print(f"✅ Loaded ENSEMBLE strategy (V{self.version}): {self.ensemble_count} components")
            print(f"   Formula: {self.formula_readable[:100]}..." if len(self.formula_readable) > 100 else f"   Formula: {self.formula_readable}")
            print(f"   Weights: {self.ensemble_weights}")
        else:
            print(f"✅ Loaded strategy (V{self.version}): {self.formula_readable}")
        print(f"   Signal Threshold: ±{self.signal_threshold} (Hysteresis: {self.hysteresis})")
        print(f"   Features: {len(self.features)} ({', '.join(self.features[:5])}{'...' if len(self.features) > 5 else ''})")
        if self.norm_params:
            print(f"   Using saved normalization params")
    
    def _decode_formula(self) -> str:
        tokens = list(self.formula_tokens)
        features = self.features if hasattr(self, 'features') else FEATURES
        def _parse():
            if not tokens: return ""
            t = tokens.pop(0)
            if t < len(features): return features[t]
            args = [_parse() for _ in range(OP_ARITY_MAP[t])]
            return f"{VOCAB[t]}({','.join(args)})"
        try:
            return _parse()
        except:
            return "Invalid"
    
    def _load_benchmark_data(self, dates, start_date, end_date):
        """加載 SPY 和 VIX 數據（V2 策略需要）"""
        if not self.is_v2:
            return
        
        from datetime import timedelta
        
        # 下載 SPY
        try:
            print("   Loading SPY data for RS factor...")
            spy_ticker = yf.Ticker('SPY')
            spy_df = spy_ticker.history(start=start_date - timedelta(days=30), end=end_date)
            if not spy_df.empty:
                spy_df = spy_df.reset_index()
                spy_df['Date'] = pd.to_datetime(spy_df['Date']).dt.tz_localize(None)
                self.spy_data = spy_df.set_index('Date')['Close']
        except Exception as e:
            print(f"   Warning: Failed to load SPY data: {e}")
            self.spy_data = None
        
        # 下載 VIX
        try:
            print("   Loading VIX data...")
            vix_ticker = yf.Ticker('^VIX')
            vix_df = vix_ticker.history(start=start_date - timedelta(days=30), end=end_date)
            if not vix_df.empty:
                vix_df = vix_df.reset_index()
                vix_df['Date'] = pd.to_datetime(vix_df['Date']).dt.tz_localize(None)
                self.vix_data = vix_df.set_index('Date')['Close']
        except Exception as e:
            print(f"   Warning: Failed to load VIX data: {e}")
            self.vix_data = None
    
    def _compute_features_at_day(self, df: pd.DataFrame, day_idx: int) -> torch.Tensor:
        """
        計算截至 day_idx（包含）的特徵
        
        關鍵：只使用 day_idx 及之前的數據，不使用未來數據！
        支持 V1 (5因子) 和 V2 (11因子)
        """
        # 只取到 day_idx 的數據
        df_slice = df.iloc[:day_idx + 1]
        
        close = df_slice['Close'].values.astype(np.float32)
        high = df_slice['High'].values.astype(np.float32) if 'High' in df_slice else close
        low = df_slice['Low'].values.astype(np.float32) if 'Low' in df_slice else close
        vol = df_slice['Volume'].values.astype(np.float32)
        
        # 獲取當前日期（用於對齊 SPY/VIX）
        current_date = df_slice.index[day_idx] if hasattr(df_slice.index, '__getitem__') else None
        
        # ==================== 基礎因子 (V1 + V2) ====================
        # RET
        ret = np.zeros_like(close)
        if len(close) > 1:
            ret[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-6)
        
        # RET5
        ret5 = pd.Series(close).pct_change(5).fillna(0).values.astype(np.float32)
        
        # VOL_CHG
        vol_ma = pd.Series(vol).rolling(20, min_periods=1).mean().values
        vol_chg = np.zeros_like(vol)
        mask = vol_ma > 0
        vol_chg[mask] = vol[mask] / vol_ma[mask] - 1
        vol_chg = np.nan_to_num(vol_chg).astype(np.float32)
        
        # V_RET
        v_ret = (ret * (vol_chg + 1)).astype(np.float32)
        
        # TREND
        ma60 = pd.Series(close).rolling(60, min_periods=1).mean().values
        trend = np.zeros_like(close)
        mask = ma60 > 0
        trend[mask] = close[mask] / ma60[mask] - 1
        trend = np.nan_to_num(trend).astype(np.float32)
        
        # Robust normalization
        def robust_norm(x, feature_name=None):
            x = x.astype(np.float32)
            if self.norm_params and feature_name and feature_name in self.norm_params:
                median = self.norm_params[feature_name]['median']
                mad = self.norm_params[feature_name]['mad']
            else:
                median = np.nanmedian(x)
                mad = np.nanmedian(np.abs(x - median)) + 1e-6
            res = (x - median) / mad
            return np.clip(res, -5, 5).astype(np.float32)
        
        # V1 策略：只返回5個因子
        if not self.is_v2:
            features = torch.stack([
                torch.from_numpy(robust_norm(ret, 'ret')),
                torch.from_numpy(robust_norm(ret5, 'ret5')),
                torch.from_numpy(robust_norm(vol_chg, 'vol_chg')),
                torch.from_numpy(robust_norm(v_ret, 'v_ret')),
                torch.from_numpy(robust_norm(trend, 'trend')),
            ])
            return features
        
        # ==================== V2 額外因子 ====================
        # ATR (Average True Range)
        tr1 = high - low
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        atr = pd.Series(tr).rolling(14, min_periods=1).mean().values.astype(np.float32)
        atr_norm = atr / (close + 1e-6)  # 標準化
        
        # RSI (Relative Strength Index)
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / (loss + 1e-6)
        rsi = (100 - 100 / (1 + rs)).fillna(50).values.astype(np.float32)
        rsi_norm = (rsi - 50) / 25.0  # 標準化到約 [-2, 2]
        
        # CLV (Close Location Value)
        hl_range = high - low + 1e-6
        clv = ((close - low) - (high - close)) / hl_range
        clv = np.nan_to_num(clv).astype(np.float32)
        
        # RS (Relative Strength vs SPY)
        if self.spy_data is not None and current_date is not None:
            try:
                # 對齊 SPY 數據
                spy_aligned = self.spy_data.reindex(df_slice.index).ffill().bfill().values.astype(np.float32)
                spy_ret = np.zeros_like(spy_aligned)
                spy_ret[1:] = (spy_aligned[1:] - spy_aligned[:-1]) / (spy_aligned[:-1] + 1e-6)
                rs_factor = ret - spy_ret  # 超額收益
                rs_20 = pd.Series(rs_factor).rolling(20, min_periods=1).sum().values.astype(np.float32)
            except:
                rs_20 = np.zeros_like(close)
        else:
            rs_20 = np.zeros_like(close)
        
        # MOM (Momentum 20日)
        mom = pd.Series(close).pct_change(20).fillna(0).values.astype(np.float32)
        
        # VIX
        if self.vix_data is not None and current_date is not None:
            try:
                vix_aligned = self.vix_data.reindex(df_slice.index).ffill().bfill().values.astype(np.float32)
                vix_norm = (vix_aligned - 20) / 10  # 標準化
                vix_norm = np.clip(vix_norm, -3, 6).astype(np.float32)
            except:
                vix_norm = np.zeros_like(close)
        else:
            vix_norm = np.zeros_like(close)
        
        # 構建 V2 特徵張量（11 個因子）
        features = torch.stack([
            torch.from_numpy(robust_norm(ret, 'ret')),
            torch.from_numpy(robust_norm(ret5, 'ret5')),
            torch.from_numpy(robust_norm(vol_chg, 'vol_chg')),
            torch.from_numpy(robust_norm(v_ret, 'v_ret')),
            torch.from_numpy(robust_norm(trend, 'trend')),
            torch.from_numpy(robust_norm(atr_norm, 'atr')),
            torch.from_numpy(rsi_norm),  # RSI 已標準化
            torch.from_numpy(clv),  # CLV 已在 [-1, 1]
            torch.from_numpy(robust_norm(rs_20, 'rs')),
            torch.from_numpy(robust_norm(mom, 'mom')),
            torch.from_numpy(vix_norm),  # VIX 已標準化
        ])
        
        return features
    
    def _execute_formula(self, features: torch.Tensor, formula_tokens: list = None) -> Optional[float]:
        """執行公式，返回最後一天的因子值
        
        Args:
            features: 特徵張量
            formula_tokens: 可選的公式tokens，如果不提供則使用 self.formula_tokens
        """
        stack = []
        n_features = len(self.features)
        tokens = formula_tokens if formula_tokens is not None else self.formula_tokens
        try:
            for t in reversed(tokens):
                if t < n_features:
                    stack.append(features[t].unsqueeze(0))
                else:
                    arity = OP_ARITY_MAP[t]
                    if len(stack) < arity:
                        return None
                    args = [stack.pop() for _ in range(arity)]
                    func = OP_FUNC_MAP[t]
                    if arity == 2:
                        res = func(args[0], args[1])
                    else:
                        res = func(args[0])
                    if torch.isnan(res).any():
                        res = torch.nan_to_num(res)
                    stack.append(res)
            
            if len(stack) >= 1:
                final = stack[-1]
                if final.dim() == 2 and final.shape[0] == 1:
                    final = final.squeeze(0)
                # 返回最後一天的因子值
                return final[-1].item()
        except Exception as e:
            print(f"Formula execution error: {e}")
        return None
    
    def _execute_ensemble_formula(self, features: torch.Tensor) -> Optional[float]:
        """執行集成公式，返回加權平均的因子值"""
        if not self.is_ensemble:
            return self._execute_formula(features)
        
        total_value = 0.0
        total_weight = 0.0
        valid_count = 0
        component_values = []
        
        for i, tokens in enumerate(self.formula_tokens_list):
            value = self._execute_formula(features, tokens)
            component_values.append(value)
            if value is not None:
                weight = self.ensemble_weights[i]
                total_value += value * weight
                total_weight += weight
                valid_count += 1
        
        if valid_count == 0:
            return None
        
        # Debug: 首次執行時打印各組件值
        if not hasattr(self, '_ensemble_debug_printed'):
            self._ensemble_debug_printed = True
            import sys
            print(f"\n" + "="*60, flush=True)
            print(f"📊 ENSEMBLE DEBUG (first calculation)", flush=True)
            print(f"="*60, flush=True)
            print(f"   Total components: {len(self.formula_tokens_list)}", flush=True)
            print(f"   Valid components: {valid_count}", flush=True)
            for i, val in enumerate(component_values):
                status = f"{val:.4f}" if val is not None else "None (FAILED!)"
                print(f"   [{i+1}] value={status}, weight={self.ensemble_weights[i]:.2f}", flush=True)
            avg = total_value / total_weight if total_weight > 0 else 0
            print(f"   Weighted average factor: {avg:.4f}", flush=True)
            print(f"   Signal (tanh): {np.tanh(avg):.4f}", flush=True)
            print(f"="*60 + "\n", flush=True)
            sys.stdout.flush()
        
        # 返回加權平均值
        return total_value / total_weight if total_weight > 0 else total_value / valid_count
    
    def get_signal_at_day(self, df: pd.DataFrame, day_idx: int) -> Tuple[float, int]:
        """
        計算第 day_idx 天收盤後的信號
        
        Returns:
            (signal_strength, position): signal_strength in [-1, 1], position in {-1, 0, 1}
        """
        features = self._compute_features_at_day(df, day_idx)
        
        # 根據策略類型選擇執行方式
        if self.is_ensemble:
            factor_value = self._execute_ensemble_formula(features)
        else:
            factor_value = self._execute_formula(features)
        
        if factor_value is None:
            return 0.0, 0
        
        signal_strength = np.tanh(factor_value)
        
        # 確定倉位
        if signal_strength > self.signal_threshold:
            position = 1  # 做多
        elif signal_strength < -self.signal_threshold:
            position = -1  # 做空
        else:
            position = 0  # 觀望
        
        return signal_strength, position
    
    def backtest(
        self,
        ticker: str,
        period: str = "1y",
        initial_capital: float = 100000,
        start_date: str = None,
        end_date: str = None,
    ) -> Optional[Dict]:
        """
        執行回測
        
        核心邏輯：
        - Day T 收盤後計算信號
        - Day T+1 開盤時根據信號調整倉位
        - 收益計算：position[T] * (Open[T+1] - Open[T]) / Open[T]
          （T 的信號決定 T+1 開盤的倉位，收益是 Open[T] 到 Open[T+1]）
        
        等等，這裡有個問題！
        
        正確的邏輯應該是：
        - Day T 收盤後計算信號，決定 T+1 的倉位
        - 如果信號是 BUY，在 T+1 開盤買入
        - 持有直到信號變為非 BUY（可能是 T+2, T+3, ... 或更久）
        - 收益計算：position 持有期間的價格變化
        
        讓我重新設計...
        
        簡化版本（每日結算）：
        - Day T 收盤後計算信號 → position_for_T+1
        - Day T+1 的收益 = position_for_T+1 * (Close[T+1] - Open[T+1]) / Open[T+1]
        
        但這是 T1OTC 模式...
        
        對於持有模式：
        - Day T 收盤後計算信號
        - 如果信號改變，在 T+1 開盤調整倉位
        - 持倉期間的收益 = 累計的 (Close[t] - Close[t-1]) / Close[t-1]
        
        讓我實現正確的邏輯...
        """
        print(f"\n{'='*70}")
        print(f"📊 Backtesting {ticker} (Realistic Mode)")
        print(f"{'='*70}")
        
        # 下載數據
        # 重要：需要額外的歷史數據進行 warmup（特徵計算需要至少 60 天）
        from datetime import datetime, timedelta
        
        yf_ticker = yf.Ticker(ticker)
        
        if start_date and end_date:
            # 下載額外的 180 天歷史數據用於 warmup
            actual_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=180)).strftime('%Y-%m-%d')
            df = yf_ticker.history(start=actual_start, end=end_date, auto_adjust=True)
            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date'])
            test_start_idx = df[df['Date'] >= start_date].index[0] if len(df[df['Date'] >= start_date]) > 0 else 60
        else:
            # 使用 period 時，也需要額外下載 warmup 數據
            # 先下載 period + 額外 180 天
            period_days_map = {
                '1mo': 30, '3mo': 90, '6mo': 180, 
                '1y': 365, '2y': 730, '3y': 1095, '5y': 1825, '10y': 3650, 'max': 9999
            }
            period_days = period_days_map.get(period, 365)
            total_days = period_days + 120  # 額外 120 天用於 warmup
            
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=total_days)
            
            df = yf_ticker.history(start=start_dt.strftime('%Y-%m-%d'), 
                                   end=end_dt.strftime('%Y-%m-%d'), 
                                   auto_adjust=True)
            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date'])
            
            # 計算測試期開始日期（實際數據中的 warmup 期後）
            if len(df) > 60:
                # 測試期開始 = 總天數 - period 對應的交易日數
                # 大約 252 個交易日/年
                period_trading_days = int(period_days * 252 / 365)
                test_start_idx = max(60, len(df) - period_trading_days)
            else:
                test_start_idx = 60
        
        if df.empty or len(df) < 60:
            print(f"❌ Insufficient data for {ticker}")
            return None
        
        dates = pd.to_datetime(df['Date'])
        # 處理時區
        if dates.dt.tz is not None:
            dates = dates.dt.tz_localize(None)
        df.index = dates
        
        close = df['Close'].values.astype(np.float32)
        open_prices = df['Open'].values.astype(np.float32)
        
        # 為 V2 策略加載 SPY 和 VIX 數據
        if self.is_v2:
            self._load_benchmark_data(dates, dates.iloc[0], dates.iloc[-1])
        
        # 確定測試期開始位置
        warmup_days = test_start_idx
        print(f"📅 Full data period: {dates.iloc[0].date()} ~ {dates.iloc[-1].date()}")
        print(f"📅 Test period: {dates.iloc[test_start_idx].date()} ~ {dates.iloc[-1].date()}")
        print(f"📈 Total days: {len(df)} (Warmup: {test_start_idx}, Test: {len(df) - test_start_idx})")
        
        # ========== 逐日計算信號（含滯後/死區機制）==========
        signals = []
        positions = []
        
        threshold = self.signal_threshold
        hysteresis = self.hysteresis
        close_threshold = threshold - hysteresis  # 平倉閾值更低
        
        print(f"\n🔄 Calculating daily signals (warmup={warmup_days} days)...")
        print(f"   Threshold: ±{threshold}, Hysteresis: {hysteresis} (Close: ±{close_threshold})")
        
        for day_idx in range(len(df)):
            if day_idx < 60:  # 絕對 warmup 期（需要 60 天歷史數據計算特徵）
                # Warmup 期間，不產生信號
                signals.append(0.0)
                positions.append(0)
            else:
                # 用 day_idx 及之前的數據計算信號（不含仓位判断）
                features = self._compute_features_at_day(df, day_idx)
                
                # 根據策略類型選擇執行方式
                if self.is_ensemble:
                    factor_value = self._execute_ensemble_formula(features)
                else:
                    factor_value = self._execute_formula(features)
                
                if factor_value is None:
                    signal_strength = 0.0
                else:
                    signal_strength = np.tanh(factor_value)
                
                signals.append(signal_strength)
                
                # 使用滯後機制確定倉位（與 times_us_v2.py 的 discretize_position 一致）
                prev_pos = positions[-1] if positions else 0
                
                if prev_pos == 0:
                    # 當前無倉位：使用開倉閾值
                    if signal_strength > threshold:
                        position = 1
                    elif signal_strength < -threshold:
                        position = -1
                    else:
                        position = 0
                elif prev_pos == 1:
                    # 當前持多倉
                    if signal_strength < -threshold:
                        position = -1  # 翻空
                    elif signal_strength < close_threshold:
                        position = 0   # 平倉（signal 跌破平倉閾值）
                    else:
                        position = 1   # 繼續持多（signal >= close_threshold）
                else:  # prev_pos == -1
                    # 當前持空倉
                    if signal_strength > threshold:
                        position = 1   # 翻多
                    elif signal_strength > -close_threshold:
                        position = 0   # 平倉（signal 漲破平倉閾值）
                    else:
                        position = -1  # 繼續持空（signal <= -close_threshold）
                
                positions.append(position)
        
        signals = np.array(signals)
        positions = np.array(positions)
        
        # 只保留測試期數據進行收益計算
        # 切片到測試期（兩種模式都需要，因為都有 warmup 數據）
        # 
        # 重要：需要保留 test_start_idx - 1 的 position，因為：
        # - positions[test_start_idx - 1] = start_date 前一天收盤的信號
        # - 這個信號決定 start_date 當天的倉位
        prev_day_position = positions[test_start_idx - 1] if test_start_idx > 0 else 0
        
        if test_start_idx > 0:
            test_dates = dates[test_start_idx:]
            test_close = close[test_start_idx:]
            test_open_prices = open_prices[test_start_idx:]
            test_signals = signals[test_start_idx:]
            test_positions = positions[test_start_idx:]
            
            # 更新變量
            dates = test_dates.reset_index(drop=True)
            close = test_close
            open_prices = test_open_prices
            signals = test_signals
            positions = test_positions
            
            print(f"   📊 Sliced to test period: {len(dates)} days")
        
        # ========== 計算收益 ==========
        # T日收盤計算信號 → T+1開盤建倉 → 持有直到信號改變
        #
        # 正確的收益計算：
        # - 新建倉當天：Open-to-Close (開盤買入 → 收盤)
        # - 持倉中：Close-to-Close (昨收 → 今收)
        # - 平倉當天：Close-to-Open (昨收 → 今開賣出)
        
        # 倉位 shift：positions[T] 決定 T+1 的倉位
        # 重要：第 0 天的倉位由 prev_day_position 決定（start_date 前一天的信號）
        effective_positions = np.zeros(len(positions))
        effective_positions[0] = prev_day_position  # ← 使用前一天的信號
        effective_positions[1:] = positions[:-1]    # T 的信號影響 T+1 的倉位
        
        # 計算每日收益（根據倉位變化選擇正確的價格）
        strategy_ret = np.zeros(len(close))
        
        # 處理第 0 天（T-1 收盤產生信號 → T 開盤買入）
        # 買入價 = Open[0]，收益 = (Close[0] - Open[0]) / Open[0]
        if effective_positions[0] != 0:
            ret = effective_positions[0] * (close[0] - open_prices[0]) / (open_prices[0] + 1e-6)
            strategy_ret[0] = ret - COST_RATE  # 建倉成本
        
        for i in range(1, len(close)):
            today_pos = effective_positions[i]
            yesterday_pos = effective_positions[i - 1]
            
            if today_pos == 0:
                # 今天空倉
                if yesterday_pos != 0:
                    # 平倉：昨收 → 今開
                    ret = yesterday_pos * (open_prices[i] - close[i - 1]) / (close[i - 1] + 1e-6)
                    strategy_ret[i] = ret - COST_RATE  # 平倉成本
                else:
                    strategy_ret[i] = 0
            else:
                # 今天有倉位
                if yesterday_pos == 0:
                    # 新建倉：今開 → 今收
                    ret = today_pos * (close[i] - open_prices[i]) / (open_prices[i] + 1e-6)
                    strategy_ret[i] = ret - COST_RATE  # 建倉成本
                elif yesterday_pos == today_pos:
                    # 持倉中：昨收 → 今收
                    ret = today_pos * (close[i] - close[i - 1]) / (close[i - 1] + 1e-6)
                    strategy_ret[i] = ret
                else:
                    # 翻倉（多轉空 或 空轉多）
                    # 先平舊倉：昨收 → 今開
                    ret1 = yesterday_pos * (open_prices[i] - close[i - 1]) / (close[i - 1] + 1e-6)
                    # 再建新倉：今開 → 今收
                    ret2 = today_pos * (close[i] - open_prices[i]) / (open_prices[i] + 1e-6)
                    strategy_ret[i] = ret1 + ret2 - 2 * COST_RATE  # 平倉 + 建倉成本
        
        strategy_equity = (1 + strategy_ret).cumprod()
        
        # 計算換手率（用於統計）
        turnover = np.abs(effective_positions - np.roll(effective_positions, 1))
        turnover[0] = abs(effective_positions[0])
        
        # Buy & Hold（與策略相同：start_date 開盤買入）
        # - 第 0 天：Open[0] 買入 → Close[0] 持有（Open-to-Close）
        # - 第 1 天起：持倉（Close-to-Close）
        # 
        # 這樣策略和 Buy & Hold 的買入價完全一致（都是 Open[0]）
        bh_daily_ret = np.zeros(len(close))
        if len(close) > 0:
            # 第 0 天：Open 買入 → Close
            bh_daily_ret[0] = (close[0] - open_prices[0]) / (open_prices[0] + 1e-6)
            # 第 1 天起：持倉（Close → Close）
            if len(close) > 1:
                bh_daily_ret[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-6)
        bh_equity = (1 + bh_daily_ret).cumprod()
        
        # ========== 統計 ==========
        # 計算指標
        total_ret = strategy_equity[-1] - 1
        ann_ret = strategy_equity[-1] ** (252 / len(strategy_equity)) - 1
        vol = np.std(strategy_ret) * np.sqrt(252)
        sharpe = (ann_ret - 0.02) / (vol + 1e-6)
        
        # Sortino 比率（只使用下行波動率）
        negative_ret = strategy_ret[strategy_ret < 0]
        downside_std = np.std(negative_ret) * np.sqrt(252) if len(negative_ret) > 0 else 1e-6
        sortino = (ann_ret - 0.02) / (downside_std + 1e-6)
        
        dd = 1 - strategy_equity / np.maximum.accumulate(strategy_equity)
        max_dd = np.max(dd)
        
        win_rate = np.mean(strategy_ret[strategy_ret != 0] > 0) if (strategy_ret != 0).any() else 0
        
        # 倉位統計
        long_days = (effective_positions == 1).sum()
        short_days = (effective_positions == -1).sum()
        hold_days = (effective_positions == 0).sum()
        
        # 交易次數（倉位變化次數）
        trade_count = (turnover > 0).sum()
        
        # Buy & Hold 完整指標
        bh_total_ret = bh_equity[-1] - 1
        bh_ann_ret = bh_equity[-1] ** (252 / len(bh_equity)) - 1
        bh_vol = np.std(bh_daily_ret) * np.sqrt(252)
        bh_sharpe = (bh_ann_ret - 0.02) / (bh_vol + 1e-6)
        
        # Buy & Hold Sortino
        bh_negative_ret = bh_daily_ret[bh_daily_ret < 0]
        bh_downside_std = np.std(bh_negative_ret) * np.sqrt(252) if len(bh_negative_ret) > 0 else 1e-6
        bh_sortino = (bh_ann_ret - 0.02) / (bh_downside_std + 1e-6)
        
        # Buy & Hold Max Drawdown
        bh_dd = 1 - bh_equity / np.maximum.accumulate(bh_equity)
        bh_max_dd = np.max(bh_dd)
        
        # Buy & Hold Win Rate
        bh_win_rate = np.mean(bh_daily_ret[bh_daily_ret != 0] > 0) if (bh_daily_ret != 0).any() else 0
        
        # ========== 輸出結果 ==========
        print("-" * 70)
        print(f"📊 Position Stats:")
        print(f"   Long days:  {long_days} ({long_days/len(df)*100:.1f}%)")
        print(f"   Short days: {short_days} ({short_days/len(df)*100:.1f}%)")
        print(f"   Hold days:  {hold_days} ({hold_days/len(df)*100:.1f}%)")
        print(f"   Trades:     {trade_count}")
        print("-" * 70)
        print(f"{'Metric':<20} {'Strategy':>15} {'Buy & Hold':>15}")
        print("-" * 70)
        print(f"{'Total Return':<20} {total_ret:>14.2%} {bh_total_ret:>14.2%}")
        print(f"{'Ann. Return':<20} {ann_ret:>14.2%} {bh_ann_ret:>14.2%}")
        print(f"{'Ann. Volatility':<20} {vol:>14.2%} {bh_vol:>14.2%}")
        print(f"{'Sharpe Ratio':<20} {sharpe:>14.2f} {bh_sharpe:>14.2f}")
        print(f"{'Sortino Ratio':<20} {sortino:>14.2f} {bh_sortino:>14.2f}")
        print(f"{'Max Drawdown':<20} {max_dd:>14.2%} {bh_max_dd:>14.2%}")
        print(f"{'Win Rate':<20} {win_rate:>14.2%} {bh_win_rate:>14.2%}")
        print("-" * 70)
        
        # 資金變化
        final_value = initial_capital * strategy_equity[-1]
        bh_final_value = initial_capital * bh_equity[-1]
        print(f"\n💰 Capital Change (Initial: ${initial_capital:,.2f}):")
        print(f"   Strategy:   ${final_value:,.2f} (P&L: ${final_value - initial_capital:,.2f})")
        print(f"   Buy & Hold: ${bh_final_value:,.2f} (P&L: ${bh_final_value - initial_capital:,.2f})")
        
        # ========== 生成交易記錄 ==========
        # 
        # 時間線邏輯：
        # ┌────────────────────────────────────────────────────────────────┐
        # │ T-1 收盤 → 計算 signal[T-1] → 決定 position[T-1]               │
        # │                                    ↓                           │
        # │ T 日開盤 → 根據 position[T-1] 調整倉位 = today_position        │
        # │ T 日收盤 → 計算 signal[T] → 決定 position[T] (明天的倉位)      │
        # │ T 日收益 = today_position × (Close[T] - Close[T-1])            │
        # └────────────────────────────────────────────────────────────────┘
        #
        # 字段說明：
        # - signal: T 日收盤後計算的信號 (決定 T+1 的倉位)
        # - signal_discrete: 離散化後的信號 {-1, 0, 1} (決定 T+1 的倉位)
        # - signal_continuous: 連續信號 = signal (決定 T+1 的倉位)
        # - today_position: T 日實際持有的倉位 (由 T-1 的信號決定)
        # - action: T 日開盤時的操作
        
        trade_log = []
        
        # 追蹤持股數量和成本
        shares_held = 0
        avg_cost = 0.0
        
        for i in range(len(close)):
            date_str = dates.iloc[i].strftime('%Y-%m-%d')
            open_price = open_prices[i]
            close_price = close[i]
            signal_val = signals[i]                    # T 日收盤後的信號 (決定明天倉位)
            signal_discrete = int(positions[i])       # 離散化信號 (決定明天倉位)
            today_pos = int(effective_positions[i])   # 今天實際的倉位 (由昨天信號決定)
            
            # 計算開盤時的 equity（用前一天收盤的 equity）
            if i == 0:
                opening_equity = initial_capital
            else:
                opening_equity = initial_capital * strategy_equity[i - 1]
            
            # 計算收盤時的 equity
            closing_equity = initial_capital * strategy_equity[i]
            
            # 計算可以買多少股（用開盤時的資金和開盤價）
            affordable_shares = int(opening_equity / open_price) if open_price > 0 else 0
            
            # 確定今天開盤時的操作（詳細版）
            action = "HOLD"
            if i > 0:
                yesterday_pos = effective_positions[i - 1]
                if today_pos != yesterday_pos:
                    if today_pos == 1:
                        # 買入
                        shares_held = affordable_shares
                        avg_cost = open_price
                        if yesterday_pos == 0:
                            action = f"BUY {shares_held} shares @ ${open_price:.2f}"
                        else:  # 從空頭轉多頭
                            action = f"COVER & BUY {shares_held} shares @ ${open_price:.2f}"
                    elif today_pos == -1:
                        # 做空
                        shares_held = -affordable_shares
                        avg_cost = open_price
                        if yesterday_pos == 0:
                            action = f"SHORT {abs(shares_held)} shares @ ${open_price:.2f}"
                        else:  # 從多頭轉空頭
                            action = f"SELL & SHORT {abs(shares_held)} shares @ ${open_price:.2f}"
                    else:
                        # 平倉
                        pnl = shares_held * (open_price - avg_cost) if shares_held > 0 else abs(shares_held) * (avg_cost - open_price)
                        action = f"CLOSE {abs(shares_held)} shares @ ${open_price:.2f} (PnL: ${pnl:.2f})"
                        shares_held = 0
                        avg_cost = 0.0
            elif today_pos != 0:
                # 第一天就有倉位
                shares_held = affordable_shares if today_pos == 1 else -affordable_shares
                avg_cost = open_price
                action = f"BUY {shares_held} shares @ ${open_price:.2f}" if today_pos == 1 else f"SHORT {abs(shares_held)} shares @ ${open_price:.2f}"
            
            # tomorrow_position: 明天的倉位 (由今天的 signal_discrete 決定)
            tomorrow_pos = signal_discrete
            
            trade_log.append({
                'date': date_str,
                'open_price': round(open_price, 2),
                'close_price': round(close_price, 2),
                'today_position': today_pos,              # T日實際倉位 (由T-1信號決定)
                'shares': shares_held,                    # 持股數量（負數表示做空）
                'action': action,                         # T日開盤操作（詳細版）
                'signal': round(signal_val, 4),           # T日收盤信號 (決定T+1倉位)
                'tomorrow_position': tomorrow_pos,        # T+1日倉位 (由今天信號決定)
                'daily_ret_pct': round(strategy_ret[i] * 100, 4),
                'open_equity': round(opening_equity, 2),  # 開盤時資金
                'close_equity': round(closing_equity, 2), # 收盤時資金
            })
        
        return {
            'ticker': ticker,
            'period': f"{dates.iloc[0].date()} ~ {dates.iloc[-1].date()}",
            'metrics': {
                'total_ret': total_ret,
                'ann_ret': ann_ret,
                'vol': vol,
                'sharpe': sharpe,
                'sortino': sortino,
                'max_dd': max_dd,
                'win_rate': win_rate,
            },
            # Buy & Hold 完整指標
            'bh_metrics': {
                'total_ret': bh_total_ret,
                'ann_ret': bh_ann_ret,
                'vol': bh_vol,
                'sharpe': bh_sharpe,
                'sortino': bh_sortino,
                'max_dd': bh_max_dd,
                'win_rate': bh_win_rate,
            },
            # 快捷訪問（兼容 summary 生成）
            'total_return': total_ret,
            'ann_return': ann_ret,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'long_days': int(long_days),
            'short_days': int(short_days),
            'hold_days': int(hold_days),
            # Buy & Hold 快捷訪問
            'bh_total_ret': bh_total_ret,
            'bh_ann_ret': bh_ann_ret,
            'bh_sharpe': bh_sharpe,
            'bh_sortino': bh_sortino,
            'bh_max_dd': bh_max_dd,
            'bh_win_rate': bh_win_rate,
            # 其他數據
            'equity': strategy_equity,
            'bh_equity': bh_equity,
            'positions': effective_positions,
            'signals': np.array(signals),  # 轉換為 numpy 數組以便繪圖
            'dates': dates,
            'close': close,
            'trade_log': trade_log,
            'trade_count': trade_count,
            'initial_capital': initial_capital,
            'final_value': final_value,
        }
    
    def plot_results(self, result: Dict, output_file: str = None):
        """繪製回測結果"""
        if output_file is None:
            output_file = os.path.join(OUTPUT_DIR, f"{result['ticker']}_backtest_v2.png")
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2.5, 1.5, 1.5]})
        
        dates = result['dates']
        equity = result['equity']
        bh_equity = result['bh_equity']
        positions = result['positions']
        signals = result.get('signals', None)
        
        # Plot 1: Equity curves
        ax1 = axes[0]
        ax1.plot(dates, equity, label=f"Strategy | Sharpe {result['metrics']['sharpe']:.2f}", 
                 linewidth=2, color='#2E86AB')
        ax1.plot(dates, bh_equity, label=f"Buy & Hold ({result['bh_total_ret']:.1%})", 
                 linewidth=1.5, alpha=0.7, color='#A23B72')
        ax1.fill_between(dates, equity, alpha=0.2, color='#2E86AB')
        ax1.set_title(f"{result['ticker']} - Realistic Backtest | {result['period']}", fontsize=14)
        ax1.set_ylabel('Cumulative Return')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Position + Signal（與 times_us_v2.py 一致）
        ax2 = axes[1]
        
        # 繪製 Signal 曲線（如果有的話）
        if signals is not None and len(signals) > 0:
            ax2.plot(dates, signals, color='#2E86AB', alpha=0.8, linewidth=1.2, label='Signal')
        
        # 繪製 Position 區域（半透明填充）
        ax2.fill_between(dates, positions, step='mid', alpha=0.3, color='#2E86AB', label='Position')
        
        # 基準線
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.axhline(y=1, color='green', linestyle=':', linewidth=0.5, alpha=0.5)
        ax2.axhline(y=-1, color='red', linestyle=':', linewidth=0.5, alpha=0.5)
        
        # 閾值線（顯示信號轉換為倉位的臨界點）
        threshold = self.signal_threshold
        ax2.axhline(y=threshold, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax2.axhline(y=-threshold, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        
        long_days = (positions == 1).sum()
        short_days = (positions == -1).sum()
        hold_days = (positions == 0).sum()
        hold_pct = hold_days / len(positions) * 100 if len(positions) > 0 else 0
        ax2.set_title(f'Position + Signal | L:{long_days} S:{short_days} H:{hold_days}({hold_pct:.0f}%)', fontsize=12)
        ax2.set_ylabel('Signal/Pos')
        ax2.set_ylim(-1.5, 1.5)
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Drawdown Comparison（策略 + Buy & Hold）
        ax3 = axes[2]
        
        # 策略回撤
        dd_strategy = 1 - equity / np.maximum.accumulate(equity)
        max_dd_strategy = np.max(dd_strategy)
        ax3.fill_between(dates, -dd_strategy * 100, alpha=0.4, color='#2E86AB', 
                         label=f'Strategy MaxDD: {max_dd_strategy:.1%}')
        
        # Buy & Hold 回撤
        dd_bh = 1 - bh_equity / np.maximum.accumulate(bh_equity)
        max_dd_bh = np.max(dd_bh)
        ax3.fill_between(dates, -dd_bh * 100, alpha=0.3, color='#A23B72', 
                         label=f'Buy & Hold MaxDD: {max_dd_bh:.1%}')
        
        ax3.set_title('Drawdown Comparison (%)', fontsize=12)
        ax3.set_ylabel('Drawdown %')
        ax3.set_xlabel('Date')
        ax3.legend(loc='lower left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        print(f"📈 Chart saved to: {output_file}")
        plt.close()
    
    def export_trade_log(self, result: Dict, output_file: str = None):
        """導出交易記錄"""
        if output_file is None:
            output_file = os.path.join(OUTPUT_DIR, f"{result['ticker']}_trade_log_v2.csv")
        
        df = pd.DataFrame(result['trade_log'])
        df.to_csv(output_file, index=False)
        print(f"📁 Trade log saved to: {output_file}")
        
        # 只顯示有交易的日子
        trades_only = df[df['action'] != 'HOLD']
        if len(trades_only) > 0:
            print(f"\n--- Trade Actions ({len(trades_only)} trades) ---")
            print(trades_only.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description='Realistic Strategy Backtester V2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
回測邏輯：
  T日收盤信號 → T+1開盤建倉 → 持有直到信號改變
  收益 = position × (Close[T] - Close[T-1]) / Close[T-1]

使用範例：
  python backtest_strategy_v2.py --strategy output/NVDA_xxx/best_strategy.json --tickers NVDA --period 3y
  python backtest_strategy_v2.py --strategy output/SPY_best_strategy.json --tickers SPY --start 2020-01-01 --end 2024-01-01
        """
    )
    parser.add_argument('--strategy', type=str, required=True,
                        help='Strategy JSON file path')
    parser.add_argument('--tickers', type=str, default='SPY',
                        help='Comma-separated list of tickers')
    parser.add_argument('--period', type=str, default='1y',
                        help='Backtest period (e.g., 1y, 2y, 6mo)')
    parser.add_argument('--capital', type=float, default=100000,
                        help='Initial capital')
    parser.add_argument('--start', type=str, default=None,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Override signal threshold')
    parser.add_argument('--hysteresis', type=float, default=None,
                        help='Override hysteresis for position changes (default: 0.05)')
    parser.add_argument('--export', action='store_true',
                        help='Export trade logs')
    parser.add_argument('--plot', action='store_true',
                        help='Generate charts')
    
    args = parser.parse_args()
    
    # 解析 tickers
    tickers = [t.strip().upper() for t in args.tickers.split(',')]
    
    # 初始化回測器
    backtester = RealisticBacktester(args.strategy, override_threshold=args.threshold, 
                                     override_hysteresis=args.hysteresis)
    
    # 創建輸出文件夾
    output_folder = create_backtest_folder(backtester.strategy_name, tickers)
    
    print("=" * 70)
    print("🚀 Realistic Strategy Backtester V2")
    print("=" * 70)
    print(f"Strategy: {args.strategy}")
    print(f"Tickers:  {', '.join(tickers)}")
    print(f"Period:   {args.period}")
    print(f"Capital:  ${args.capital:,.2f}")
    print(f"Output:   {output_folder}")
    print("=" * 70)
    
    # 收集所有結果
    all_results = []
    
    # 回測每個 ticker
    for ticker in tickers:
        if args.start and args.end:
            result = backtester.backtest(ticker, args.period, args.capital, args.start, args.end)
        else:
            result = backtester.backtest(ticker, args.period, args.capital)
        
        if result:
            all_results.append(result)
            
            if args.export:
                output_file = os.path.join(output_folder, f"{ticker}_trade_log.csv")
                backtester.export_trade_log(result, output_file)
            
            if args.plot:
                output_file = os.path.join(output_folder, f"{ticker}_backtest.png")
                backtester.plot_results(result, output_file)
    
    # 保存回測摘要
    if all_results:
        summary = {
            'strategy_file': args.strategy,
            'signal_threshold': float(backtester.signal_threshold),
            'tickers': tickers,
            'results': [{
                'ticker': r['ticker'],
                # Strategy metrics
                'total_return': float(r['total_return']),
                'ann_return': float(r['ann_return']),
                'sharpe': float(r['sharpe']),
                'sortino': float(r['sortino']),
                'max_drawdown': float(r['max_drawdown']),
                'win_rate': float(r['win_rate']),
                'long_days': int(r['long_days']),
                'short_days': int(r['short_days']),
                'hold_days': int(r['hold_days']),
                'trade_count': int(r['trade_count']),
                # Buy & Hold metrics
                'bh_total_return': float(r['bh_total_ret']),
                'bh_ann_return': float(r['bh_ann_ret']),
                'bh_sharpe': float(r['bh_sharpe']),
                'bh_sortino': float(r['bh_sortino']),
                'bh_max_drawdown': float(r['bh_max_dd']),
                'bh_win_rate': float(r['bh_win_rate']),
            } for r in all_results]
        }
        summary_file = os.path.join(output_folder, 'summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n📋 Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
