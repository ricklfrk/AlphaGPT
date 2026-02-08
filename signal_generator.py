"""
AlphaGPT Signal Generator - 實時信號生成器

讀取訓練好的策略，生成買賣信號。

交易邏輯：
- T日收盤信號 → T+1開盤建倉 → 持有直到信號改變

Usage:
    # 查看今日信號
    python signal_generator.py --strategy output/SPY_xxx/best_strategy.json
    
    # 監控模式（每分鐘更新）
    python signal_generator.py --strategy output/SPY_best_strategy.json --monitor
    
    # 批量掃描多個標的
    python signal_generator.py --symbols SPY,QQQ,AAPL,MSFT,NVDA
"""

import json
import argparse
import os
import sys
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch
import yfinance as yf

# Windows 編碼修復
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# 輸出目錄
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 信號閾值配置 ====================
DEFAULT_SIGNAL_THRESHOLD = 0.1  # 默認閾值
DEFAULT_HYSTERESIS = 0.05       # 死區/滯後閾值，避免頻繁交易

def get_trade_instruction(direction):
    """根據方向返回具體的交易指令"""
    if direction == "BUY":
        return "明日開盤買入 → 持有直到信號改變"
    elif direction == "SELL":
        return "明日開盤做空 → 持有直到信號改變"
    else:
        return "觀望，不操作"

# ==================== 算子定義（與訓練時一致）====================
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SignalGenerator:
    """信號生成器 - 支持 V1 (5因子) 和 V2 (11因子) 策略"""
    
    def __init__(self, strategy_file=None, formula_tokens=None, override_threshold=None, override_hysteresis=None):
        """
        初始化信號生成器
        
        Args:
            strategy_file: 策略JSON文件路徑
            formula_tokens: 直接傳入公式token列表
            override_threshold: 覆蓋策略文件中的閾值
            override_hysteresis: 覆蓋策略文件中的滯後值
        """
        if strategy_file:
            with open(strategy_file, 'r') as f:
                strategy = json.load(f)
            
            # 檢測策略類型（單一公式 vs 集成策略）
            self.is_ensemble = strategy.get('type') == 'ensemble'
            
            if self.is_ensemble:
                # 集成策略：多個公式的加權組合
                self.formula_tokens_list = strategy['formula_tokens_list']
                self.formula_tokens = self.formula_tokens_list[0]  # 用第一個作為默認
                # 獲取每個組件的權重
                self.ensemble_weights = []
                if 'component_formulas' in strategy:
                    for comp in strategy['component_formulas']:
                        self.ensemble_weights.append(comp.get('weight', 1.0 / len(self.formula_tokens_list)))
                else:
                    self.ensemble_weights = [1.0 / len(self.formula_tokens_list)] * len(self.formula_tokens_list)
                self.ensemble_count = len(self.formula_tokens_list)
            else:
                # 單一公式策略
                self.formula_tokens = strategy['formula_tokens']
                self.is_ensemble = False
            
            self.formula_readable = strategy.get('formula_readable', 'N/A')
            self.symbol = strategy.get('symbol', 'Unknown')
            self.strategy_threshold = strategy.get('signal_threshold', DEFAULT_SIGNAL_THRESHOLD)
            self.strategy_hysteresis = strategy.get('hysteresis', DEFAULT_HYSTERESIS)
            # 歸一化參數（從訓練時保存）
            self.norm_params = strategy.get('norm_params', None)
            # 檢測策略版本
            self.version = strategy.get('version', '1.0')
        elif formula_tokens:
            self.formula_tokens = formula_tokens
            self.formula_readable = self._decode_formula(formula_tokens)
            self.symbol = 'Custom'
            self.strategy_threshold = DEFAULT_SIGNAL_THRESHOLD  # 默認閾值
            self.strategy_hysteresis = DEFAULT_HYSTERESIS
            self.norm_params = None
            self.version = '1.0'
            self.is_ensemble = False
        else:
            raise ValueError("Must provide either strategy_file or formula_tokens")
        
        # 根據版本選擇因子列表
        self.is_v2 = self.version.startswith('2')
        self.features = FEATURES if self.is_v2 else FEATURES_V1
        
        # SPY 和 VIX 數據（V2 策略需要）
        self.spy_data = None
        self.vix_data = None
        
        # 設置實際使用的閾值（可覆蓋）
        if override_threshold is not None:
            self.signal_threshold = override_threshold
            if self.signal_threshold != self.strategy_threshold:
                print(f"Warning: Threshold override: Strategy trained with {self.strategy_threshold}, using {self.signal_threshold}")
        else:
            self.signal_threshold = self.strategy_threshold
        
        # 設置滯後閾值（可覆蓋）
        if override_hysteresis is not None:
            self.hysteresis = override_hysteresis
        else:
            self.hysteresis = self.strategy_hysteresis
        
        # 追蹤前一天的倉位（用於滯後邏輯）
        self.last_position = 0
    
    def _decode_formula(self, tokens):
        """將token序列解碼為可讀公式"""
        stream = list(tokens)
        features = self.features if hasattr(self, 'features') else FEATURES
        def _parse():
            if not stream: return ""
            t = stream.pop(0)
            if t < len(features): return features[t]
            args = [_parse() for _ in range(OP_ARITY_MAP[t])]
            return f"{VOCAB[t]}({','.join(args)})"
        try: 
            return _parse()
        except: 
            return "Invalid"
    
    def _load_benchmark_data(self, dates):
        """加載 SPY 和 VIX 數據（V2 策略需要）"""
        if not self.is_v2:
            return
        
        start_date = dates.iloc[0] - timedelta(days=30)
        end_date = dates.iloc[-1] + timedelta(days=1)
        
        # 下載 SPY
        try:
            spy_ticker = yf.Ticker('SPY')
            spy_df = spy_ticker.history(start=start_date, end=end_date)
            if not spy_df.empty:
                spy_df = spy_df.reset_index()
                spy_df['Date'] = pd.to_datetime(spy_df['Date']).dt.tz_localize(None)
                self.spy_data = spy_df.set_index('Date')['Close']
        except Exception as e:
            self.spy_data = None
        
        # 下載 VIX
        try:
            vix_ticker = yf.Ticker('^VIX')
            vix_df = vix_ticker.history(start=start_date, end=end_date)
            if not vix_df.empty:
                vix_df = vix_df.reset_index()
                vix_df['Date'] = pd.to_datetime(vix_df['Date']).dt.tz_localize(None)
                self.vix_data = vix_df.set_index('Date')['Close']
        except Exception as e:
            self.vix_data = None
    
    def _compute_features(self, df):
        """計算因子特徵 - 支持 V1 (5因子) 和 V2 (11因子)"""
        close = df['Close'].values.astype(np.float32)
        open_ = df['Open'].values.astype(np.float32)
        high = df['High'].values.astype(np.float32)
        low = df['Low'].values.astype(np.float32)
        vol = df['Volume'].values.astype(np.float32)

        # ==================== 基礎因子 (V1 + V2) ====================
        # 1. 日收益率
        ret = np.zeros_like(close)
        ret[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-6)

        # 2. 5日收益率
        ret5 = pd.Series(close).pct_change(5).fillna(0).values.astype(np.float32)

        # 3. 成交量變化
        vol_ma = pd.Series(vol).rolling(20, min_periods=1).mean().values
        vol_chg = np.zeros_like(vol)
        mask = vol_ma > 0
        vol_chg[mask] = vol[mask] / vol_ma[mask] - 1
        vol_chg = np.nan_to_num(vol_chg).astype(np.float32)

        # 4. 量價收益
        v_ret = (ret * (vol_chg + 1)).astype(np.float32)

        # 5. 趨勢
        ma60 = pd.Series(close).rolling(60, min_periods=1).mean().values
        trend = np.zeros_like(close)
        mask = ma60 > 0
        trend[mask] = close[mask] / ma60[mask] - 1
        trend = np.nan_to_num(trend).astype(np.float32)

        # Robust Normalization
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
            ]).to(DEVICE)
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
        atr_norm = atr / (close + 1e-6)

        # RSI (Relative Strength Index)
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / (loss + 1e-6)
        rsi = (100 - 100 / (1 + rs)).fillna(50).values.astype(np.float32)
        rsi_norm = (rsi - 50) / 25.0

        # CLV (Close Location Value)
        hl_range = high - low + 1e-6
        clv = ((close - low) - (high - close)) / hl_range
        clv = np.nan_to_num(clv).astype(np.float32)

        # RS (Relative Strength vs SPY)
        if self.spy_data is not None:
            try:
                spy_aligned = self.spy_data.reindex(df.index).ffill().bfill().values.astype(np.float32)
                spy_ret = np.zeros_like(spy_aligned)
                spy_ret[1:] = (spy_aligned[1:] - spy_aligned[:-1]) / (spy_aligned[:-1] + 1e-6)
                rs_factor = ret - spy_ret
                rs_20 = pd.Series(rs_factor).rolling(20, min_periods=1).sum().values.astype(np.float32)
            except:
                rs_20 = np.zeros_like(close)
        else:
            rs_20 = np.zeros_like(close)

        # MOM (Momentum 20日)
        mom = pd.Series(close).pct_change(20).fillna(0).values.astype(np.float32)

        # VIX
        if self.vix_data is not None:
            try:
                vix_aligned = self.vix_data.reindex(df.index).ffill().bfill().values.astype(np.float32)
                vix_norm = (vix_aligned - 20) / 10
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
            torch.from_numpy(rsi_norm),
            torch.from_numpy(clv),
            torch.from_numpy(robust_norm(rs_20, 'rs')),
            torch.from_numpy(robust_norm(mom, 'mom')),
            torch.from_numpy(vix_norm),
        ]).to(DEVICE)
        
        return features
    
    def _execute_formula(self, features, formula_tokens=None):
        """
        執行公式計算因子值
        
        注意：時序算子期望輸入是 [B, T] 形狀，所以需要 unsqueeze/squeeze 處理
        
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
                    # 特徵是 [T] 形狀，unsqueeze 成 [1, T] 供時序算子使用
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
                # squeeze 回 [T] 形狀
                if final.dim() == 2 and final.shape[0] == 1:
                    final = final.squeeze(0)
                return final
        except Exception as e:
            print(f"Formula execution error: {e}")
        return None
    
    def _execute_ensemble_formula(self, features):
        """執行集成公式，返回加權平均的因子值序列"""
        if not self.is_ensemble:
            return self._execute_formula(features)
        
        all_values = []
        all_weights = []
        
        for i, tokens in enumerate(self.formula_tokens_list):
            values = self._execute_formula(features, tokens)
            if values is not None:
                all_values.append(values)
                all_weights.append(self.ensemble_weights[i])
        
        if len(all_values) == 0:
            return None
        
        # 加權平均
        total_weight = sum(all_weights)
        result = torch.zeros_like(all_values[0])
        for values, weight in zip(all_values, all_weights):
            result += values * (weight / total_weight)
        
        return result
    
    def _predict_next_open(self, df, factor_values, current_signal):
        """
        基於策略信號預測次日開盤價
        
        邏輯：
        1. 計算歷史上相同信號方向時的隔夜跳空幅度（Close-to-Open）
        2. 結合當前波動率和信號強度進行加權預測
        3. 提供預測置信度
        
        Args:
            df: 歷史數據 DataFrame
            factor_values: 因子值序列
            current_signal: 當前信號強度 [-1, 1]
            
        Returns:
            (predicted_open, expected_return, confidence)
        """
        close = df['Close'].values.astype(np.float32)
        open_prices = df['Open'].values.astype(np.float32)
        
        # 計算歷史隔夜跳空（Close_t -> Open_{t+1}）
        overnight_gaps = np.zeros(len(close) - 1)
        overnight_gaps = (open_prices[1:] - close[:-1]) / (close[:-1] + 1e-6)
        
        # 將歷史因子值轉為信號
        factor_np = factor_values.cpu().numpy() if hasattr(factor_values, 'cpu') else factor_values
        historical_signals = np.tanh(factor_np[:-1])  # 對齊到 overnight_gaps
        
        latest_close = close[-1]
        
        # ========== 方法1：基於信號方向的條件統計 ==========
        if current_signal > 0.1:
            # 看多信號：統計歷史上信號為正時的隔夜跳空
            mask = historical_signals > 0.1
            if mask.sum() > 10:
                relevant_gaps = overnight_gaps[mask]
                avg_gap = np.mean(relevant_gaps)
                std_gap = np.std(relevant_gaps)
            else:
                avg_gap = np.mean(overnight_gaps)
                std_gap = np.std(overnight_gaps)
        elif current_signal < -0.1:
            # 看空信號：統計歷史上信號為負時的隔夜跳空
            mask = historical_signals < -0.1
            if mask.sum() > 10:
                relevant_gaps = overnight_gaps[mask]
                avg_gap = np.mean(relevant_gaps)
                std_gap = np.std(relevant_gaps)
            else:
                avg_gap = np.mean(overnight_gaps)
                std_gap = np.std(overnight_gaps)
        else:
            # 中性信號：使用全體樣本
            avg_gap = np.mean(overnight_gaps)
            std_gap = np.std(overnight_gaps)
        
        # ========== 方法2：結合ATR波動率進行調整 ==========
        # 計算近期ATR（14日）
        high = df['High'].values.astype(np.float32)
        low = df['Low'].values.astype(np.float32)
        tr1 = high - low
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        atr_14 = np.mean(tr[-14:])
        atr_pct = atr_14 / latest_close  # ATR 佔收盤價的百分比
        
        # ========== 綜合預測 ==========
        # 預期收益率 = 歷史均值 + 信號強度 × ATR調整因子
        # 信號越強，預期波動越接近ATR方向
        signal_adjustment = current_signal * atr_pct * 0.3  # 保守係數 0.3
        expected_return = avg_gap + signal_adjustment
        
        # 預測次日開盤價
        predicted_open = latest_close * (1 + expected_return)
        
        # ========== 置信度計算 ==========
        # 基於：1. 歷史樣本量  2. 信號強度  3. 波動穩定性
        sample_size_factor = min(1.0, len(overnight_gaps) / 100)  # 樣本量
        signal_factor = abs(current_signal)  # 信號強度
        stability_factor = 1 - min(1.0, std_gap / (abs(avg_gap) + 1e-6) / 5)  # 波動穩定性
        
        confidence = (sample_size_factor * 0.3 + signal_factor * 0.4 + stability_factor * 0.3)
        confidence = max(0.1, min(0.95, confidence))  # 限制在 10%~95%
        
        return round(predicted_open, 2), round(expected_return * 100, 3), round(confidence * 100, 1)
    
    def get_signal(self, symbol, lookback_days=120):
        """
        獲取單個標的的信號
        
        Args:
            symbol: 股票代碼
            lookback_days: 回看天數（需要足夠的歷史數據計算因子）
            
        Returns:
            dict: 包含信號和相關信息
        """
        # 下載數據
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 30)  # 額外buffer
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
        
        if df.empty or len(df) < 60:
            return {
                'symbol': symbol,
                'error': 'Insufficient data',
                'signal': None
            }
        
        # 處理時區
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df = df.set_index('Date')
        
        # 為 V2 策略加載 SPY 和 VIX 數據
        if self.is_v2:
            self._load_benchmark_data(df.index.to_series())
        
        # 計算特徵
        features = self._compute_features(df)
        
        # 執行公式（根據策略類型選擇）
        if self.is_ensemble:
            factor_values = self._execute_ensemble_formula(features)
        else:
            factor_values = self._execute_formula(features)
        
        if factor_values is None:
            return {
                'symbol': symbol,
                'error': 'Formula execution failed',
                'signal': None
            }
        
        # 獲取最新信號
        latest_factor = factor_values[-1].item()
        signal_strength = np.tanh(latest_factor)  # 映射到 [-1, 1]
        
        # 確定方向（使用滯後閾值，與訓練/回測保持一致）
        threshold = self.signal_threshold
        hysteresis = self.hysteresis
        close_threshold = threshold - hysteresis  # 平倉閾值更低
        
        prev_pos = self.last_position
        
        if prev_pos == 0:
            # 當前無倉位：使用開倉閾值
            if signal_strength > threshold:
                position = 1
            elif signal_strength < -threshold:
                position = -1
            else:
                position = 0
        elif prev_pos == 1:
            # 當前持多倉（與 times_us_v2.py 的 discretize_position 一致）
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
        
        # 更新倉位記錄
        self.last_position = position
        
        # 確定方向標籤
        if position == 1:
            direction = "BUY"
            color = "🟢"
        elif position == -1:
            direction = "SELL"
            color = "🔴"
        else:
            direction = "HOLD"
            color = "⚪"
        
        # 獲取交易指令
        trade_instruction = get_trade_instruction(direction)
        
        # 獲取最新價格
        latest_close = df['Close'].iloc[-1]
        latest_open = df['Open'].iloc[-1]
        latest_date = df.index[-1]
        
        # 計算近期趨勢
        ret_1d = (df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100
        ret_5d = (df['Close'].iloc[-1] / df['Close'].iloc[-6] - 1) * 100 if len(df) > 5 else 0
        
        # ========== 預測次日開盤價 ==========
        predicted_open, expected_ret, confidence = self._predict_next_open(
            df, factor_values, signal_strength
        )
        
        return {
            'symbol': symbol,
            'date': latest_date.strftime('%Y-%m-%d'),
            'price': latest_close,
            'open': latest_open,
            'factor_value': latest_factor,
            'signal_strength': signal_strength,
            'signal_threshold': threshold,
            'direction': direction,
            'color': color,
            'ret_1d': ret_1d,
            'ret_5d': ret_5d,
            # 交易指令
            'trade_instruction': trade_instruction,
            # 次日開盤預測
            'predicted_open': predicted_open,
            'expected_return': expected_ret,
            'prediction_confidence': confidence,
            'error': None
        }
    
    def scan_multiple(self, symbols):
        """
        掃描多個標的
        
        Args:
            symbols: 股票代碼列表
            
        Returns:
            list: 信號列表
        """
        results = []
        for symbol in symbols:
            print(f"  Scanning {symbol}...", end=" ")
            result = self.get_signal(symbol)
            if result['error']:
                print(f"❌ {result['error']}")
            else:
                print(f"{result['color']} {result['direction']} ({result['signal_strength']:.2f})")
            results.append(result)
        return results


def print_signal_report(results, formula_readable, signal_threshold):
    """打印信號報告"""
    print("\n" + "="*95)
    print("📊 ALPHAGPT SIGNAL REPORT")
    print("="*95)
    print(f"📜 Formula: {formula_readable}")
    print(f"📊 Threshold: ±{signal_threshold} (|signal| < {signal_threshold} → HOLD)")
    print(f"⏰ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*95)
    print(f"{'Symbol':<8} {'Close':>10} {'Signal':>8} {'Strength':>8} {'1D':>7} {'5D':>7} {'NextOpen':>10} {'ExpRet':>8}")
    print("-"*95)
    
    # 按信號強度排序
    sorted_results = sorted(
        [r for r in results if r['error'] is None],
        key=lambda x: x['signal_strength'],
        reverse=True
    )
    
    for r in sorted_results:
        pred_open = r.get('predicted_open', 0)
        exp_ret = r.get('expected_return', 0)
        print(f"{r['color']} {r['symbol']:<6} {r['price']:>10.2f} {r['direction']:>8} "
              f"{r['signal_strength']:>8.2f} {r['ret_1d']:>6.1f}% {r['ret_5d']:>6.1f}% "
              f"{pred_open:>10.2f} {exp_ret:>7.2f}%")
    
    # 錯誤的標的
    errors = [r for r in results if r['error'] is not None]
    if errors:
        print("-"*95)
        print("⚠️ Errors:")
        for r in errors:
            print(f"   {r['symbol']}: {r['error']}")
    
    print("="*95)
    
    # 總結
    buys = [r for r in sorted_results if r['direction'] == 'BUY']
    sells = [r for r in sorted_results if r['direction'] == 'SELL']
    
    print(f"\n📈 Summary: {len(buys)} BUY | {len(sells)} SELL | {len(sorted_results) - len(buys) - len(sells)} HOLD")
    
    if buys:
        print(f"\n🟢 Top BUY signals:")
        for r in buys[:3]:
            pred_open = r.get('predicted_open', 0)
            exp_ret = r.get('expected_return', 0)
            conf = r.get('prediction_confidence', 0)
            instruction = r.get('trade_instruction', '')
            print(f"   {r['symbol']}: Close=${r['price']:.2f} → NextOpen=${pred_open:.2f} "
                  f"(ExpRet: {exp_ret:+.2f}%, Conf: {conf:.0f}%)")
            print(f"      📋 {instruction}")
    
    if sells:
        print(f"\n🔴 Top SELL signals:")
        for r in sells[:3]:
            pred_open = r.get('predicted_open', 0)
            exp_ret = r.get('expected_return', 0)
            conf = r.get('prediction_confidence', 0)
            instruction = r.get('trade_instruction', '')
            print(f"   {r['symbol']}: Close=${r['price']:.2f} → NextOpen=${pred_open:.2f} "
                  f"(ExpRet: {exp_ret:+.2f}%, Conf: {conf:.0f}%)")
            print(f"      📋 {instruction}")


def monitor_mode(generator, symbols, interval=60):
    """監控模式：定期更新信號"""
    print(f"\n🔄 Monitor mode started (updating every {interval}s)")
    print(f"   Threshold: ±{generator.signal_threshold}")
    print("   Press Ctrl+C to stop\n")
    
    try:
        while True:
            results = generator.scan_multiple(symbols)
            print_signal_report(results, generator.formula_readable, generator.signal_threshold)
            print(f"\n⏳ Next update in {interval} seconds...")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n👋 Monitor stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='AlphaGPT Signal Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
交易邏輯：
  T日收盤信號 → T+1開盤建倉 → 持有直到信號改變

使用範例：
  python signal_generator.py --strategy output/SPY_xxx/best_strategy.json
  python signal_generator.py --symbols SPY,QQQ,NVDA --monitor
  python signal_generator.py --strategy output/xxx.json --threshold 0.2  # 調整閾值
        """
    )
    parser.add_argument('--strategy', type=str, default=None,
                        help='Strategy JSON file path')
    parser.add_argument('--symbols', type=str, default=None,
                        help='Comma-separated list of symbols to scan')
    parser.add_argument('--threshold', type=float, default=None,
                        help=f'Override signal threshold (default: use strategy threshold). '
                             f'|signal| < threshold → HOLD')
    parser.add_argument('--hysteresis', type=float, default=None,
                        help=f'Override hysteresis for position changes (default: {DEFAULT_HYSTERESIS}). '
                             f'Close threshold = threshold - hysteresis')
    parser.add_argument('--monitor', action='store_true',
                        help='Enable monitor mode (continuous updates)')
    parser.add_argument('--interval', type=int, default=60,
                        help='Update interval in seconds (monitor mode)')
    args = parser.parse_args()
    
    # 確定要掃描的標的
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    else:
        # 默認掃描的標的
        symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMD', 'GOOGL', 'META']
    
    # 加載策略
    if args.strategy:
        # 如果路徑不存在，嘗試在 output/ 目錄下查找
        strategy_path = args.strategy
        if not os.path.exists(strategy_path):
            alt_path = os.path.join(OUTPUT_DIR, args.strategy)
            if os.path.exists(alt_path):
                strategy_path = alt_path
        
        generator = SignalGenerator(strategy_file=strategy_path, override_threshold=args.threshold, override_hysteresis=args.hysteresis)
        if generator.is_ensemble:
            print(f"✅ Loaded ENSEMBLE strategy from {strategy_path} ({generator.ensemble_count} components)")
        else:
            print(f"✅ Loaded strategy from {strategy_path}")
    else:
        # 嘗試查找 output/ 目錄下最新的策略文件（包括子目錄）
        strategy_files = []
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for f in files:
                if f == 'best_strategy.json' or f.endswith('_best_strategy.json'):
                    strategy_files.append(os.path.join(root, f))
        
        if strategy_files:
            # 按修改時間排序，選擇最新的
            latest_strategy = max(strategy_files, key=os.path.getmtime)
            generator = SignalGenerator(strategy_file=latest_strategy, override_threshold=args.threshold, override_hysteresis=args.hysteresis)
            if generator.is_ensemble:
                print(f"✅ Auto-loaded latest ENSEMBLE strategy: {latest_strategy} ({generator.ensemble_count} components)")
            else:
                print(f"✅ Auto-loaded latest strategy: {latest_strategy}")
        else:
            # 如果沒有策略文件，使用默認公式
            default_formula = [10, 0]  # MA20(RET)
            generator = SignalGenerator(formula_tokens=default_formula, override_threshold=args.threshold, override_hysteresis=args.hysteresis)
            print("⚠️ No strategy file found, using default formula: MA20(RET)")
    
    print(f"📜 Formula: {generator.formula_readable}")
    print(f"📊 Threshold    : ±{generator.signal_threshold} (|signal| < {generator.signal_threshold} → HOLD)")
    print(f"📋 Scanning: {', '.join(symbols)}")
    
    if args.monitor:
        monitor_mode(generator, symbols, args.interval)
    else:
        print("\n🔍 Generating signals...")
        results = generator.scan_multiple(symbols)
        print_signal_report(results, generator.formula_readable, generator.signal_threshold)
