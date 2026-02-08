"""
Strategy Backtester - Test trained strategy on multiple tickers
支持離散倉位和連續倉位的比較

預測目標模式：
- T1OTC:  T+1 Open-to-Close (日內策略) - T日收盤信號 → T+1開盤買 → T+1收盤賣
- T1OT2O: T+1 Open-to-T+2 Open (隔夜策略) - T日收盤信號 → T+1開盤買 → T+2開盤賣

注意：用戶可以手動指定模式，允許使用 T1OTC 訓練的策略在 T1OT2O 模式下回測（反之亦然）

Usage:
    python backtest_strategy.py --strategy output/NVDA_T1OTC_xxx/best_strategy.json --tickers SPY,QQQ --period 1y
    python backtest_strategy.py --strategy output/SPY_best_strategy.json --tickers MSFT --mode T1OTC --period 2y
    python backtest_strategy.py --strategy output/NVDA_T1OTC_xxx/best_strategy.json --tickers NVDA --mode T1OT2O  # 跨模式測試
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Optional

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

# ==================== 預測目標模式 ====================
TARGET_MODE_T1OTC = 'T1OTC'    # T+1 Open → T+1 Close
TARGET_MODE_T1OT2O = 'T1OT2O'  # T+1 Open → T+2 Open
VALID_TARGET_MODES = [TARGET_MODE_T1OTC, TARGET_MODE_T1OT2O]

# ==================== 信號閾值配置 ====================
DEFAULT_SIGNAL_THRESHOLD = 0.1  # 默認閾值

def discretize_position(signal, threshold=DEFAULT_SIGNAL_THRESHOLD):
    """將連續信號 [-1, 1] 離散化為 {-1, 0, +1}"""
    pos = np.zeros_like(signal)
    pos[signal > threshold] = 1
    pos[signal < -threshold] = -1
    return pos

def get_mode_description(mode):
    """返回模式的詳細描述"""
    if mode == TARGET_MODE_T1OTC:
        return "T+1 Open-to-Close (日內策略: T日收盤信號 → T+1開盤買 → T+1收盤賣)"
    elif mode == TARGET_MODE_T1OT2O:
        return "T+1 Open-to-T+2 Open (隔夜策略: T日收盤信號 → T+1開盤買 → T+2開盤賣)"
    return "Unknown"

def create_backtest_folder(strategy_name, mode, tickers):
    """創建帶時間戳的回測輸出文件夾"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    ticker_str = "_".join(tickers[:3])  # 最多顯示3個ticker
    if len(tickers) > 3:
        ticker_str += f"_+{len(tickers)-3}"
    folder_name = f"backtest_{strategy_name}_{mode}_{ticker_str}_{timestamp}"
    folder_path = os.path.join(OUTPUT_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

# ==================== Operators (same as times_us.py) ====================
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

# ==================== Config ====================
FEATURES = ['RET', 'RET5', 'VOL_CHG', 'V_RET', 'TREND']

OPS_CONFIG = [
    ('ADD', lambda x, y: x + y, 2),
    ('SUB', lambda x, y: x - y, 2),
    ('MUL', lambda x, y: x * y, 2),
    ('DIV', lambda x, y: x / (y + 1e-6 * torch.sign(y)), 2),
    ('NEG', lambda x: -x, 1),
    ('ABS', lambda x: torch.abs(x), 1),
    ('SIGN', lambda x: torch.sign(x), 1),
    ('DELTA5', lambda x: _ts_delta(x, 5), 1),
    ('DELTA10', lambda x: _ts_delta(x, 10), 1),
    ('MA10', lambda x: _ts_decay_linear(x, 10), 1),
    ('MA20', lambda x: _ts_decay_linear(x, 20), 1),
    ('STD20', lambda x: _ts_zscore(x, 20), 1),
    ('MAX20', lambda x: _ts_max(x, 20), 1),
    ('MIN20', lambda x: _ts_min(x, 20), 1),
]

VOCAB = FEATURES + [cfg[0] for cfg in OPS_CONFIG]
OP_FUNC_MAP = {i + len(FEATURES): cfg[1] for i, cfg in enumerate(OPS_CONFIG)}
OP_ARITY_MAP = {i + len(FEATURES): cfg[2] for i, cfg in enumerate(OPS_CONFIG)}

COST_RATE = 0.0005  # Transaction cost


def calc_metrics(daily_ret, equity):
    """計算回測指標"""
    total_ret = equity[-1] - 1
    ann_ret = equity[-1] ** (252 / len(equity)) - 1
    vol = np.std(daily_ret) * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / (vol + 1e-6)
    
    dd = 1 - equity / np.maximum.accumulate(equity)
    max_dd = np.max(dd)
    calmar = ann_ret / (max_dd + 1e-6)
    
    win_rate = np.mean(daily_ret > 0)
    avg_win = np.mean(daily_ret[daily_ret > 0]) if np.any(daily_ret > 0) else 0
    avg_loss = np.abs(np.mean(daily_ret[daily_ret < 0])) if np.any(daily_ret < 0) else 1e-6
    profit_factor = avg_win / avg_loss
    
    return {
        'total_ret': total_ret,
        'ann_ret': ann_ret,
        'vol': vol,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'calmar': calmar,
        'win_rate': win_rate,
        'profit_factor': profit_factor
    }


class StrategyBacktester:
    """Strategy backtester with discrete and continuous position support"""
    
    def __init__(self, strategy_file: str, override_mode: str = None, override_threshold: float = None):
        """
        初始化回測器
        
        Args:
            strategy_file: 策略JSON文件路徑
            override_mode: 覆蓋策略文件中的模式（允許跨模式測試）
            override_threshold: 覆蓋策略文件中的閾值
        """
        with open(strategy_file, 'r') as f:
            strategy = json.load(f)
        
        self.formula_tokens = strategy['formula_tokens']
        self.formula_readable = strategy.get('formula_readable', self._decode_formula())
        self.strategy_file = strategy_file
        
        # 歸一化參數（從訓練時保存，確保回測一致性）
        self.norm_params = strategy.get('norm_params', None)
        if self.norm_params:
            print(f"✅ Using saved normalization params from training")
        else:
            print(f"⚠️  No norm_params found, will compute from backtest data (may differ from training!)")
        
        # 策略原始訓練模式
        self.strategy_mode = strategy.get('target_mode', TARGET_MODE_T1OT2O)
        
        # 策略原始閾值
        self.strategy_threshold = strategy.get('signal_threshold', DEFAULT_SIGNAL_THRESHOLD)
        
        # 實際使用的回測模式（可覆蓋）
        if override_mode:
            self.target_mode = override_mode
            if self.target_mode != self.strategy_mode:
                print(f"⚠️  Cross-mode testing: Strategy trained with {self.strategy_mode}, testing with {self.target_mode}")
        else:
            self.target_mode = self.strategy_mode
        
        # 實際使用的閾值（可覆蓋）
        if override_threshold is not None:
            self.signal_threshold = override_threshold
            if self.signal_threshold != self.strategy_threshold:
                print(f"⚠️  Threshold override: Strategy trained with {self.strategy_threshold}, testing with {self.signal_threshold}")
        else:
            self.signal_threshold = self.strategy_threshold
        
        print(f"✅ Loaded strategy: {self.formula_readable}")
        print(f"   Tokens: {self.formula_tokens}")
        print(f"   Strategy Mode: {self.strategy_mode}")
        print(f"   Backtest Mode: {self.target_mode} ({get_mode_description(self.target_mode)})")
        print(f"   Signal Threshold: ±{self.signal_threshold} (|signal| < {self.signal_threshold} → 觀望)")
    
    def _decode_formula(self) -> str:
        tokens = list(self.formula_tokens)
        def _parse():
            if not tokens: return ""
            t = tokens.pop(0)
            if t < len(FEATURES): return FEATURES[t]
            args = [_parse() for _ in range(OP_ARITY_MAP[t])]
            return f"{VOCAB[t]}({','.join(args)})"
        try:
            return _parse()
        except:
            return "Invalid"
    
    def _compute_features(self, df: pd.DataFrame) -> torch.Tensor:
        """Compute features from OHLCV data"""
        close = df['Close'].values.astype(np.float32)
        high = df['High'].values.astype(np.float32)
        low = df['Low'].values.astype(np.float32)
        vol = df['Volume'].values.astype(np.float32)
        
        # RET
        ret = np.zeros_like(close)
        ret[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-6)
        
        # RET5
        ret5 = pd.Series(close).pct_change(5).fillna(0).values.astype(np.float32)
        
        # VOL_CHG
        vol_ma = pd.Series(vol).rolling(20).mean().values
        vol_chg = np.zeros_like(vol)
        mask = vol_ma > 0
        vol_chg[mask] = vol[mask] / vol_ma[mask] - 1
        vol_chg = np.nan_to_num(vol_chg).astype(np.float32)
        
        # V_RET
        v_ret = (ret * (vol_chg + 1)).astype(np.float32)
        
        # TREND
        ma60 = pd.Series(close).rolling(60).mean().values
        trend = np.zeros_like(close)
        mask = ma60 > 0
        trend[mask] = close[mask] / ma60[mask] - 1
        trend = np.nan_to_num(trend).astype(np.float32)
        
        # Robust normalization - 使用訓練時保存的參數（如果有）
        def robust_norm(x, feature_name=None):
            x = x.astype(np.float32)
            # 如果有保存的歸一化參數，使用它們（保持與訓練一致）
            if self.norm_params and feature_name and feature_name in self.norm_params:
                median = self.norm_params[feature_name]['median']
                mad = self.norm_params[feature_name]['mad']
            else:
                # 否則從當前數據計算（可能與訓練不一致！）
                median = np.nanmedian(x)
                mad = np.nanmedian(np.abs(x - median)) + 1e-6
            res = (x - median) / mad
            return np.clip(res, -5, 5).astype(np.float32)
        
        features = torch.stack([
            torch.from_numpy(robust_norm(ret, 'ret')),
            torch.from_numpy(robust_norm(ret5, 'ret5')),
            torch.from_numpy(robust_norm(vol_chg, 'vol_chg')),
            torch.from_numpy(robust_norm(v_ret, 'v_ret')),
            torch.from_numpy(robust_norm(trend, 'trend')),
        ])
        
        return features
    
    def _execute_formula(self, features: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Execute formula on features
        
        注意：時序算子期望輸入是 [B, T] 形狀，所以需要 unsqueeze/squeeze 處理
        """
        stack = []
        try:
            for t in reversed(self.formula_tokens):
                if t < len(FEATURES):
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
            print(f"❌ Formula execution error: {e}")
        return None
    
    def backtest(
        self,
        ticker: str,
        period: str = "1y",
        initial_capital: float = 100000,
        start_date: str = None,
        end_date: str = None,
    ) -> Optional[Dict]:
        """
        Backtest strategy on a single ticker with both discrete and continuous positions
        """
        print(f"\n{'='*70}")
        print(f"📊 Backtesting {ticker}")
        print(f"{'='*70}")
        
        # Download data
        yf_ticker = yf.Ticker(ticker)
        
        if start_date and end_date:
            df = yf_ticker.history(start=start_date, end=end_date, auto_adjust=True)
        else:
            df = yf_ticker.history(period=period, auto_adjust=True)
        
        if df.empty or len(df) < 60:
            print(f"❌ Insufficient data for {ticker}")
            return None
        
        df = df.reset_index()
        dates = pd.to_datetime(df['Date'])
        close = df['Close'].values.astype(np.float32)
        
        print(f"📅 Data period: {dates.iloc[0].date()} ~ {dates.iloc[-1].date()}")
        print(f"📈 Total days: {len(df)}")
        
        # Compute features and execute formula
        features = self._compute_features(df)
        factor_values = self._execute_formula(features)
        
        if factor_values is None:
            print("❌ Formula execution failed")
            return None
        
        # Generate signals
        factor_np = factor_values.numpy()
        signal = np.tanh(factor_np)
        
        # Calculate returns based on target mode
        open_prices = df['Open'].values.astype(np.float32)
        close_prices = df['Close'].values.astype(np.float32)
        
        if self.target_mode == TARGET_MODE_T1OTC:
            # T+1 Open-to-Close (日內策略)
            next_open = np.roll(open_prices, -1)
            next_close = np.roll(close_prices, -1)
            target_ret = (next_close - next_open) / (next_open + 1e-6)
            target_ret[-1] = 0
            print(f"   Target: T+1 Open-to-Close (日內策略)")
        else:
            # T+1 Open-to-T+2 Open (隔夜策略)
            open_t1 = np.roll(open_prices, -1)
            open_t2 = np.roll(open_prices, -2)
            target_ret = (open_t2 - open_t1) / (open_t1 + 1e-6)
            target_ret[-2:] = 0
            print(f"   Target: T+1 Open-to-T+2 Open (隔夜策略)")
        
        # ========== 離散倉位 (Discrete) ==========
        # 使用閾值將信號離散化為 {-1, 0, +1}
        pos_discrete = discretize_position(signal, self.signal_threshold)
        turnover_d = np.abs(pos_discrete - np.roll(pos_discrete, 1))
        turnover_d[0] = np.abs(pos_discrete[0])
        daily_ret_d = pos_discrete * target_ret - turnover_d * COST_RATE
        equity_d = (1 + daily_ret_d).cumprod()
        metrics_d = calc_metrics(daily_ret_d, equity_d)
        
        # 計算倉位統計
        long_days = (pos_discrete == 1).sum()
        short_days = (pos_discrete == -1).sum()
        hold_days = (pos_discrete == 0).sum()
        hold_ratio = hold_days / len(pos_discrete) * 100
        
        # ========== 連續倉位 (Continuous) ==========
        pos_continuous = signal
        turnover_c = np.abs(pos_continuous - np.roll(pos_continuous, 1))
        turnover_c[0] = np.abs(pos_continuous[0])
        daily_ret_c = pos_continuous * target_ret - turnover_c * COST_RATE
        equity_c = (1 + daily_ret_c).cumprod()
        metrics_c = calc_metrics(daily_ret_c, equity_c)
        
        # Print comparison
        print("-" * 70)
        print(f"📊 Position Stats : {long_days} long | {short_days} short | {hold_days} hold ({hold_ratio:.1f}%)")
        print("-" * 70)
        print(f"{'指標':<18} {'離散倉位 (±1,0)':<20} {'連續倉位 (Signal)':<20}")
        print("-" * 70)
        print(f"{'📈 Total Return':<18} {metrics_d['total_ret']:>18.2%} {metrics_c['total_ret']:>18.2%}")
        print(f"{'📈 Ann. Return':<18} {metrics_d['ann_ret']:>18.2%} {metrics_c['ann_ret']:>18.2%}")
        print(f"{'📊 Ann. Volatility':<18} {metrics_d['vol']:>18.2%} {metrics_c['vol']:>18.2%}")
        print(f"{'⭐ Sharpe Ratio':<18} {metrics_d['sharpe']:>18.2f} {metrics_c['sharpe']:>18.2f}")
        print(f"{'📉 Max Drawdown':<18} {metrics_d['max_dd']:>18.2%} {metrics_c['max_dd']:>18.2%}")
        print(f"{'🎯 Calmar Ratio':<18} {metrics_d['calmar']:>18.2f} {metrics_c['calmar']:>18.2f}")
        print(f"{'✅ Win Rate':<18} {metrics_d['win_rate']:>18.2%} {metrics_c['win_rate']:>18.2%}")
        print(f"{'💰 Profit Factor':<18} {metrics_d['profit_factor']:>18.2f} {metrics_c['profit_factor']:>18.2f}")
        print("-" * 70)
        
        # Capital calculation (using discrete position)
        final_value_d = initial_capital * equity_d[-1]
        final_value_c = initial_capital * equity_c[-1]
        
        print(f"\n--- 資金變化 (初始: ${initial_capital:,.2f}) ---")
        print(f"離散倉位: ${final_value_d:,.2f} (P&L: ${final_value_d - initial_capital:,.2f})")
        print(f"連續倉位: ${final_value_c:,.2f} (P&L: ${final_value_c - initial_capital:,.2f})")
        
        # ========== Buy & Hold 基準線 (Close-to-Close) ==========
        # 用標準日收益率，而不是策略目標收益率
        benchmark_ret = np.zeros_like(close_prices)
        benchmark_ret[1:] = (close_prices[1:] - close_prices[:-1]) / (close_prices[:-1] + 1e-6)
        benchmark_equity = (1 + benchmark_ret).cumprod()
        benchmark_total_ret = benchmark_equity[-1] - 1
        
        print(f"📊 Buy & Hold     : {benchmark_total_ret:.2%} (標準 Close-to-Close)")
        
        # Generate trade log (for discrete position)
        trade_log = []
        for i in range(len(df)):
            action = "HOLD"
            
            if i == 0:
                # 第一天：根據初始倉位決定動作
                if pos_discrete[i] > 0:
                    action = "BUY"
                elif pos_discrete[i] < 0:
                    action = "SELL/SHORT"
                else:
                    action = "HOLD"
            else:
                # 之後的日子：根據倉位變化決定動作
                if pos_discrete[i] != pos_discrete[i-1]:
                    if pos_discrete[i] > 0:
                        if pos_discrete[i-1] < 0:
                            action = "CLOSE & BUY"  # 從做空轉做多
                        else:
                            action = "BUY"
                    elif pos_discrete[i] < 0:
                        if pos_discrete[i-1] > 0:
                            action = "CLOSE & SHORT"  # 從做多轉做空
                        else:
                            action = "SELL/SHORT"
                    else:
                        action = "CLOSE"
            
            trade_log.append({
                'date': dates.iloc[i].strftime('%Y-%m-%d'),
                'price': round(close[i], 2),
                'signal': round(signal[i], 4),
                'position_discrete': int(pos_discrete[i]),
                'position_continuous': round(pos_continuous[i], 4),
                'action': action,
            })
        
        trades_only = [t for t in trade_log if t['action'] != 'HOLD']
        print(f"\n📋 Total trades: {len(trades_only)}")
        
        return {
            'ticker': ticker,
            'target_mode': self.target_mode,
            'strategy_mode': self.strategy_mode,
            'period': f"{dates.iloc[0].date()} ~ {dates.iloc[-1].date()}",
            'discrete': {
                'metrics': metrics_d,
                'equity': equity_d,
                'position': pos_discrete,
                'final_value': final_value_d,
            },
            'continuous': {
                'metrics': metrics_c,
                'equity': equity_c,
                'position': pos_continuous,
                'final_value': final_value_c,
            },
            'dates': dates,
            'close': close,
            'signal': signal,
            'target_ret': target_ret,
            'benchmark_ret': benchmark_ret,
            'benchmark_equity': benchmark_equity,
            'benchmark_total_ret': benchmark_total_ret,
            'trade_log': trade_log,
            'trades_only': trades_only,
            'initial_capital': initial_capital,
        }
    
    def export_trade_log(self, result: Dict, output_folder: str = None):
        """Export trade log to CSV"""
        if output_folder is None:
            output_file = os.path.join(OUTPUT_DIR, f"{result['ticker']}_trade_log.csv")
        else:
            output_file = os.path.join(output_folder, f"{result['ticker']}_trade_log.csv")
        
        df = pd.DataFrame(result['trade_log'])
        df.to_csv(output_file, index=False)
        print(f"📁 Trade log saved to: {output_file}")
        
        # Print trades only
        if result['trades_only']:
            print(f"\n--- Trade Actions ({len(result['trades_only'])} trades) ---")
            trades_df = pd.DataFrame(result['trades_only'])
            print(trades_df.to_string(index=False))
    
    def plot_results(self, result: Dict, output_folder: str = None):
        """Plot backtest results with discrete and continuous comparison"""
        if output_folder is None:
            output_file = os.path.join(OUTPUT_DIR, f"{result['ticker']}_backtest.png")
        else:
            output_file = os.path.join(output_folder, f"{result['ticker']}_backtest.png")
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        dates = result['dates']
        equity_d = result['discrete']['equity']
        equity_c = result['continuous']['equity']
        benchmark = result['benchmark_equity']
        metrics_d = result['discrete']['metrics']
        metrics_c = result['continuous']['metrics']
        
        # Plot 1: Equity curves comparison
        ax1 = axes[0]
        ax1.plot(dates, equity_d, label=f'Discrete (±1) | Sharpe {metrics_d["sharpe"]:.2f}', 
                 linewidth=2, color='#2E86AB')
        ax1.plot(dates, equity_c, label=f'Continuous (Signal) | Sharpe {metrics_c["sharpe"]:.2f}', 
                 linewidth=2, color='#28A745', linestyle='--')
        bench_total = result.get('benchmark_total_ret', benchmark[-1] - 1)
        ax1.plot(dates, benchmark, label=f'{result["ticker"]} Buy & Hold ({bench_total:.1%})', 
                 linewidth=1.5, alpha=0.5, color='#A23B72')
        ax1.fill_between(dates, equity_d, alpha=0.2, color='#2E86AB')
        mode_str = result.get('target_mode', 'Unknown')
        ax1.set_title(f'{result["ticker"]} [{mode_str}] - Strategy Comparison | {result["period"]}', fontsize=14)
        ax1.set_ylabel('Cumulative Return')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Drawdown comparison
        ax2 = axes[1]
        dd_d = 1 - equity_d / np.maximum.accumulate(equity_d)
        dd_c = 1 - equity_c / np.maximum.accumulate(equity_c)
        ax2.fill_between(dates, -dd_d * 100, alpha=0.5, color='#2E86AB', 
                         label=f'Discrete DD (Max: {metrics_d["max_dd"]:.1%})')
        ax2.fill_between(dates, -dd_c * 100, alpha=0.5, color='#28A745', 
                         label=f'Continuous DD (Max: {metrics_c["max_dd"]:.1%})')
        ax2.set_title('Drawdown Comparison (%)', fontsize=12)
        ax2.set_ylabel('Drawdown %')
        ax2.legend(loc='lower left')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Position comparison
        ax3 = axes[2]
        ax3.plot(dates, result['discrete']['position'], label='Discrete Position', 
                 linewidth=1, color='#2E86AB', alpha=0.8)
        ax3.plot(dates, result['continuous']['position'], label='Continuous Position', 
                 linewidth=1, color='#28A745', alpha=0.8)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_title('Position Comparison', fontsize=12)
        ax3.set_ylabel('Position')
        ax3.set_xlabel('Date')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        print(f"📈 Chart saved to: {output_file}")
        plt.close()


def print_summary_table(results: Dict[str, Dict], target_mode: str):
    """Print summary table for multiple tickers"""
    print("\n" + "=" * 110)
    print(f"📊 BACKTEST SUMMARY [{target_mode}] (Discrete / Continuous)")
    print(f"   Mode: {get_mode_description(target_mode)}")
    print("=" * 110)
    print(f"{'Ticker':<8} {'Return(D)':<12} {'Return(C)':<12} {'Sharpe(D)':<10} {'Sharpe(C)':<10} "
          f"{'MaxDD(D)':<10} {'MaxDD(C)':<10} {'P&L(D)':<14}")
    print("-" * 110)
    
    for ticker, result in results.items():
        md = result['discrete']['metrics']
        mc = result['continuous']['metrics']
        pnl = result['discrete']['final_value'] - result['initial_capital']
        print(f"{ticker:<8} {md['total_ret']:>10.2%} {mc['total_ret']:>10.2%} "
              f"{md['sharpe']:>10.2f} {mc['sharpe']:>10.2f} "
              f"{md['max_dd']:>10.2%} {mc['max_dd']:>10.2%} "
              f"${pnl:>12,.2f}")
    
    print("=" * 110)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Backtest AlphaGPT Strategy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
預測目標模式說明：
  T1OTC   T+1 Open-to-Close (日內策略)
          T日收盤信號 → T+1開盤買 → T+1收盤賣
          
  T1OT2O  T+1 Open-to-T+2 Open (隔夜策略)
          T日收盤信號 → T+1開盤買 → T+2開盤賣

跨模式測試：
  允許使用 T1OTC 訓練的策略在 T1OT2O 模式下回測（反之亦然）
  例如：--mode T1OT2O 會覆蓋策略文件中的原始模式

使用範例：
  python backtest_strategy.py --strategy output/NVDA_T1OTC_xxx/best_strategy.json --tickers SPY,QQQ
  python backtest_strategy.py --strategy output/SPY_best_strategy.json --tickers MSFT --mode T1OTC
  python backtest_strategy.py --strategy output/NVDA_T1OTC_xxx/best_strategy.json --mode T1OT2O  # 跨模式測試
  python backtest_strategy.py --strategy output/xxx.json --threshold 0.2  # 調整閾值測試
        """
    )
    parser.add_argument('--strategy', type=str, required=True,
                        help='Strategy JSON file (e.g., output/SPY_T1OTC_xxx/best_strategy.json)')
    parser.add_argument('--tickers', type=str, default='SPY',
                        help='Comma-separated list of tickers (e.g., SPY,QQQ,AAPL)')
    parser.add_argument('--mode', type=str, default=None, choices=VALID_TARGET_MODES,
                        help=f'Override target mode: {TARGET_MODE_T1OTC}=日內策略, {TARGET_MODE_T1OT2O}=隔夜策略 (default: use strategy mode)')
    parser.add_argument('--threshold', type=float, default=None,
                        help=f'Override signal threshold (default: use strategy threshold). '
                             f'|signal| < threshold → hold (position=0)')
    parser.add_argument('--period', type=str, default='1y',
                        help='Backtest period (e.g., 1y, 2y, 6mo, ytd)')
    parser.add_argument('--capital', type=float, default=100000,
                        help='Initial capital (default: 100000)')
    parser.add_argument('--start', type=str, default=None,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--export', action='store_true',
                        help='Export trade logs to CSV')
    parser.add_argument('--plot', action='store_true',
                        help='Generate charts')
    
    args = parser.parse_args()
    
    # Parse tickers
    tickers = [t.strip().upper() for t in args.tickers.split(',')]
    
    # Load strategy and backtest (with optional mode and threshold override)
    backtester = StrategyBacktester(args.strategy, override_mode=args.mode, override_threshold=args.threshold)
    
    # Create output folder for this backtest run
    strategy_name = os.path.splitext(os.path.basename(args.strategy))[0]
    output_folder = create_backtest_folder(strategy_name, backtester.target_mode, tickers)
    
    print("=" * 70)
    print("🚀 AlphaGPT Strategy Backtester")
    print("=" * 70)
    print(f"Strategy      : {args.strategy}")
    print(f"Strategy Mode : {backtester.strategy_mode}")
    print(f"Backtest Mode : {backtester.target_mode}")
    print(f"              : {get_mode_description(backtester.target_mode)}")
    print(f"Threshold     : ±{backtester.signal_threshold} (|signal| < {backtester.signal_threshold} → 觀望)")
    print(f"Tickers       : {', '.join(tickers)}")
    print(f"Period        : {args.period}")
    print(f"Capital       : ${args.capital:,.2f}")
    print(f"Output Folder : {output_folder}")
    print("=" * 70)
    
    results = {}
    for ticker in tickers:
        if args.start and args.end:
            result = backtester.backtest(ticker, args.period, args.capital, args.start, args.end)
        else:
            result = backtester.backtest(ticker, args.period, args.capital)
        
        if result:
            results[ticker] = result
            
            if args.export:
                backtester.export_trade_log(result, output_folder)
            
            if args.plot:
                backtester.plot_results(result, output_folder)
    
    # Print summary
    if len(results) > 0:
        print_summary_table(results, backtester.target_mode)
        
        # Save summary to file
        summary_file = os.path.join(output_folder, 'summary.json')
        # 輔助函數：將 numpy 類型轉換為 Python 原生類型
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(v) for v in obj]
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        summary_data = {
            'strategy_file': args.strategy,
            'strategy_mode': backtester.strategy_mode,
            'backtest_mode': backtester.target_mode,
            'backtest_mode_description': get_mode_description(backtester.target_mode),
            'formula': backtester.formula_readable,
            'tickers': tickers,
            'period': args.period,
            'initial_capital': args.capital,
            'results': {}
        }
        for ticker, result in results.items():
            summary_data['results'][ticker] = {
                'period': result['period'],
                'discrete': convert_to_native(result['discrete']['metrics']),
                'continuous': convert_to_native(result['continuous']['metrics']),
                'final_value_discrete': float(result['discrete']['final_value']),
                'final_value_continuous': float(result['continuous']['final_value']),
            }
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"\n💾 Summary saved to: {summary_file}")
        print(f"✅ All outputs saved to: {output_folder}")
