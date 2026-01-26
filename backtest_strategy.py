"""
Strategy Backtester - Test trained strategy on multiple tickers
支持離散倉位和連續倉位的比較

Usage:
    python backtest_strategy.py --strategy output/NVDA_best_strategy.json --tickers SPY,QQQ --period 1y
    python backtest_strategy.py --strategy output/SPY_best_strategy.json --tickers MSFT --period 2y --capital 50000
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
FEATURES = ['RET', 'RET5', 'RET20', 'VOL_CHG', 'V_RET', 'TREND', 'ATR', 'RSI']

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
    
    def __init__(self, strategy_file: str):
        with open(strategy_file, 'r') as f:
            strategy = json.load(f)
        
        self.formula_tokens = strategy['formula_tokens']
        self.formula_readable = strategy.get('formula_readable', self._decode_formula())
        self.strategy_file = strategy_file
        
        print(f"✅ Loaded strategy: {self.formula_readable}")
        print(f"   Tokens: {self.formula_tokens}")
    
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
        
        # RET5, RET20
        ret5 = pd.Series(close).pct_change(5).fillna(0).values.astype(np.float32)
        ret20 = pd.Series(close).pct_change(20).fillna(0).values.astype(np.float32)
        
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
        
        # ATR
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        atr = pd.Series(tr).rolling(14).mean().fillna(0).values.astype(np.float32)
        atr_norm = atr / (close + 1e-6)
        
        # RSI
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-6)
        rsi = (100 - 100 / (1 + rs)).fillna(50).values.astype(np.float32)
        rsi_norm = (rsi - 50) / 50
        
        # Robust normalization
        def robust_norm(x):
            x = x.astype(np.float32)
            median = np.nanmedian(x)
            mad = np.nanmedian(np.abs(x - median)) + 1e-6
            res = (x - median) / mad
            return np.clip(res, -5, 5).astype(np.float32)
        
        features = torch.stack([
            torch.from_numpy(robust_norm(ret)),
            torch.from_numpy(robust_norm(ret5)),
            torch.from_numpy(robust_norm(ret20)),
            torch.from_numpy(robust_norm(vol_chg)),
            torch.from_numpy(robust_norm(v_ret)),
            torch.from_numpy(robust_norm(trend)),
            torch.from_numpy(robust_norm(atr_norm)),
            torch.from_numpy(rsi_norm),
        ])
        
        return features
    
    def _execute_formula(self, features: torch.Tensor) -> Optional[torch.Tensor]:
        """Execute formula on features"""
        stack = []
        try:
            for t in reversed(self.formula_tokens):
                if t < len(FEATURES):
                    stack.append(features[t])
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
                return stack[-1]
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
        
        # Calculate returns (Open-to-Open)
        open_prices = df['Open'].values.astype(np.float32)
        open_t1 = np.roll(open_prices, -1)
        open_t2 = np.roll(open_prices, -2)
        target_ret = (open_t2 - open_t1) / (open_t1 + 1e-6)
        target_ret[-2:] = 0
        
        # ========== 離散倉位 (Discrete) ==========
        pos_discrete = np.sign(signal)
        turnover_d = np.abs(pos_discrete - np.roll(pos_discrete, 1))
        turnover_d[0] = np.abs(pos_discrete[0])
        daily_ret_d = pos_discrete * target_ret - turnover_d * COST_RATE
        equity_d = (1 + daily_ret_d).cumprod()
        metrics_d = calc_metrics(daily_ret_d, equity_d)
        
        # ========== 連續倉位 (Continuous) ==========
        pos_continuous = signal
        turnover_c = np.abs(pos_continuous - np.roll(pos_continuous, 1))
        turnover_c[0] = np.abs(pos_continuous[0])
        daily_ret_c = pos_continuous * target_ret - turnover_c * COST_RATE
        equity_c = (1 + daily_ret_c).cumprod()
        metrics_c = calc_metrics(daily_ret_c, equity_c)
        
        # Print comparison
        print("-" * 70)
        print(f"{'指標':<18} {'離散倉位 (±1)':<20} {'連續倉位 (Signal)':<20}")
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
            'benchmark_equity': (1 + target_ret).cumprod(),
            'trade_log': trade_log,
            'trades_only': trades_only,
            'initial_capital': initial_capital,
        }
    
    def export_trade_log(self, result: Dict, output_file: str = None):
        """Export trade log to CSV"""
        if output_file is None:
            output_file = os.path.join(OUTPUT_DIR, f"{result['ticker']}_trade_log.csv")
        
        df = pd.DataFrame(result['trade_log'])
        df.to_csv(output_file, index=False)
        print(f"📁 Trade log saved to: {output_file}")
        
        # Print trades only
        if result['trades_only']:
            print(f"\n--- Trade Actions ({len(result['trades_only'])} trades) ---")
            trades_df = pd.DataFrame(result['trades_only'])
            print(trades_df.to_string(index=False))
    
    def plot_results(self, result: Dict, output_file: str = None):
        """Plot backtest results with discrete and continuous comparison"""
        if output_file is None:
            output_file = os.path.join(OUTPUT_DIR, f"{result['ticker']}_backtest.png")
        
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
        ax1.plot(dates, benchmark, label=f'{result["ticker"]} Buy & Hold', 
                 linewidth=1.5, alpha=0.5, color='#A23B72')
        ax1.fill_between(dates, equity_d, alpha=0.2, color='#2E86AB')
        ax1.set_title(f'{result["ticker"]} - Strategy Comparison | {result["period"]}', fontsize=14)
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


def print_summary_table(results: Dict[str, Dict]):
    """Print summary table for multiple tickers"""
    print("\n" + "=" * 110)
    print("📊 BACKTEST SUMMARY (Discrete / Continuous)")
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
    parser = argparse.ArgumentParser(description='Backtest AlphaGPT Strategy')
    parser.add_argument('--strategy', type=str, required=True,
                        help='Strategy JSON file (e.g., output/SPY_best_strategy.json)')
    parser.add_argument('--tickers', type=str, default='SPY',
                        help='Comma-separated list of tickers (e.g., SPY,QQQ,AAPL)')
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
    
    print("=" * 70)
    print("🚀 AlphaGPT Strategy Backtester")
    print("=" * 70)
    print(f"Strategy   : {args.strategy}")
    print(f"Tickers    : {', '.join(tickers)}")
    print(f"Period     : {args.period}")
    print(f"Capital    : ${args.capital:,.2f}")
    print(f"Output Dir : {OUTPUT_DIR}/")
    print("=" * 70)
    
    # Load strategy and backtest
    backtester = StrategyBacktester(args.strategy)
    
    results = {}
    for ticker in tickers:
        if args.start and args.end:
            result = backtester.backtest(ticker, args.period, args.capital, args.start, args.end)
        else:
            result = backtester.backtest(ticker, args.period, args.capital)
        
        if result:
            results[ticker] = result
            
            if args.export:
                backtester.export_trade_log(result)
            
            if args.plot:
                backtester.plot_results(result)
    
    # Print summary
    if len(results) > 0:
        print_summary_table(results)
