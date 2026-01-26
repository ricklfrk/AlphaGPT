"""
AlphaGPT Signal Generator - 實時信號生成器

讀取訓練好的策略，生成買賣信號。

Usage:
    # 查看今日信號
    python signal_generator.py --strategy output/SPY_best_strategy.json
    
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


# ==================== 配置 ====================
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SignalGenerator:
    """信號生成器"""
    
    def __init__(self, strategy_file=None, formula_tokens=None):
        """
        初始化信號生成器
        
        Args:
            strategy_file: 策略JSON文件路徑
            formula_tokens: 直接傳入公式token列表
        """
        if strategy_file:
            with open(strategy_file, 'r') as f:
                strategy = json.load(f)
            self.formula_tokens = strategy['formula_tokens']
            self.formula_readable = strategy.get('formula_readable', 'N/A')
            self.symbol = strategy.get('symbol', 'Unknown')
        elif formula_tokens:
            self.formula_tokens = formula_tokens
            self.formula_readable = self._decode_formula(formula_tokens)
            self.symbol = 'Custom'
        else:
            raise ValueError("Must provide either strategy_file or formula_tokens")
    
    def _decode_formula(self, tokens):
        """將token序列解碼為可讀公式"""
        stream = list(tokens)
        def _parse():
            if not stream: return ""
            t = stream.pop(0)
            if t < len(FEATURES): return FEATURES[t]
            args = [_parse() for _ in range(OP_ARITY_MAP[t])]
            return f"{VOCAB[t]}({','.join(args)})"
        try: 
            return _parse()
        except: 
            return "Invalid"
    
    def _compute_features(self, df):
        """計算因子特徵"""
        close = df['Close'].values.astype(np.float32)
        open_ = df['Open'].values.astype(np.float32)
        high = df['High'].values.astype(np.float32)
        low = df['Low'].values.astype(np.float32)
        vol = df['Volume'].values.astype(np.float32)

        # 1. 日收益率
        ret = np.zeros_like(close)
        ret[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-6)

        # 2. 5日收益率
        ret5 = pd.Series(close).pct_change(5).fillna(0).values.astype(np.float32)
        
        # 3. 20日收益率
        ret20 = pd.Series(close).pct_change(20).fillna(0).values.astype(np.float32)

        # 4. 成交量變化
        vol_ma = pd.Series(vol).rolling(20).mean().values
        vol_chg = np.zeros_like(vol)
        mask = vol_ma > 0
        vol_chg[mask] = vol[mask] / vol_ma[mask] - 1
        vol_chg = np.nan_to_num(vol_chg).astype(np.float32)

        # 5. 量價收益
        v_ret = (ret * (vol_chg + 1)).astype(np.float32)

        # 6. 趨勢
        ma60 = pd.Series(close).rolling(60).mean().values
        trend = np.zeros_like(close)
        mask = ma60 > 0
        trend[mask] = close[mask] / ma60[mask] - 1
        trend = np.nan_to_num(trend).astype(np.float32)

        # 7. ATR
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        atr = pd.Series(tr).rolling(14).mean().fillna(0).values.astype(np.float32)
        atr_norm = atr / (close + 1e-6)

        # 8. RSI
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-6)
        rsi = (100 - 100 / (1 + rs)).fillna(50).values.astype(np.float32)
        rsi_norm = (rsi - 50) / 50

        # Robust Normalization
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
        ]).to(DEVICE)
        
        return features
    
    def _execute_formula(self, features):
        """執行公式計算因子值"""
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
            print(f"Formula execution error: {e}")
        return None
    
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
        
        # 計算特徵
        features = self._compute_features(df)
        
        # 執行公式
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
        
        # 確定方向
        if signal_strength > 0.3:
            direction = "BUY"
            color = "🟢"
        elif signal_strength < -0.3:
            direction = "SELL"
            color = "🔴"
        else:
            direction = "HOLD"
            color = "⚪"
        
        # 獲取最新價格
        latest_price = df['Close'].iloc[-1]
        latest_date = df.index[-1]
        
        # 計算近期趨勢
        ret_1d = (df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100
        ret_5d = (df['Close'].iloc[-1] / df['Close'].iloc[-6] - 1) * 100 if len(df) > 5 else 0
        
        return {
            'symbol': symbol,
            'date': latest_date.strftime('%Y-%m-%d'),
            'price': latest_price,
            'factor_value': latest_factor,
            'signal_strength': signal_strength,
            'direction': direction,
            'color': color,
            'ret_1d': ret_1d,
            'ret_5d': ret_5d,
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


def print_signal_report(results, formula_readable):
    """打印信號報告"""
    print("\n" + "="*70)
    print("📊 ALPHAGPT SIGNAL REPORT")
    print("="*70)
    print(f"📜 Formula: {formula_readable}")
    print(f"⏰ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*70)
    print(f"{'Symbol':<8} {'Price':>10} {'Signal':>8} {'Strength':>10} {'1D':>8} {'5D':>8}")
    print("-"*70)
    
    # 按信號強度排序
    sorted_results = sorted(
        [r for r in results if r['error'] is None],
        key=lambda x: x['signal_strength'],
        reverse=True
    )
    
    for r in sorted_results:
        print(f"{r['color']} {r['symbol']:<6} {r['price']:>10.2f} {r['direction']:>8} "
              f"{r['signal_strength']:>10.2f} {r['ret_1d']:>7.1f}% {r['ret_5d']:>7.1f}%")
    
    # 錯誤的標的
    errors = [r for r in results if r['error'] is not None]
    if errors:
        print("-"*70)
        print("⚠️ Errors:")
        for r in errors:
            print(f"   {r['symbol']}: {r['error']}")
    
    print("="*70)
    
    # 總結
    buys = [r for r in sorted_results if r['direction'] == 'BUY']
    sells = [r for r in sorted_results if r['direction'] == 'SELL']
    
    print(f"\n📈 Summary: {len(buys)} BUY | {len(sells)} SELL | {len(sorted_results) - len(buys) - len(sells)} HOLD")
    
    if buys:
        print(f"\n🟢 Top BUY signals:")
        for r in buys[:3]:
            print(f"   {r['symbol']}: strength={r['signal_strength']:.2f}, price=${r['price']:.2f}")
    
    if sells:
        print(f"\n🔴 Top SELL signals:")
        for r in sells[:3]:
            print(f"   {r['symbol']}: strength={r['signal_strength']:.2f}, price=${r['price']:.2f}")


def monitor_mode(generator, symbols, interval=60):
    """監控模式：定期更新信號"""
    print(f"\n🔄 Monitor mode started (updating every {interval}s)")
    print("   Press Ctrl+C to stop\n")
    
    try:
        while True:
            results = generator.scan_multiple(symbols)
            print_signal_report(results, generator.formula_readable)
            print(f"\n⏳ Next update in {interval} seconds...")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n👋 Monitor stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AlphaGPT Signal Generator')
    parser.add_argument('--strategy', type=str, default=None,
                        help='Strategy JSON file path')
    parser.add_argument('--symbols', type=str, default=None,
                        help='Comma-separated list of symbols to scan')
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
        
        generator = SignalGenerator(strategy_file=strategy_path)
        print(f"✅ Loaded strategy from {strategy_path}")
    else:
        # 嘗試查找 output/ 目錄下最新的策略文件
        strategy_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('_best_strategy.json')]
        if strategy_files:
            latest_strategy = os.path.join(OUTPUT_DIR, sorted(strategy_files)[-1])
            generator = SignalGenerator(strategy_file=latest_strategy)
            print(f"✅ Auto-loaded latest strategy: {latest_strategy}")
        else:
            # 如果沒有策略文件，使用默認公式
            default_formula = [10, 0]  # MA20(RET)
            generator = SignalGenerator(formula_tokens=default_formula)
            print("⚠️ No strategy file found, using default formula: MA20(RET)")
    
    print(f"📜 Formula: {generator.formula_readable}")
    print(f"📋 Scanning: {', '.join(symbols)}")
    
    if args.monitor:
        monitor_mode(generator, symbols, args.interval)
    else:
        print("\n🔍 Generating signals...")
        results = generator.scan_multiple(symbols)
        print_signal_report(results, generator.formula_readable)
