"""
验证回测计算是否正确
"""
import numpy as np
import pandas as pd
import yfinance as yf
import json
import torch
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Load strategy and norm_params
with open('output/NVDA_T1OT2O_20260127_105823/best_strategy.json', 'r') as f:
    strategy = json.load(f)

formula_tokens = strategy['formula_tokens']
norm_params = strategy.get('norm_params', None)
print(f"Formula: {strategy['formula_readable']}")
print(f"Norm params available: {norm_params is not None}")

# Operators (no jit for simplicity)
def _ts_delta(x, d):
    result = x.clone()
    if d > 0:
        result[:, d:] = x[:, d:] - x[:, :-d]
        result[:, :d] = 0
    return result

def _ts_min(x, d):
    if d <= 1:
        return x
    B, T = x.shape
    result = torch.zeros_like(x)
    for i in range(T):
        start = max(0, i - d + 1)
        result[:, i] = x[:, start:i+1].min(dim=-1)[0]
    return result

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
    ('MA10', lambda x: x, 1),  # simplified
    ('MA20', lambda x: x, 1),  # simplified
    ('STD20', lambda x: x, 1),  # simplified
    ('MAX20', lambda x: x, 1),  # simplified
    ('MIN20', lambda x: _ts_min(x, 20), 1),
]
OP_FUNC_MAP = {i + len(FEATURES): cfg[1] for i, cfg in enumerate(OPS_CONFIG)}
OP_ARITY_MAP = {i + len(FEATURES): cfg[2] for i, cfg in enumerate(OPS_CONFIG)}

def robust_norm(x, feature_name=None):
    x = x.astype(np.float32)
    if norm_params and feature_name and feature_name in norm_params:
        median = norm_params[feature_name]['median']
        mad = norm_params[feature_name]['mad']
    else:
        median = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - median)) + 1e-6
    res = (x - median) / mad
    return np.clip(res, -5, 5).astype(np.float32)

# Download data
print("\nDownloading NVDA data...")
df = yf.Ticker('NVDA').history(start='2023-01-01', end='2026-01-27', auto_adjust=True)
print(f"Period: {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)} days")

close = df['Close'].values.astype(np.float32)
open_prices = df['Open'].values.astype(np.float32)
vol = df['Volume'].values.astype(np.float32)

# Compute features
ret = np.zeros_like(close)
ret[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-6)

ret5 = pd.Series(close).pct_change(5).fillna(0).values.astype(np.float32)

vol_ma = pd.Series(vol).rolling(20).mean().values
vol_chg = np.zeros_like(vol)
mask = vol_ma > 0
vol_chg[mask] = vol[mask] / vol_ma[mask] - 1
vol_chg = np.nan_to_num(vol_chg).astype(np.float32)

v_ret = (ret * (vol_chg + 1)).astype(np.float32)

ma60 = pd.Series(close).rolling(60).mean().values
trend = np.zeros_like(close)
mask = ma60 > 0
trend[mask] = close[mask] / ma60[mask] - 1
trend = np.nan_to_num(trend).astype(np.float32)

features = torch.stack([
    torch.from_numpy(robust_norm(ret, 'ret')),
    torch.from_numpy(robust_norm(ret5, 'ret5')),
    torch.from_numpy(robust_norm(vol_chg, 'vol_chg')),
    torch.from_numpy(robust_norm(v_ret, 'v_ret')),
    torch.from_numpy(robust_norm(trend, 'trend')),
])

print(f"Features shape: {features.shape}")

# Execute formula: MUL(MIN20(DELTA10(V_RET)),ADD(MIN20(RET),V_RET))
# tokens: [7, 18, 13, 3, 5, 18, 0, 3]
print(f"\nExecuting formula with tokens: {formula_tokens}")
stack = []
for t in reversed(formula_tokens):
    if t < len(FEATURES):
        stack.append(features[t].unsqueeze(0))
        print(f"  Push feature {FEATURES[t]}")
    else:
        arity = OP_ARITY_MAP[t]
        args = [stack.pop() for _ in range(arity)]
        func = OP_FUNC_MAP[t]
        op_name = OPS_CONFIG[t - len(FEATURES)][0]
        if arity == 2:
            res = func(args[0], args[1])
        else:
            res = func(args[0])
        res = torch.nan_to_num(res)
        stack.append(res)
        print(f"  Apply {op_name} (arity={arity})")

factor = stack[-1].squeeze(0).numpy()
signal = np.tanh(factor)

print(f"\nSignal stats:")
print(f"  Range: {signal.min():.4f} ~ {signal.max():.4f}")
print(f"  Mean: {signal.mean():.4f}")

# Position
threshold = 0.1
pos = np.zeros_like(signal)
pos[signal > threshold] = 1
pos[signal < -threshold] = -1

# Target return (T1OT2O)
open_t1 = np.roll(open_prices, -1)
open_t2 = np.roll(open_prices, -2)
target_ret = (open_t2 - open_t1) / (open_t1 + 1e-6)
target_ret[-2:] = 0

print(f"\nTarget return (T1OT2O) stats:")
print(f"  Range: {target_ret.min():.4f} ~ {target_ret.max():.4f}")
print(f"  Mean: {target_ret.mean():.4f}")

# Backtest
turnover = np.abs(pos - np.roll(pos, 1))
turnover[0] = 0
COST_RATE = 0.0005
daily_ret = pos * target_ret - turnover * COST_RATE
equity = (1 + daily_ret).cumprod()

# Buy & Hold
bh_ret = np.zeros_like(close)
bh_ret[1:] = (close[1:] - close[:-1]) / close[:-1]
bh_equity = (1 + bh_ret).cumprod()

# Always Long T1OT2O
always_long_ret = target_ret
always_long_equity = (1 + always_long_ret).cumprod()

print("\n" + "="*60)
print("=== BACKTEST VERIFICATION ===")
print("="*60)
print(f"Strategy equity: {equity[-1]:.2f}x ({equity[-1]-1:.1%})")
print(f"Buy & Hold equity: {bh_equity[-1]:.2f}x ({bh_equity[-1]-1:.1%})")
print(f"Always Long T1OT2O: {always_long_equity[-1]:.2f}x ({always_long_equity[-1]-1:.1%})")
print()
print("Position stats:")
print(f"  Long days: {(pos == 1).sum()}")
print(f"  Short days: {(pos == -1).sum()}")
print(f"  Hold days: {(pos == 0).sum()}")
print(f"  Trades: {(turnover > 0).sum()}")
print()

# Check if strategy beats always long
if equity[-1] > always_long_equity[-1]:
    print("⚠️ Strategy beats Always Long T1OT2O!")
    print("   This means the strategy's timing/shorting is adding value")
    
    # Calculate contribution from shorts
    short_days = pos == -1
    short_return = -target_ret[short_days]  # Negative because we're short
    print(f"   Short day returns: sum={short_return.sum():.2%}, count={short_days.sum()}")
    
    # If market dropped on short days, we made money
    dropped_on_short = (target_ret[short_days] < 0).sum()
    print(f"   Days market dropped while short: {dropped_on_short}/{short_days.sum()}")
else:
    print("✅ Strategy is close to Always Long T1OT2O (expected)")

print("\n" + "="*60)
