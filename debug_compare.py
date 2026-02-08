"""
详细对比训练评估和回测的差异
"""
import numpy as np
import pandas as pd
import torch
import yfinance as yf
import json
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ===== Load strategy =====
with open('output/SPY_T1OT2O_20260127_102730/best_strategy.json', 'r') as f:
    strategy = json.load(f)
formula_tokens = strategy['formula_tokens']
print(f"Formula: {strategy['formula_readable']}")
print(f"Tokens: {formula_tokens}")

# ===== Operators (same as times_us.py) =====
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

FEATURES = ['RET', 'RET5', 'VOL_CHG', 'V_RET', 'TREND']
OPS_CONFIG = [
    ('ADD', lambda x, y: x + y, 2),
    ('SUB', lambda x, y: x - y, 2),
    ('MUL', lambda x, y: x * y, 2),
    ('DIV', lambda x, y: x / (y + 1e-6 * torch.sign(y)), 2),
    ('NEG', lambda x: -x, 1),
    ('ABS', lambda x: torch.abs(x), 1),
    ('SIGN', lambda x: torch.sign(x), 1),
    ('DELTA5', lambda x: x, 1),
    ('DELTA10', lambda x: x, 1),
    ('MA10', lambda x: _ts_decay_linear(x, 10), 1),
    ('MA20', lambda x: _ts_decay_linear(x, 20), 1),
    ('STD20', lambda x: x, 1),
    ('MAX20', lambda x: _ts_max(x, 20), 1),
    ('MIN20', lambda x: x, 1),
]
OP_FUNC_MAP = {i + len(FEATURES): cfg[1] for i, cfg in enumerate(OPS_CONFIG)}
OP_ARITY_MAP = {i + len(FEATURES): cfg[2] for i, cfg in enumerate(OPS_CONFIG)}

def robust_norm(x):
    median = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - median)) + 1e-6
    res = (x - median) / mad
    return np.clip(res, -5, 5).astype(np.float32)

def compute_features(df):
    close = df['Close'].values.astype(np.float32)
    vol = df['Volume'].values.astype(np.float32)
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
    return torch.stack([
        torch.from_numpy(robust_norm(ret)),
        torch.from_numpy(robust_norm(ret5)),
        torch.from_numpy(robust_norm(vol_chg)),
        torch.from_numpy(robust_norm(v_ret)),
        torch.from_numpy(robust_norm(trend)),
    ])

def execute_formula(features, tokens):
    stack = []
    for t in reversed(tokens):
        if t < len(FEATURES):
            stack.append(features[t].unsqueeze(0))
        else:
            arity = OP_ARITY_MAP[t]
            args = [stack.pop() for _ in range(arity)]
            func = OP_FUNC_MAP[t]
            res = func(args[0], args[1]) if arity == 2 else func(args[0])
            res = torch.nan_to_num(res)
            stack.append(res)
    final = stack[-1]
    if final.dim() == 2: final = final.squeeze(0)
    return final

COST_RATE = 0.0005
threshold = 0.1

# ===== TRAINING simulation =====
print()
print('=== TRAINING Simulation (2015-2026 data, eval on 2015-2024) ===')
df_full = yf.Ticker('SPY').history(start='2015-01-01', end='2026-01-27', auto_adjust=True)
features_full = compute_features(df_full)
factor_full = execute_formula(features_full, formula_tokens)

# Split index
dates = df_full.index
split_mask = dates < pd.Timestamp('2024-01-01', tz=dates.tz)
split_idx = split_mask.sum()

# Only take training portion
factor_train = factor_full[:split_idx]
signal_train = torch.tanh(factor_train).numpy()

# Target return (T1OT2O)
open_prices_full = df_full['Open'].values.astype(np.float32)
open_t1 = np.roll(open_prices_full, -1)
open_t2 = np.roll(open_prices_full, -2)
target_full = (open_t2 - open_t1) / (open_t1 + 1e-6)
target_full[-2:] = 0
target_train = target_full[:split_idx]

# Position
pos_train = np.zeros_like(signal_train)
pos_train[signal_train > threshold] = 1
pos_train[signal_train < -threshold] = -1

# Turnover
turnover_train = np.abs(pos_train - np.roll(pos_train, 1))
turnover_train[0] = 0

# PnL
pnl_train = pos_train * target_train - turnover_train * COST_RATE
equity_train = (1 + pnl_train).cumprod()

print(f'  Data days: {len(df_full)}, Split: {split_idx}')
print(f'  Factor unique values: {len(torch.unique(factor_train))}')
print(f'  Long: {(pos_train==1).sum()}, Short: {(pos_train==-1).sum()}, Hold: {(pos_train==0).sum()}')
print(f'  Total Return: {equity_train[-1] - 1:.2%}')

# ===== BACKTEST simulation =====
print()
print('=== BACKTEST Simulation (2015-2024 data only) ===')
df_bt = yf.Ticker('SPY').history(start='2015-01-01', end='2024-01-01', auto_adjust=True)
features_bt = compute_features(df_bt)
factor_bt = execute_formula(features_bt, formula_tokens)
signal_bt = torch.tanh(factor_bt).numpy()

# Target return
open_prices_bt = df_bt['Open'].values.astype(np.float32)
open_t1_bt = np.roll(open_prices_bt, -1)
open_t2_bt = np.roll(open_prices_bt, -2)
target_bt = (open_t2_bt - open_t1_bt) / (open_t1_bt + 1e-6)
target_bt[-2:] = 0

# Position
pos_bt = np.zeros_like(signal_bt)
pos_bt[signal_bt > threshold] = 1
pos_bt[signal_bt < -threshold] = -1

# Turnover
turnover_bt = np.abs(pos_bt - np.roll(pos_bt, 1))
turnover_bt[0] = 0

# PnL
pnl_bt = pos_bt * target_bt - turnover_bt * COST_RATE
equity_bt = (1 + pnl_bt).cumprod()

print(f'  Data days: {len(df_bt)}')
print(f'  Factor unique values: {len(torch.unique(factor_bt))}')
print(f'  Long: {(pos_bt==1).sum()}, Short: {(pos_bt==-1).sum()}, Hold: {(pos_bt==0).sum()}')
print(f'  Total Return: {equity_bt[-1] - 1:.2%}')

# ===== KEY COMPARISON =====
print()
print('=== KEY COMPARISON ===')
print(f'  Training sim return:  {equity_train[-1] - 1:.2%}')
print(f'  Backtest sim return:  {equity_bt[-1] - 1:.2%}')
print(f'  DIFFERENCE:           {(equity_train[-1] - equity_bt[-1]):.2%}')

# Check position differences
min_len = min(len(pos_train), len(pos_bt))
pos_diff = (pos_train[:min_len] != pos_bt[:min_len]).sum()
print(f'  Position mismatches:  {pos_diff} / {min_len} ({pos_diff/min_len*100:.1f}%)')

# Check factor value differences
factor_diff = (factor_train[:min_len].numpy() != factor_bt[:min_len].numpy()).sum()
print(f'  Factor mismatches:    {factor_diff} / {min_len} ({factor_diff/min_len*100:.1f}%)')

# ===== 10-YEAR BACKTEST =====
print()
print('=== 10-YEAR BACKTEST (2016-2026) ===')
df_10y = yf.Ticker('SPY').history(start='2016-01-01', end='2026-01-27', auto_adjust=True)
features_10y = compute_features(df_10y)
factor_10y = execute_formula(features_10y, formula_tokens)
signal_10y = torch.tanh(factor_10y).numpy()

# Target return
open_prices_10y = df_10y['Open'].values.astype(np.float32)
open_t1_10y = np.roll(open_prices_10y, -1)
open_t2_10y = np.roll(open_prices_10y, -2)
target_10y = (open_t2_10y - open_t1_10y) / (open_t1_10y + 1e-6)
target_10y[-2:] = 0

# Position
pos_10y = np.zeros_like(signal_10y)
pos_10y[signal_10y > threshold] = 1
pos_10y[signal_10y < -threshold] = -1

# Turnover
turnover_10y = np.abs(pos_10y - np.roll(pos_10y, 1))
turnover_10y[0] = 0

# PnL
pnl_10y = pos_10y * target_10y - turnover_10y * COST_RATE
equity_10y = (1 + pnl_10y).cumprod()

print(f'  Data days: {len(df_10y)}')
print(f'  Long: {(pos_10y==1).sum()}, Short: {(pos_10y==-1).sum()}, Hold: {(pos_10y==0).sum()}')
print(f'  Total Return: {equity_10y[-1] - 1:.2%}')

# Buy & Hold
bh_ret = df_10y['Close'].iloc[-1] / df_10y['Close'].iloc[0] - 1
print(f'  Buy & Hold: {bh_ret:.2%}')

print()
print('='*60)
print('CONCLUSION: The difference between Training and Backtest')
print('shows there is likely a BUG in the training evaluation code!')
print('='*60)
