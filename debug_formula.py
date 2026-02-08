"""
调试脚本：验证训练报告的指标是否准确
"""
import json
import numpy as np
import pandas as pd
import torch
import yfinance as yf
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ==================== 算子定义 ====================
@torch.jit.script
def _ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    if d == 0: return x
    pad = torch.zeros((x.shape[0], d), device=x.device)
    return torch.cat([pad, x[:, :-d]], dim=1)

@torch.jit.script
def _ts_delta(x: torch.Tensor, d: int) -> torch.Tensor:
    return x - _ts_delay(x, d)

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

# ==================== 配置 ====================
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
    ('STD20', lambda x: x, 1),  # 简化
    ('MAX20', lambda x: _ts_max(x, 20), 1),
    ('MIN20', lambda x: x, 1),  # 简化
]

VOCAB = FEATURES + [cfg[0] for cfg in OPS_CONFIG]
OP_FUNC_MAP = {i + len(FEATURES): cfg[1] for i, cfg in enumerate(OPS_CONFIG)}
OP_ARITY_MAP = {i + len(FEATURES): cfg[2] for i, cfg in enumerate(OPS_CONFIG)}

COST_RATE = 0.0005

def robust_norm(x):
    x = x.astype(np.float32)
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
    
    features = torch.stack([
        torch.from_numpy(robust_norm(ret)),
        torch.from_numpy(robust_norm(ret5)),
        torch.from_numpy(robust_norm(vol_chg)),
        torch.from_numpy(robust_norm(v_ret)),
        torch.from_numpy(robust_norm(trend)),
    ])
    return features

def execute_formula(features, formula_tokens):
    stack = []
    for t in reversed(formula_tokens):
        if t < len(FEATURES):
            stack.append(features[t].unsqueeze(0))
        else:
            arity = OP_ARITY_MAP[t]
            args = [stack.pop() for _ in range(arity)]
            func = OP_FUNC_MAP[t]
            if arity == 2:
                res = func(args[0], args[1])
            else:
                res = func(args[0])
            res = torch.nan_to_num(res)
            stack.append(res)
    
    final = stack[-1]
    if final.dim() == 2:
        final = final.squeeze(0)
    return final

def decode_formula(tokens):
    stream = list(tokens)
    def _parse():
        if not stream: return ""
        t = stream.pop(0)
        if t < len(FEATURES): return FEATURES[t]
        args = [_parse() for _ in range(OP_ARITY_MAP[t])]
        return f"{VOCAB[t]}({','.join(args)})"
    return _parse()

# ==================== 主程序 ====================
if __name__ == "__main__":
    print("="*70)
    print("🔍 调试：验证公式在训练集上的真实表现")
    print("="*70)
    
    # 公式 tokens（从训练报告中获取）
    # SIGN(MA20(DIV(ADD(MAX20(RET5),RET),RET5)))
    # 需要从 best_strategy.json 读取
    
    # 下载数据
    print("\n📊 下载 SPY 数据 (2015-2024，训练集)...")
    ticker = yf.Ticker("SPY")
    df = ticker.history(start="2015-01-01", end="2024-01-01", auto_adjust=True)
    df = df.reset_index()
    print(f"   数据范围: {df['Date'].min()} ~ {df['Date'].max()}")
    print(f"   总天数: {len(df)}")
    
    # 计算特征
    features = compute_features(df)
    print(f"   特征形状: {features.shape}")
    
    # 计算目标收益 (T1OT2O)
    open_prices = df['Open'].values.astype(np.float32)
    open_t1 = np.roll(open_prices, -1)
    open_t2 = np.roll(open_prices, -2)
    target_ret = (open_t2 - open_t1) / (open_t1 + 1e-6)
    target_ret[-2:] = 0
    print(f"   目标收益范围: {target_ret.min():.4f} ~ {target_ret.max():.4f}")
    
    # 尝试查找最新的策略文件
    import os
    import glob
    strategy_files = glob.glob("output/**/best_strategy.json", recursive=True)
    if strategy_files:
        latest = max(strategy_files, key=os.path.getmtime)
        print(f"\n📜 加载策略: {latest}")
        with open(latest, 'r') as f:
            strategy = json.load(f)
        formula_tokens = strategy['formula_tokens']
        formula_str = strategy.get('formula_readable', decode_formula(formula_tokens))
    else:
        # 使用默认公式
        # SIGN(MA20(DIV(ADD(MAX20(RET5),RET),RET5)))
        # SIGN=11, MA20=15, DIV=8, ADD=5, MAX20=17, RET5=1, RET=0
        formula_tokens = [11, 15, 8, 5, 17, 1, 0, 1]  # 需要确认
        formula_str = "Manual tokens"
    
    print(f"   公式: {formula_str}")
    print(f"   Tokens: {formula_tokens}")
    
    # 执行公式
    factor = execute_formula(features, formula_tokens)
    print(f"\n📈 因子值统计:")
    print(f"   形状: {factor.shape}")
    print(f"   范围: {factor.min().item():.4f} ~ {factor.max().item():.4f}")
    print(f"   均值: {factor.mean().item():.4f}")
    print(f"   标准差: {factor.std().item():.4f}")
    
    # 检查 SIGN 的影响
    unique_vals = torch.unique(factor)
    print(f"   唯一值数量: {len(unique_vals)}")
    if len(unique_vals) <= 10:
        print(f"   唯一值: {unique_vals.tolist()}")
    
    # 信号
    signal = torch.tanh(factor).numpy()
    print(f"\n📊 信号统计 (tanh后):")
    print(f"   范围: {signal.min():.4f} ~ {signal.max():.4f}")
    print(f"   唯一值: {np.unique(signal)}")
    
    # 离散仓位 (阈值 0.1)
    threshold = 0.1
    pos = np.zeros_like(signal)
    pos[signal > threshold] = 1
    pos[signal < -threshold] = -1
    
    long_days = (pos == 1).sum()
    short_days = (pos == -1).sum()
    hold_days = (pos == 0).sum()
    
    print(f"\n📊 仓位统计 (阈值={threshold}):")
    print(f"   做多: {long_days} 天 ({long_days/len(pos)*100:.1f}%)")
    print(f"   做空: {short_days} 天 ({short_days/len(pos)*100:.1f}%)")
    print(f"   观望: {hold_days} 天 ({hold_days/len(pos)*100:.1f}%)")
    
    # 换手率
    turnover = np.abs(pos - np.roll(pos, 1))
    turnover[0] = 0
    avg_turnover = turnover.mean()
    total_trades = (turnover > 0).sum()
    print(f"   总交易次数: {total_trades}")
    print(f"   平均换手率: {avg_turnover:.4f}")
    
    # 回测
    daily_ret = pos * target_ret - turnover * COST_RATE
    equity = (1 + daily_ret).cumprod()
    total_return = equity[-1] - 1
    
    print(f"\n📈 回测结果 (训练集 2015-2024):")
    print(f"   累计收益: {total_return:.2%}")
    print(f"   最终净值: {equity[-1]:.4f}")
    
    # Sharpe
    ann_ret = equity[-1] ** (252 / len(equity)) - 1
    vol = np.std(daily_ret) * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / (vol + 1e-6)
    
    print(f"   年化收益: {ann_ret:.2%}")
    print(f"   年化波动: {vol:.2%}")
    print(f"   Sharpe: {sharpe:.2f}")
    
    # Max Drawdown
    dd = 1 - equity / np.maximum.accumulate(equity)
    max_dd = np.max(dd)
    print(f"   最大回撤: {max_dd:.2%}")
    
    # 胜率
    win_rate = (daily_ret > 0).mean()
    print(f"   胜率: {win_rate:.2%}")
    
    print("\n" + "="*70)
    print("⚠️ 如果这里的收益与训练报告的 245% 不符，说明代码有 bug！")
    print("="*70)
