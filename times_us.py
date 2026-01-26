"""
AlphaGPT - US Stock Version
美股因子挖掘與回測腳本

使用 yfinance 獲取美股數據，支持：
- 個股（如 AAPL, TSLA, NVDA）
- ETF（如 SPY, QQQ, IWM）
- 指數（如 ^GSPC, ^NDX）

Usage:
    python times_us.py --symbol SPY --start 2015-01-01 --end 2024-01-01
"""

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==================== 配置 ====================
DEFAULT_SYMBOL = 'SPY'           # 默認標的：S&P 500 ETF
START_DATE = '2015-01-01'        # 訓練數據開始
END_DATE = '2024-01-01'          # 訓練數據結束
TEST_END_DATE = '2025-01-01'     # 測試時間結束

BATCH_SIZE = 1024
TRAIN_ITERATIONS = 400
MAX_SEQ_LEN = 8                  # 限制公式長度，防止過擬合
COST_RATE = 0.0001               # 美股交易成本較低（約萬一，含滑點）

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

# ==================== 時序算子 ====================
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

# ==================== 算子配置 ====================
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

# ==================== 因子配置 ====================
FEATURES = ['RET', 'RET5', 'RET20', 'VOL_CHG', 'V_RET', 'TREND', 'ATR', 'RSI']

VOCAB = FEATURES + [cfg[0] for cfg in OPS_CONFIG]
VOCAB_SIZE = len(VOCAB)
OP_FUNC_MAP = {i + len(FEATURES): cfg[1] for i, cfg in enumerate(OPS_CONFIG)}
OP_ARITY_MAP = {i + len(FEATURES): cfg[2] for i, cfg in enumerate(OPS_CONFIG)}


# ==================== 模型定義 ====================
class AlphaGPT(nn.Module):
    def __init__(self, d_model=64, n_head=4, n_layer=2):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, MAX_SEQ_LEN + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=128, 
            batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        self.ln_f = nn.LayerNorm(d_model)
        self.head_actor = nn.Linear(d_model, VOCAB_SIZE)
        self.head_critic = nn.Linear(d_model, 1)

    def forward(self, idx):
        B, T = idx.size()
        x = self.token_emb(idx) + self.pos_emb[:, :T, :]
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(idx.device)
        x = self.blocks(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        last = x[:, -1, :]
        return self.head_actor(last), self.head_critic(last)


# ==================== 數據引擎 ====================
class USDataEngine:
    def __init__(self, symbol, start_date, end_date, test_end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.test_end_date = test_end_date
        self.cache_path = f'data_cache_{symbol}.parquet'
        
    def load(self):
        if os.path.exists(self.cache_path):
            print(f"📁 Loading cached data for {self.symbol}...")
            df = pd.read_parquet(self.cache_path)
        else:
            print(f"🌐 Downloading {self.symbol} from Yahoo Finance...")
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(start=self.start_date, end=self.test_end_date, auto_adjust=True)
            
            if df.empty:
                raise ValueError(f"No data found for {self.symbol}")
            
            df = df.reset_index()
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits']
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
            df.to_parquet(self.cache_path)
            print(f"✅ Data cached to {self.cache_path}")

        # 數據清洗
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce').ffill().bfill()

        self.dates = pd.to_datetime(df['date'])
        
        close = df['close'].values.astype(np.float32)
        open_ = df['open'].values.astype(np.float32)
        high = df['high'].values.astype(np.float32)
        low = df['low'].values.astype(np.float32)
        vol = df['volume'].values.astype(np.float32)

        # ==================== 因子計算 ====================
        
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

        # 6. 趨勢（相對60日均線）
        ma60 = pd.Series(close).rolling(60).mean().values
        trend = np.zeros_like(close)
        mask = ma60 > 0
        trend[mask] = close[mask] / ma60[mask] - 1
        trend = np.nan_to_num(trend).astype(np.float32)

        # 7. ATR (Average True Range) - 波動率指標
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        atr = pd.Series(tr).rolling(14).mean().fillna(0).values.astype(np.float32)
        atr_norm = atr / (close + 1e-6)  # 標準化

        # 8. RSI (Relative Strength Index)
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-6)
        rsi = (100 - 100 / (1 + rs)).fillna(50).values.astype(np.float32)
        rsi_norm = (rsi - 50) / 50  # 標準化到 [-1, 1]

        # Robust Normalization
        def robust_norm(x):
            x = x.astype(np.float32)
            median = np.nanmedian(x)
            mad = np.nanmedian(np.abs(x - median)) + 1e-6
            res = (x - median) / mad
            return np.clip(res, -5, 5).astype(np.float32)

        # 構建特徵張量
        self.feat_data = torch.stack([
            torch.from_numpy(robust_norm(ret)).to(DEVICE),
            torch.from_numpy(robust_norm(ret5)).to(DEVICE),
            torch.from_numpy(robust_norm(ret20)).to(DEVICE),
            torch.from_numpy(robust_norm(vol_chg)).to(DEVICE),
            torch.from_numpy(robust_norm(v_ret)).to(DEVICE),
            torch.from_numpy(robust_norm(trend)).to(DEVICE),
            torch.from_numpy(robust_norm(atr_norm)).to(DEVICE),
            torch.from_numpy(rsi_norm).to(DEVICE),  # RSI 已經標準化
        ])

        # 目標收益 (Open-to-Open)
        open_tensor = torch.from_numpy(open_).to(DEVICE)
        open_t1 = torch.roll(open_tensor, -1)
        open_t2 = torch.roll(open_tensor, -2)
        self.target_oto_ret = (open_t2 - open_t1) / (open_t1 + 1e-6)
        self.target_oto_ret[-2:] = 0.0

        self.raw_open = open_tensor
        self.raw_close = torch.from_numpy(close).to(DEVICE)

        # 80% 訓練，20% 測試
        self.split_idx = int(len(df) * 0.8)
        
        print(f"✅ {self.symbol} Data Ready!")
        print(f"   Total: {len(df)} days | Train: {self.split_idx} | Test: {len(df) - self.split_idx}")
        print(f"   Features: {FEATURES}")
        return self


# ==================== 因子挖掘器 ====================
class DeepQuantMiner:
    def __init__(self, engine):
        self.engine = engine
        self.model = AlphaGPT().to(DEVICE)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-5)
        self.best_sharpe = -10.0
        self.best_formula_tokens = None

    def get_strict_mask(self, open_slots, step):
        B = open_slots.shape[0]
        mask = torch.full((B, VOCAB_SIZE), float('-inf'), device=DEVICE)
        remaining_steps = MAX_SEQ_LEN - step

        done_mask = (open_slots == 0)
        mask[done_mask, 0] = 0.0

        active_mask = ~done_mask
        must_pick_feat = (open_slots >= remaining_steps)

        mask[active_mask, :len(FEATURES)] = 0.0
        can_pick_op_mask = active_mask & (~must_pick_feat)
        if can_pick_op_mask.any():
            mask[can_pick_op_mask, len(FEATURES):] = 0.0
        return mask

    def solve_one(self, tokens):
        stack = []
        try:
            for t in reversed(tokens):
                if t < len(FEATURES):
                    stack.append(self.engine.feat_data[t])
                else:
                    arity = OP_ARITY_MAP[t]
                    if len(stack) < arity: raise ValueError
                    args = [stack.pop() for _ in range(arity)]
                    func = OP_FUNC_MAP[t]
                    if arity == 2: res = func(args[0], args[1])
                    else: res = func(args[0])
                    if torch.isnan(res).any(): res = torch.nan_to_num(res)
                    stack.append(res)

            if len(stack) >= 1:
                final = stack[-1]
                if final.std() < 1e-4: return None
                return final
        except:
            return None
        return None

    def solve_batch(self, token_seqs):
        B = token_seqs.shape[0]
        results = torch.zeros((B, self.engine.feat_data.shape[1]), device=DEVICE)
        valid_mask = torch.zeros(B, dtype=torch.bool, device=DEVICE)

        for i in range(B):
            res = self.solve_one(token_seqs[i].cpu().tolist())
            if res is not None:
                results[i] = res
                valid_mask[i] = True
        return results, valid_mask

    def backtest(self, factors):
        if factors.shape[0] == 0: return torch.tensor([], device=DEVICE)

        split = self.engine.split_idx
        target = self.engine.target_oto_ret[:split]

        rewards = torch.zeros(factors.shape[0], device=DEVICE)

        for i in range(factors.shape[0]):
            f = factors[i, :split]

            if torch.isnan(f).all() or (f == 0).all() or f.numel() == 0:
                rewards[i] = -2.0
                continue

            sig = torch.tanh(f)
            pos = torch.sign(sig)

            turnover = torch.abs(pos - torch.roll(pos, 1))
            if turnover.numel() > 0:
                turnover[0] = 0.0
            else:
                rewards[i] = -2.0
                continue

            pnl = pos * target - turnover * COST_RATE

            if pnl.numel() < 10:
                rewards[i] = -2.0
                continue

            mu = pnl.mean()
            std = pnl.std() + 1e-6

            # Sortino Ratio
            downside_returns = pnl[pnl < 0]
            if downside_returns.numel() > 5:
                down_std = downside_returns.std() + 1e-6
                sortino = mu / down_std * 15.87
            else:
                sortino = mu / std * 15.87

            if mu < 0: sortino = -2.0
            if turnover.mean() > 0.5: sortino -= 1.0
            if (pos == 0).all(): sortino = -2.0

            rewards[i] = sortino

        return torch.clamp(rewards, -3, 5)

    def train(self):
        print(f"🚀 Training AlphaGPT for {self.engine.symbol}...")
        print(f"   MAX_SEQ_LEN={MAX_SEQ_LEN} | BATCH={BATCH_SIZE} | ITER={TRAIN_ITERATIONS}")
        pbar = tqdm(range(TRAIN_ITERATIONS))

        for _ in pbar:
            B = BATCH_SIZE
            open_slots = torch.ones(B, dtype=torch.long, device=DEVICE)
            log_probs, tokens = [], []
            curr_inp = torch.zeros((B, 1), dtype=torch.long, device=DEVICE)

            for step in range(MAX_SEQ_LEN):
                logits, val = self.model(curr_inp)
                mask = self.get_strict_mask(open_slots, step)
                dist = Categorical(logits=(logits + mask))
                action = dist.sample()

                log_probs.append(dist.log_prob(action))
                tokens.append(action)
                curr_inp = torch.cat([curr_inp, action.unsqueeze(1)], dim=1)

                is_op = action >= len(FEATURES)
                delta = torch.full((B,), -1, device=DEVICE)
                arity_tens = torch.zeros(VOCAB_SIZE, dtype=torch.long, device=DEVICE)
                for k,v in OP_ARITY_MAP.items(): arity_tens[k] = v
                op_delta = arity_tens[action] - 1
                delta = torch.where(is_op, op_delta, delta)
                delta[open_slots==0] = 0
                open_slots += delta

            seqs = torch.stack(tokens, dim=1)

            with torch.no_grad():
                f_vals, valid_mask = self.solve_batch(seqs)
                valid_idx = torch.where(valid_mask)[0]
                rewards = torch.full((B,), -1.0, device=DEVICE)

                if len(valid_idx) > 0:
                    bt_scores = self.backtest(f_vals[valid_idx])
                    rewards[valid_idx] = bt_scores

                    best_sub_idx = torch.argmax(bt_scores)
                    current_best_score = bt_scores[best_sub_idx].item()

                    if current_best_score > self.best_sharpe:
                        self.best_sharpe = current_best_score
                        self.best_formula_tokens = seqs[valid_idx[best_sub_idx]].cpu().tolist()

            adv = rewards - rewards.mean()
            loss = -(torch.stack(log_probs, 1).sum(1) * adv).mean()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            pbar.set_postfix({'Valid': f"{len(valid_idx)/B:.1%}", 'BestSortino': f"{self.best_sharpe:.2f}"})

    def decode(self, tokens=None):
        if tokens is None: tokens = self.best_formula_tokens
        if tokens is None: return "N/A"
        stream = list(tokens)
        def _parse():
            if not stream: return ""
            t = stream.pop(0)
            if t < len(FEATURES): return FEATURES[t]
            args = [_parse() for _ in range(OP_ARITY_MAP[t])]
            return f"{VOCAB[t]}({','.join(args)})"
        try: return _parse()
        except: return "Invalid"


# ==================== 回測報告 ====================
def final_reality_check(miner, engine):
    print("\n" + "="*60)
    print(f"🔬 FINAL REALITY CHECK - {engine.symbol} (Out-of-Sample)")
    print("="*60)

    formula_str = miner.decode()
    if miner.best_formula_tokens is None:
        print("❌ No valid formula found!")
        return
    print(f"📜 Strategy Formula: {formula_str}")

    factor_all = miner.solve_one(miner.best_formula_tokens)
    if factor_all is None:
        print("❌ Formula execution failed!")
        return

    split = engine.split_idx
    test_dates = engine.dates[split:]
    test_factors = factor_all[split:].cpu().numpy()
    test_ret = engine.target_oto_ret[split:].cpu().numpy()

    signal = np.tanh(test_factors)
    position = np.sign(signal)

    turnover = np.abs(position - np.roll(position, 1))
    turnover[0] = 0

    daily_ret = position * test_ret - turnover * COST_RATE
    equity = (1 + daily_ret).cumprod()

    # 統計指標
    total_ret = equity[-1] - 1
    ann_ret = equity[-1] ** (252/len(equity)) - 1
    vol = np.std(daily_ret) * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / (vol + 1e-6)

    dd = 1 - equity / np.maximum.accumulate(equity)
    max_dd = np.max(dd)
    calmar = ann_ret / (max_dd + 1e-6)

    # 勝率
    win_rate = np.mean(daily_ret > 0)
    
    # 盈虧比
    avg_win = np.mean(daily_ret[daily_ret > 0]) if np.any(daily_ret > 0) else 0
    avg_loss = np.abs(np.mean(daily_ret[daily_ret < 0])) if np.any(daily_ret < 0) else 1e-6
    profit_factor = avg_win / avg_loss

    print("-" * 60)
    print(f"📅 Test Period    : {test_dates.iloc[0].date()} ~ {test_dates.iloc[-1].date()}")
    print(f"📈 Total Return   : {total_ret:.2%}")
    print(f"📈 Ann. Return    : {ann_ret:.2%}")
    print(f"📊 Ann. Volatility: {vol:.2%}")
    print(f"⭐ Sharpe Ratio   : {sharpe:.2f}")
    print(f"📉 Max Drawdown   : {max_dd:.2%}")
    print(f"🎯 Calmar Ratio   : {calmar:.2f}")
    print(f"✅ Win Rate       : {win_rate:.2%}")
    print(f"💰 Profit Factor  : {profit_factor:.2f}")
    print("-" * 60)

    # 繪圖
    plt.style.use('bmh')
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 上圖：淨值曲線
    ax1 = axes[0]
    ax1.plot(test_dates, equity, label='AlphaGPT Strategy', linewidth=2, color='#2E86AB')
    
    bench_equity = (1 + test_ret).cumprod()
    ax1.plot(test_dates, bench_equity, label=f'{engine.symbol} Buy & Hold', alpha=0.6, linewidth=1.5, color='#A23B72')
    
    ax1.fill_between(test_dates, equity, alpha=0.3, color='#2E86AB')
    ax1.set_title(f'{engine.symbol} AlphaGPT Strategy | Ann: {ann_ret:.1%} | Sharpe: {sharpe:.2f} | MaxDD: {max_dd:.1%}', fontsize=14)
    ax1.set_ylabel('Cumulative Return')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 下圖：回撤
    ax2 = axes[1]
    ax2.fill_between(test_dates, -dd * 100, alpha=0.7, color='#E94F37', label='Drawdown')
    ax2.set_title('Drawdown (%)', fontsize=12)
    ax2.set_ylabel('Drawdown %')
    ax2.set_xlabel('Date')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = f'{engine.symbol}_strategy_performance.png'
    plt.savefig(output_file, dpi=150)
    print(f"📈 Chart saved to '{output_file}'")

    # 保存策略
    import json
    strategy_file = f'{engine.symbol}_best_strategy.json'
    with open(strategy_file, 'w') as f:
        json.dump({
            'symbol': engine.symbol,
            'formula_tokens': miner.best_formula_tokens,
            'formula_readable': formula_str,
            'train_sharpe': miner.best_sharpe,
            'test_sharpe': sharpe,
            'test_ann_return': ann_ret,
            'test_max_drawdown': max_dd
        }, f, indent=2)
    print(f"💾 Strategy saved to '{strategy_file}'")


# ==================== 主程序 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AlphaGPT US Stock Factor Mining')
    parser.add_argument('--symbol', type=str, default=DEFAULT_SYMBOL, 
                        help='Stock/ETF symbol (e.g., SPY, AAPL, QQQ)')
    parser.add_argument('--start', type=str, default=START_DATE, 
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=END_DATE, 
                        help='Training end date (YYYY-MM-DD)')
    parser.add_argument('--test_end', type=str, default=TEST_END_DATE, 
                        help='Test end date (YYYY-MM-DD)')
    parser.add_argument('--iterations', type=int, default=TRAIN_ITERATIONS,
                        help='Training iterations')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size')
    args = parser.parse_args()

    # 更新全局配置
    TRAIN_ITERATIONS = args.iterations
    BATCH_SIZE = args.batch_size

    print("="*60)
    print("🚀 AlphaGPT - US Stock Factor Mining")
    print("="*60)
    print(f"Symbol     : {args.symbol}")
    print(f"Train      : {args.start} ~ {args.end}")
    print(f"Test       : {args.end} ~ {args.test_end}")
    print(f"Device     : {DEVICE}")
    print("="*60)

    # 加載數據
    engine = USDataEngine(args.symbol, args.start, args.end, args.test_end)
    engine.load()

    # 訓練
    miner = DeepQuantMiner(engine)
    miner.train()

    # 回測報告
    final_reality_check(miner, engine)
