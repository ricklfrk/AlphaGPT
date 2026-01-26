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
import sys
import argparse
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# Windows 編碼修復
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ==================== 輸出目錄 ====================
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 配置 ====================
DEFAULT_SYMBOL = 'NVDA'           # 默認標的：S&P 500 ETF
START_DATE = '2020-01-01'        # 訓練數據開始
END_DATE = '2025-01-01'          # 訓練數據結束
TEST_END_DATE = '2026-01-23'     # 測試時間結束

BATCH_SIZE = 1024
TRAIN_ITERATIONS = 400
MAX_SEQ_LEN = 8                  # 限制公式長度，防止過擬合
COST_RATE = 0.0005               # 美股交易成本較低（約萬5，含滑點）

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
        self.cache_path = os.path.join(OUTPUT_DIR, f'data_cache_{symbol}.parquet')
        
    def load(self):
        if os.path.exists(self.cache_path):
            print(f"📁 Loading cached data for {self.symbol}...")
            df = pd.read_parquet(self.cache_path)
        else:
            print(f"🌐 Downloading {self.symbol} from Yahoo Finance...")
            
            # 重試機制
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    ticker = yf.Ticker(self.symbol)
                    df = ticker.history(start=self.start_date, end=self.test_end_date, auto_adjust=True)
                    
                    if df.empty:
                        raise ValueError(f"No data found for {self.symbol}")
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"⚠️ Attempt {attempt + 1} failed: {e}. Retrying...")
                        import time
                        time.sleep(2)
                    else:
                        raise ValueError(f"Failed to download {self.symbol} after {max_retries} attempts: {e}")
            
            # yfinance 返回的 DataFrame 索引是日期，列名是 Open, High, Low, Close, Volume 等
            df = df.reset_index()
            
            # 檢查必要的列是否存在
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            # 有些版本可能用小寫，做個兼容處理
            col_mapping = {}
            for col in df.columns:
                col_lower = col.lower()
                if col_lower == 'date' or col_lower == 'index':
                    col_mapping[col] = 'date'
                elif col_lower in ['open', 'high', 'low', 'close', 'volume']:
                    col_mapping[col] = col_lower
            
            df = df.rename(columns=col_mapping)
            
            # 確保所有必要的列都存在
            needed_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [c for c in needed_cols if c not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}. Available: {list(df.columns)}")
            
            df = df[needed_cols].copy()
            
            # 數據驗證
            print(f"📊 Data validation:")
            print(f"   - Date range: {df['date'].min()} ~ {df['date'].max()}")
            print(f"   - Total rows: {len(df)}")
            
            # 檢查缺失值
            null_counts = df.isnull().sum()
            if null_counts.sum() > 0:
                print(f"   ⚠️ Missing values detected: {null_counts.to_dict()}")
            
            # 檢查價格異常（如負數或零）
            for col in ['open', 'high', 'low', 'close']:
                invalid = (df[col] <= 0).sum()
                if invalid > 0:
                    print(f"   ⚠️ {col}: {invalid} invalid values (<=0)")
            
            # 檢查 High >= Low
            invalid_hl = (df['high'] < df['low']).sum()
            if invalid_hl > 0:
                print(f"   ⚠️ {invalid_hl} rows where High < Low")
            
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

        # 根據 END_DATE 劃分訓練集和測試集（而不是 80/20 自動劃分）
        # 訓練集：start_date ~ end_date
        # 測試集：end_date ~ test_end_date
        # 移除時區信息以便比較
        dates_naive = self.dates.dt.tz_localize(None) if self.dates.dt.tz is not None else self.dates
        end_date_dt = pd.to_datetime(self.end_date)
        split_mask = dates_naive < end_date_dt
        self.split_idx = split_mask.sum()
        
        # 計算實際日期範圍
        train_start = self.dates.iloc[0].strftime('%Y-%m-%d')
        train_end = self.dates.iloc[self.split_idx - 1].strftime('%Y-%m-%d') if self.split_idx > 0 else train_start
        test_start = self.dates.iloc[self.split_idx].strftime('%Y-%m-%d') if self.split_idx < len(self.dates) else train_end
        test_end = self.dates.iloc[-1].strftime('%Y-%m-%d')
        
        print(f"✅ {self.symbol} Data Ready!")
        print(f"   Total: {len(df)} days | Train: {self.split_idx} | Test: {len(df) - self.split_idx}")
        print(f"   Train Period: {train_start} ~ {train_end}")
        print(f"   Test Period : {test_start} ~ {test_end}")
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
            
            # ========== 離散倉位 (Discrete) ==========
            pos_discrete = torch.sign(sig)  # {-1, 0, 1}
            
            turnover_discrete = torch.abs(pos_discrete - torch.roll(pos_discrete, 1))
            if turnover_discrete.numel() > 0:
                turnover_discrete[0] = 0.0
            else:
                rewards[i] = -2.0
                continue

            pnl_discrete = pos_discrete * target - turnover_discrete * COST_RATE

            if pnl_discrete.numel() < 10:
                rewards[i] = -2.0
                continue

            mu_d = pnl_discrete.mean()
            std_d = pnl_discrete.std() + 1e-6

            downside_d = pnl_discrete[pnl_discrete < 0]
            if downside_d.numel() > 5:
                down_std_d = downside_d.std() + 1e-6
                sortino_discrete = mu_d / down_std_d * 15.87
            else:
                sortino_discrete = mu_d / std_d * 15.87

            # 離散倉位懲罰
            if mu_d < 0: sortino_discrete = -2.0
            if turnover_discrete.mean() > 0.5: sortino_discrete -= 1.0
            if (pos_discrete == 0).all(): sortino_discrete = -2.0

            # ========== 連續倉位 (Continuous) ==========
            pos_continuous = sig  # [-1, 1]
            
            turnover_continuous = torch.abs(pos_continuous - torch.roll(pos_continuous, 1))
            turnover_continuous[0] = 0.0

            pnl_continuous = pos_continuous * target - turnover_continuous * COST_RATE

            mu_c = pnl_continuous.mean()
            std_c = pnl_continuous.std() + 1e-6

            downside_c = pnl_continuous[pnl_continuous < 0]
            if downside_c.numel() > 5:
                down_std_c = downside_c.std() + 1e-6
                sortino_continuous = mu_c / down_std_c * 15.87
            else:
                sortino_continuous = mu_c / std_c * 15.87

            # 連續倉位懲罰
            if mu_c < 0: sortino_continuous = -2.0
            if (pos_continuous.abs() < 0.01).all(): sortino_continuous = -2.0

            # ========== 綜合獎勵 ==========
            # 離散權重 0.6，連續權重 0.4（偏向離散，因為學習效果更好）
            combined_reward = 0.6 * sortino_discrete + 0.4 * sortino_continuous

            rewards[i] = combined_reward

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
def calc_metrics(daily_ret, equity):
    """計算回測指標"""
    total_ret = equity[-1] - 1
    ann_ret = equity[-1] ** (252/len(equity)) - 1
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

def final_reality_check(miner, engine):
    print("\n" + "="*70)
    print(f"🔬 FINAL REALITY CHECK - {engine.symbol} (Out-of-Sample)")
    print("="*70)

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
    
    # ========== 離散倉位 (Discrete) ==========
    pos_discrete = np.sign(signal)  # {-1, 0, 1}
    turnover_d = np.abs(pos_discrete - np.roll(pos_discrete, 1))
    turnover_d[0] = 0
    daily_ret_d = pos_discrete * test_ret - turnover_d * COST_RATE
    equity_d = (1 + daily_ret_d).cumprod()
    metrics_d = calc_metrics(daily_ret_d, equity_d)
    
    # ========== 連續倉位 (Continuous) ==========
    pos_continuous = signal  # [-1, 1]
    turnover_c = np.abs(pos_continuous - np.roll(pos_continuous, 1))
    turnover_c[0] = 0
    daily_ret_c = pos_continuous * test_ret - turnover_c * COST_RATE
    equity_c = (1 + daily_ret_c).cumprod()
    metrics_c = calc_metrics(daily_ret_c, equity_c)

    # ========== 輸出結果 ==========
    print("-" * 70)
    print(f"📅 Test Period    : {test_dates.iloc[0].date()} ~ {test_dates.iloc[-1].date()}")
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
    
    # 為後續圖表和保存使用離散倉位的結果（主要指標）
    equity = equity_d
    daily_ret = daily_ret_d
    position = pos_discrete
    total_ret = metrics_d['total_ret']
    ann_ret = metrics_d['ann_ret']
    vol = metrics_d['vol']
    sharpe = metrics_d['sharpe']
    max_dd = metrics_d['max_dd']

    # 繪圖 - 同時顯示離散和連續倉位
    plt.style.use('bmh')
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 上圖：淨值曲線（三條線：離散、連續、基準）
    ax1 = axes[0]
    ax1.plot(test_dates, equity_d, label=f'Discrete (±1) | Sharpe {metrics_d["sharpe"]:.2f}', 
             linewidth=2, color='#2E86AB')
    ax1.plot(test_dates, equity_c, label=f'Continuous (Signal) | Sharpe {metrics_c["sharpe"]:.2f}', 
             linewidth=2, color='#28A745', linestyle='--')
    
    bench_equity = (1 + test_ret).cumprod()
    ax1.plot(test_dates, bench_equity, label=f'{engine.symbol} Buy & Hold', 
             alpha=0.5, linewidth=1.5, color='#A23B72')
    
    ax1.fill_between(test_dates, equity_d, alpha=0.2, color='#2E86AB')
    ax1.set_title(f'{engine.symbol} AlphaGPT Strategy Comparison | Test: {test_dates.iloc[0].date()} ~ {test_dates.iloc[-1].date()}', fontsize=14)
    ax1.set_ylabel('Cumulative Return')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 下圖：回撤比較
    ax2 = axes[1]
    dd_d = 1 - equity_d / np.maximum.accumulate(equity_d)
    dd_c = 1 - equity_c / np.maximum.accumulate(equity_c)
    ax2.fill_between(test_dates, -dd_d * 100, alpha=0.5, color='#2E86AB', label=f'Discrete DD (Max: {metrics_d["max_dd"]:.1%})')
    ax2.fill_between(test_dates, -dd_c * 100, alpha=0.5, color='#28A745', label=f'Continuous DD (Max: {metrics_c["max_dd"]:.1%})')
    ax2.set_title('Drawdown Comparison (%)', fontsize=12)
    ax2.set_ylabel('Drawdown %')
    ax2.set_xlabel('Date')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, f'{engine.symbol}_strategy_performance.png')
    plt.savefig(output_file, dpi=150)
    print(f"📈 Chart saved to '{output_file}'")

    # 保存策略（包含兩種倉位的指標）
    strategy_file = os.path.join(OUTPUT_DIR, f'{engine.symbol}_best_strategy.json')
    with open(strategy_file, 'w') as f:
        json.dump({
            'symbol': engine.symbol,
            'formula_tokens': miner.best_formula_tokens,
            'formula_readable': formula_str,
            'train_sortino': float(miner.best_sharpe),
            # 離散倉位指標
            'discrete': {
                'sharpe': float(metrics_d['sharpe']),
                'ann_return': float(metrics_d['ann_ret']),
                'max_drawdown': float(metrics_d['max_dd']),
                'win_rate': float(metrics_d['win_rate']),
                'profit_factor': float(metrics_d['profit_factor'])
            },
            # 連續倉位指標
            'continuous': {
                'sharpe': float(metrics_c['sharpe']),
                'ann_return': float(metrics_c['ann_ret']),
                'max_drawdown': float(metrics_c['max_dd']),
                'win_rate': float(metrics_c['win_rate']),
                'profit_factor': float(metrics_c['profit_factor'])
            }
        }, f, indent=2)
    print(f"💾 Strategy saved to '{strategy_file}'")
    
    # 保存詳細報告文本文件
    report_file = os.path.join(OUTPUT_DIR, f'{engine.symbol}_report.txt')
    test_start_str = str(test_dates.iloc[0].date())
    test_end_str = str(test_dates.iloc[-1].date())
    
    report_lines = [
        "=" * 70,
        f"🔬 FINAL REALITY CHECK - {engine.symbol} (Out-of-Sample)",
        "=" * 70,
        f"📜 Strategy Formula: {formula_str}",
        "-" * 70,
        f"📅 Test Period    : {test_start_str} ~ {test_end_str}",
        "-" * 70,
        f"{'指標':<18} {'離散倉位 (±1)':<20} {'連續倉位 (Signal)':<20}",
        "-" * 70,
        f"{'📈 Total Return':<18} {metrics_d['total_ret']:>18.2%} {metrics_c['total_ret']:>18.2%}",
        f"{'📈 Ann. Return':<18} {metrics_d['ann_ret']:>18.2%} {metrics_c['ann_ret']:>18.2%}",
        f"{'📊 Ann. Volatility':<18} {metrics_d['vol']:>18.2%} {metrics_c['vol']:>18.2%}",
        f"{'⭐ Sharpe Ratio':<18} {metrics_d['sharpe']:>18.2f} {metrics_c['sharpe']:>18.2f}",
        f"{'📉 Max Drawdown':<18} {metrics_d['max_dd']:>18.2%} {metrics_c['max_dd']:>18.2%}",
        f"{'🎯 Calmar Ratio':<18} {metrics_d['calmar']:>18.2f} {metrics_c['calmar']:>18.2f}",
        f"{'✅ Win Rate':<18} {metrics_d['win_rate']:>18.2%} {metrics_c['win_rate']:>18.2%}",
        f"{'💰 Profit Factor':<18} {metrics_d['profit_factor']:>18.2f} {metrics_c['profit_factor']:>18.2f}",
        "-" * 70,
        "",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"📝 Report saved to '{report_file}'")


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

    # 清理緩存文件
    if os.path.exists(engine.cache_path):
        os.remove(engine.cache_path)
        print(f"🗑️  Deleted cache file: {engine.cache_path}")
