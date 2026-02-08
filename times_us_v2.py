"""
AlphaGPT V2 - Anti-Overfitting Edition
美股因子挖掘與回測腳本（防過擬合版）

核心改進：
1. Walk-Forward 滾動驗證 - 多時間段驗證，確保因子穩健
2. 多目標獎勵機制 - Sortino + Sharpe + Return - MaxDD - Complexity
3. 更多因子 - ATR, RSI, CLV, 相對強弱 (RS)
4. 公式集成 (Ensemble) - 取 Top-K 公式的信號平均
5. 複雜度懲罰 - 奧卡姆剃刀原則

交易邏輯：
- T日收盤信號 → T+1開盤建倉 → 持有直到信號改變
- 收益計算: Close-to-Close 日收益

Usage:
    python times_us_v2.py --symbol SPY --start 2015-01-01 --end 2024-01-01
    python times_us_v2.py --symbol NVDA --iterations 600 --walk_forward
"""

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
# 🆕 混合精度訓練 (AMP) - 兼容不同 PyTorch 版本
from torch.cuda.amp import GradScaler  # GradScaler 在所有版本都在 cuda.amp 下
try:
    # PyTorch 2.0+ autocast 支持 device_type 參數
    from torch.amp import autocast
    AMP_DEVICE_TYPE_SUPPORTED = True
except ImportError:
    # PyTorch 1.x 使用舊 API
    from torch.cuda.amp import autocast
    AMP_DEVICE_TYPE_SUPPORTED = False
import os
import sys
import argparse
import json
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Numba 加速（可選，如果沒安裝則回退到純 Python）
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️ Numba not installed. Install with: pip install numba")

# Windows 編碼修復
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ==================== 輸出目錄 ====================
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================== 信號閾值配置 ====================
DEFAULT_SIGNAL_THRESHOLD = 0.1
DEFAULT_HYSTERESIS = 0.05         # 死區/滯後閾值，避免頻繁交易

# ==================== Numba 加速的倉位計算 ====================
if NUMBA_AVAILABLE:
    @njit
    def _discretize_position_numba(signal, threshold, hysteresis):
        """
        使用 Numba 加速的滯後倉位計算（編譯為 C 級別速度）
        """
        n = len(signal)
        pos = np.zeros(n, dtype=np.int32)
        
        close_threshold = threshold - hysteresis
        current_pos = 0
        
        for i in range(n):
            s = signal[i]
            
            if current_pos == 0:
                if s > threshold:
                    current_pos = 1
                elif s < -threshold:
                    current_pos = -1
            elif current_pos == 1:
                if s < -threshold:
                    current_pos = -1  # 翻空
                elif s < close_threshold:
                    current_pos = 0   # 平倉
                # else 保持持倉
            elif current_pos == -1:
                if s > threshold:
                    current_pos = 1   # 翻多
                elif s > -close_threshold:
                    current_pos = 0   # 平倉
                # else 保持持倉
                
            pos[i] = current_pos
            
        return pos
    
    @njit(parallel=True)
    def _discretize_batch_numba(signals, threshold, hysteresis):
        """
        🆕 Numba 批量加速版本（並行處理每個樣本）
        
        Args:
            signals: [B, T] 的 numpy 數組
            threshold: 開倉閾值
            hysteresis: 滯後閾值
        
        Returns:
            pos: [B, T] 的倉位數組
        """
        B, T = signals.shape
        pos = np.zeros((B, T), dtype=np.float32)
        close_threshold = threshold - hysteresis
        
        # parallel=True 會自動並行化這個外層循環
        for b in range(B):
            current_pos = 0
            for i in range(T):
                s = signals[b, i]
                
                if current_pos == 0:
                    if s > threshold:
                        current_pos = 1
                    elif s < -threshold:
                        current_pos = -1
                elif current_pos == 1:
                    if s < -threshold:
                        current_pos = -1
                    elif s < close_threshold:
                        current_pos = 0
                elif current_pos == -1:
                    if s > threshold:
                        current_pos = 1
                    elif s > -close_threshold:
                        current_pos = 0
                
                pos[b, i] = current_pos
        
        return pos


def _discretize_position_python(signal, threshold, hysteresis):
    """
    純 Python 版本的滯後倉位計算（Numba 不可用時的回退）
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    pos = np.zeros(n, dtype=np.int32)
    
    close_threshold = threshold - hysteresis
    current_pos = 0
    
    for i in range(n):
        s = signal[i]
        
        if current_pos == 0:
            if s > threshold:
                current_pos = 1
            elif s < -threshold:
                current_pos = -1
        elif current_pos == 1:
            if s < -threshold:
                current_pos = -1  # 翻空
            elif s < close_threshold:
                current_pos = 0   # 平倉
        elif current_pos == -1:
            if s > threshold:
                current_pos = 1   # 翻多
            elif s > -close_threshold:
                current_pos = 0   # 平倉
            
        pos[i] = current_pos
        
    return pos


def discretize_position(signal, threshold=DEFAULT_SIGNAL_THRESHOLD, hysteresis=DEFAULT_HYSTERESIS):
    """
    將連續信號 [-1, 1] 離散化為 {-1, 0, +1}
    
    增加滯後（死區）機制，避免信號在閾值附近震盪時頻繁交易：
    - 開倉閾值：threshold（例如 0.1）
    - 平倉閾值：threshold - hysteresis（例如 0.05）
    
    例如：
    - 當前無倉位：signal > 0.1 才開多
    - 當前持多倉：signal < 0.05 才平倉（而不是 < 0.1）
    
    自動選擇 Numba 加速版本或純 Python 版本
    """
    signal = np.asarray(signal, dtype=np.float64)
    
    if NUMBA_AVAILABLE:
        return _discretize_position_numba(signal, threshold, hysteresis)
    else:
        return _discretize_position_python(signal, threshold, hysteresis)


# ==================== GPU 端離散倉位計算（批量向量化） ====================
@torch.jit.script
def discretize_position_gpu_simple(signal: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    GPU 端簡化版離散倉位計算（無滯後，完全向量化）
    
    Args:
        signal: [B, T] 或 [T] 的信號張量，值域 [-1, 1]
        threshold: 開倉閾值
    
    Returns:
        pos: 與 signal 同形狀的倉位張量 {-1, 0, +1}
    
    這是最快的版本，適用於訓練時的快速評估
    """
    pos = torch.zeros_like(signal)
    pos = torch.where(signal > threshold, torch.ones_like(signal), pos)
    pos = torch.where(signal < -threshold, -torch.ones_like(signal), pos)
    return pos



def discretize_position_gpu_batch(signals: torch.Tensor, threshold: float, hysteresis: float, 
                                   use_hysteresis: bool = True) -> torch.Tensor:
    """
    批量離散倉位計算（帶滯後時使用 Numba CPU 加速）
    
    Args:
        signals: [B, T] 的批量信號張量（GPU 或 CPU）
        threshold: 開倉閾值
        hysteresis: 滯後閾值
        use_hysteresis: 是否使用滯後邏輯（False 時使用快速 GPU 向量化版本）
    
    Returns:
        pos: [B, T] 的批量倉位張量（與輸入同設備）
    
    策略：
    - use_hysteresis=False: 純 GPU 向量化（最快，但無滯後）
    - use_hysteresis=True: GPU→CPU (Numba)→GPU（稍慢，但與回測一致）
    """
    if not use_hysteresis:
        # 快速向量化版本（無滯後，純 GPU）
        return discretize_position_gpu_simple(signals, threshold)
    
    # ==================== 帶滯後版本：使用 Numba CPU 加速 ====================
    # 滯後邏輯需要串行狀態維護，GPU 不擅長，Numba 更快
    device = signals.device
    signals_np = signals.cpu().numpy().astype(np.float64)
    
    if NUMBA_AVAILABLE:
        # 使用 Numba 並行加速（C 級別速度）
        pos_np = _discretize_batch_numba(signals_np, threshold, hysteresis)
    else:
        # 回退到純 Python（較慢）
        B, T = signals_np.shape
        pos_np = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            pos_np[b] = _discretize_position_python(signals_np[b], threshold, hysteresis)
    
    # 傳回 GPU
    return torch.from_numpy(pos_np).float().to(device)


def create_output_folder(symbol, suffix=""):
    """創建帶時間戳的獨立輸出子文件夾"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"{symbol}_v2{suffix}_{timestamp}"
    folder_path = os.path.join(OUTPUT_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


# ==================== 配置 ====================
DEFAULT_SYMBOL = 'SPY'            # 默認：大盤 ETF（更適合尋找 Alpha）
START_DATE = '2015-01-01'         # 更長的歷史數據
END_DATE = '2024-01-01'           # 訓練結束
TEST_END_DATE = '2026-01-27'      # 測試結束

BATCH_SIZE = 512                  # 減小 batch size，提高穩定性
TRAIN_ITERATIONS = 300            # 訓練輪數
MAX_SEQ_LEN = 10                  # 限制公式長度（奧卡姆剃刀）
COST_RATE = 0.0010                # 交易成本（10 bps，適用於個股；SPY 可降至 5 bps）
MAX_LOOKBACK = 20                 # 預熱期天數（時序算子需要的歷史數據）

# Walk-Forward 配置
WALK_FORWARD_WINDOWS = 0          # 0 = 自動計算最大窗口數
TRAIN_YEARS = 3                   # 每個窗口訓練年數
VAL_YEARS = 1                     # 每個窗口驗證年數

# 多目標權重
REWARD_WEIGHTS = {
    'sortino': 0.25,       # 風險調整收益
    'sharpe': 0.10,        # 夏普比率
    'return': 0,        # 絕對收益
    'alpha_spy': 0.10,     # 超越大盤 (SPY) 的超額收益
    'alpha_target': 0.25,  # ⬆️ 超越目標 Buy & Hold（核心目標！）
    'max_dd': 0.2,        # 最大回撤懲罰
    'complexity': 0.10,    # 複雜度懲罰
}

# Ensemble 配置
TOP_K_FORMULAS = 10               # 保留 Top K 個公式
TOP_SAVE_COUNT = 3                # 保存 Top N 個公式為獨立策略

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


# ==================== 算子配置（精簡版，移除容易過擬合的算子） ====================
OPS_CONFIG = [
    ('ADD', lambda x, y: x + y, 2),
    ('SUB', lambda x, y: x - y, 2),
    ('MUL', lambda x, y: x * y, 2),
    ('DIV', lambda x, y: x / (y + 1e-6 * torch.sign(y + 1e-9)), 2),
    ('NEG', lambda x: -x, 1),
    ('ABS', lambda x: torch.abs(x), 1),
    ('SIGN', lambda x: torch.sign(x), 1),
    ('DELTA5', lambda x: _ts_delta(x, 5), 1),
    ('MA5', lambda x: _ts_decay_linear(x, 5), 1),
    ('MA10', lambda x: _ts_decay_linear(x, 10), 1),
    ('STD10', lambda x: _ts_zscore(x, 10), 1),
    ('RANK10', lambda x: _ts_rank(x, 10), 1),
    # 移除 MAX20, MIN20 等容易過擬合的極端值算子
]

# ==================== 因子配置（擴展版） ====================
# 原有因子 + 新增因子
FEATURES = [
    'RET',      # 日收益率
    'RET5',     # 5日收益率
    'VOL_CHG',  # 成交量變化
    'V_RET',    # 量價收益
    'TREND',    # 趨勢（相對 MA60）
    # 新增因子
    'ATR',      # 平均真實波幅（波動率）
    'RSI',      # 相對強弱指數
    'CLV',      # Close Location Value
    'RS',       # 相對強度（vs SPY）
    'MOM',      # 動量（20日）
    'VIX',      # 🆕 VIX 恐慌指數（市場情緒）
]

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
            batch_first=True, norm_first=True, dropout=0.1
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


# ==================== 數據引擎（擴展版） ====================
class USDataEngineV2:
    """
    增強版數據引擎：
    1. 支持更多因子（包含 VIX）
    2. 支持下載基準數據（SPY）用於相對強度計算和 Alpha 計算
    3. 支持 Walk-Forward 分割
    """
    def __init__(self, symbol, start_date, end_date, test_end_date, benchmark='SPY'):
        self.symbol = symbol
        self.benchmark_symbol = benchmark if symbol.upper() != 'SPY' else None
        self.start_date = start_date
        self.end_date = end_date
        self.test_end_date = test_end_date
        self.cache_path = os.path.join(OUTPUT_DIR, f'data_cache_{symbol}_v2.parquet')
        
        # Walk-Forward 時間窗口
        self.wf_windows = []
        
        # SPY 基準收益（用於 Alpha 計算）
        self.spy_ret = None
        
    def _download_data(self, symbol, start, end):
        """下載股票數據"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start, end=end, auto_adjust=True)
                if df.empty:
                    raise ValueError(f"No data for {symbol}")
                return df.reset_index()
            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2)
                else:
                    raise ValueError(f"Failed to download {symbol}: {e}")
        return None
    
    def load(self):
        """加載並處理數據"""
        print(f"🌐 Loading data for {self.symbol}...")
        
        # 下載主標的數據
        df = self._download_data(self.symbol, self.start_date, self.test_end_date)
        
        # 標準化列名
        col_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['date', 'index']:
                col_mapping[col] = 'date'
            elif col_lower in ['open', 'high', 'low', 'close', 'volume']:
                col_mapping[col] = col_lower
        df = df.rename(columns=col_mapping)
        
        # 數據清洗
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce').ffill().bfill()
        
        self.dates = pd.to_datetime(df['date'])
        
        close = df['close'].values.astype(np.float32)
        open_ = df['open'].values.astype(np.float32)
        high = df['high'].values.astype(np.float32)
        low = df['low'].values.astype(np.float32)
        vol = df['volume'].values.astype(np.float32)
        
        # ==================== 檢測停牌數據（ffill 造成的假數據） ====================
        # 如果連續 N 天收盤價完全相同，視為停牌/數據缺失
        price_change = np.diff(close, prepend=close[0])
        # 標記連續相同價格的天數
        consecutive_same = np.zeros_like(close, dtype=np.int32)
        count = 0
        for i in range(len(close)):
            if abs(price_change[i]) < 1e-8:  # 價格沒有變化
                count += 1
            else:
                count = 0
            consecutive_same[i] = count
        
        # 如果連續 3 天以上價格不變，標記為不可交易（可能是停牌）
        self.tradable_mask = consecutive_same < 3
        num_untradable = (~self.tradable_mask).sum()
        if num_untradable > 0:
            pct_untradable = num_untradable / len(close) * 100
            print(f"   ⚠️ Detected {num_untradable} potentially untradable days ({pct_untradable:.1f}%) due to stale prices")
        
        # 下載基準數據（SPY - 用於相對強度和 Alpha 計算）
        print(f"   📊 Downloading benchmark: SPY")
        try:
            spy_df = self._download_data('SPY', self.start_date, self.test_end_date)
            spy_df = spy_df.rename(columns={c: c.lower() for c in spy_df.columns})
            
            # 轉換為純日期字符串進行對齊（避免時區問題）
            spy_dates = pd.to_datetime(spy_df['date'])
            spy_date_strings = spy_dates.dt.strftime('%Y-%m-%d')
            main_date_strings = self.dates.dt.strftime('%Y-%m-%d')
            
            # 建立 SPY 日期 -> 數值的映射
            spy_dict = dict(zip(spy_date_strings, spy_df['close'].values))
            
            # 對齊到主數據日期
            spy_aligned = [spy_dict.get(d, np.nan) for d in main_date_strings]
            spy_close = pd.Series(spy_aligned).ffill().bfill().values.astype(np.float32)
        except Exception as e:
            print(f"   ⚠️ Failed to download SPY: {e}, using self as benchmark")
            spy_close = close.copy()
        
        # 計算 SPY 日收益（用於 Alpha 計算）
        spy_ret_arr = np.zeros_like(spy_close)
        spy_ret_arr[1:] = (spy_close[1:] - spy_close[:-1]) / (spy_close[:-1] + 1e-6)
        self.spy_ret = torch.from_numpy(spy_ret_arr).to(DEVICE)
        
        # 基準收盤價（用於相對強度因子）
        if self.benchmark_symbol:
            bench_close = spy_close
        else:
            bench_close = close.copy()
        
        # 下載 VIX 數據
        print(f"   📊 Downloading VIX index...")
        vix_available = False
        vix_close = None
        
        try:
            vix_df = self._download_data('^VIX', self.start_date, self.test_end_date)
            vix_df = vix_df.rename(columns={c: c.lower() for c in vix_df.columns})
            
            # 關鍵修復：VIX 使用 America/Chicago 時區，股票使用 America/New_York
            # 必須轉換為純日期字符串進行對齊，避免時區差異導致匹配失敗
            vix_dates = pd.to_datetime(vix_df['date'])
            vix_date_strings = vix_dates.dt.strftime('%Y-%m-%d')
            
            # 主數據也轉換為日期字符串
            main_date_strings = self.dates.dt.strftime('%Y-%m-%d')
            
            # 建立 VIX 日期 -> 數值的映射
            vix_dict = dict(zip(vix_date_strings, vix_df['close'].values))
            
            # 對齊到主數據日期
            vix_aligned = [vix_dict.get(d, np.nan) for d in main_date_strings]
            vix_series = pd.Series(vix_aligned)
            
            # ==================== 檢測 VIX 數據缺失比例 ====================
            vix_missing = vix_series.isna().sum()
            vix_missing_pct = vix_missing / len(self.dates) * 100
            
            if vix_missing_pct > 50:
                # 如果超過 50% 缺失，說明對齊失敗，使用滾動波動率
                raise ValueError(f"VIX alignment failed: {vix_missing_pct:.1f}% missing")
            
            if vix_missing_pct > 5:
                print(f"   ⚠️ VIX data has {vix_missing} missing days ({vix_missing_pct:.1f}%), filled with ffill/bfill")
            
            vix_close = vix_series.ffill().bfill().values.astype(np.float32)
            vix_available = True
            print(f"   ✅ VIX data loaded (mean: {vix_close.mean():.1f})")
            
        except Exception as e:
            print(f"   ⚠️ VIX download/alignment failed: {e}")
        
        # 如果 VIX 下載失敗或對齊失敗，使用滾動波動率作為替代
        if not vix_available:
            # 計算 20 日滾動標準差 * sqrt(252) ≈ 年化波動率
            daily_ret = np.zeros_like(close)
            daily_ret[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-6)
            rolling_vol = pd.Series(daily_ret).rolling(20, min_periods=5).std().fillna(0.01).values
            vix_close = (rolling_vol * np.sqrt(252) * 100).astype(np.float32)  # 轉為 VIX 量級（百分比）
            print(f"   📈 Using historical volatility as VIX proxy (mean: {vix_close.mean():.1f})")

        # ==================== 計算 split_idx ====================
        dates_naive = self.dates.dt.tz_localize(None) if self.dates.dt.tz is not None else self.dates
        end_date_dt = pd.to_datetime(self.end_date)
        split_mask = dates_naive < end_date_dt
        self.split_idx = split_mask.sum()

        # ==================== 因子計算 ====================
        
        # 1. 日收益率 (RET)
        ret = np.zeros_like(close)
        ret[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-6)

        # 2. 5日收益率 (RET5)
        ret5 = pd.Series(close).pct_change(5).fillna(0).values.astype(np.float32)

        # 3. 成交量變化 (VOL_CHG)
        vol_ma = pd.Series(vol).rolling(20, min_periods=1).mean().values
        vol_chg = np.zeros_like(vol)
        mask = vol_ma > 0
        vol_chg[mask] = vol[mask] / vol_ma[mask] - 1
        vol_chg = np.nan_to_num(vol_chg).astype(np.float32)

        # 4. 量價收益 (V_RET)
        v_ret = (ret * (vol_chg + 1)).astype(np.float32)

        # 5. 趨勢 (TREND) - 相對 MA60
        ma60 = pd.Series(close).rolling(60, min_periods=1).mean().values
        trend = np.zeros_like(close)
        mask = ma60 > 0
        trend[mask] = close[mask] / ma60[mask] - 1
        trend = np.nan_to_num(trend).astype(np.float32)

        # 6. ATR (Average True Range) - 標準化波動率
        tr1 = high - low
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        atr = pd.Series(tr).rolling(14, min_periods=1).mean().values.astype(np.float32)
        atr_norm = atr / (close + 1e-6)  # 標準化

        # 7. RSI (Relative Strength Index) - EMA 版本
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / (loss + 1e-6)
        rsi = (100 - 100 / (1 + rs)).fillna(50).values.astype(np.float32)
        rsi_norm = (rsi - 50) / 25.0  # 標準化到約 [-2, 2]

        # 8. CLV (Close Location Value) - 收盤價在當日範圍的位置
        hl_range = high - low + 1e-6
        clv = ((close - low) - (high - close)) / hl_range
        clv = np.nan_to_num(clv).astype(np.float32)

        # 9. RS (Relative Strength vs Benchmark)
        # 計算相對於基準的超額收益
        bench_ret = np.zeros_like(bench_close)
        bench_ret[1:] = (bench_close[1:] - bench_close[:-1]) / (bench_close[:-1] + 1e-6)
        rs_factor = ret - bench_ret  # 超額收益
        # 20日滾動超額收益
        rs_20 = pd.Series(rs_factor).rolling(20, min_periods=1).sum().values.astype(np.float32)

        # 10. MOM (Momentum) - 20日動量
        mom = pd.Series(close).pct_change(20).fillna(0).values.astype(np.float32)

        # 11. VIX - 恐慌指數（標準化）
        # VIX 通常在 10-80 範圍，標準化到 [-2, 2] 左右
        vix_norm = (vix_close - 20) / 10  # 以 20 為中心，10 為標準差
        vix_norm = np.clip(vix_norm, -3, 6).astype(np.float32)  # VIX 可能飆升到 60+
        
        # VIX 變化率（也是重要信號）
        vix_chg = pd.Series(vix_close).pct_change().fillna(0).values.astype(np.float32)

        # ==================== Robust Normalization ====================
        def robust_norm(x, split_idx):
            x = x.astype(np.float32)
            x_train = x[:split_idx]
            median = np.nanmedian(x_train)
            mad = np.nanmedian(np.abs(x_train - median)) + 1e-6
            res = (x - median) / mad
            return np.clip(res, -5, 5).astype(np.float32)

        # 構建特徵張量（11 個因子）
        self.feat_data = torch.stack([
            torch.from_numpy(robust_norm(ret, self.split_idx)).to(DEVICE),
            torch.from_numpy(robust_norm(ret5, self.split_idx)).to(DEVICE),
            torch.from_numpy(robust_norm(vol_chg, self.split_idx)).to(DEVICE),
            torch.from_numpy(robust_norm(v_ret, self.split_idx)).to(DEVICE),
            torch.from_numpy(robust_norm(trend, self.split_idx)).to(DEVICE),
            torch.from_numpy(robust_norm(atr_norm, self.split_idx)).to(DEVICE),
            torch.from_numpy(rsi_norm).to(DEVICE),  # RSI 已標準化
            torch.from_numpy(clv).to(DEVICE),  # CLV 已在 [-1, 1]
            torch.from_numpy(robust_norm(rs_20, self.split_idx)).to(DEVICE),
            torch.from_numpy(robust_norm(mom, self.split_idx)).to(DEVICE),
            torch.from_numpy(vix_norm).to(DEVICE),  # VIX 已標準化
        ])
        
        # 保存原始數據
        self.raw_close = torch.from_numpy(close).to(DEVICE)
        self.raw_open = torch.from_numpy(open_).to(DEVICE)
        self.benchmark_ret = torch.from_numpy(ret).to(DEVICE)  # Close-to-Close
        
        # 計算其他收益率（用於正確的交易模擬）
        # Open-to-Close: 今開 → 今收
        otc_ret = np.zeros_like(close)
        otc_ret = (close - open_) / (open_ + 1e-6)
        self.otc_ret = torch.from_numpy(otc_ret.astype(np.float32)).to(DEVICE)
        
        # Close-to-Open: 昨收 → 今開
        cto_ret = np.zeros_like(close)
        cto_ret[1:] = (open_[1:] - close[:-1]) / (close[:-1] + 1e-6)
        self.cto_ret = torch.from_numpy(cto_ret.astype(np.float32)).to(DEVICE)
        
        # 保存歸一化參數
        self.norm_params = {name: {
            'median': float(np.nanmedian(data[:self.split_idx])),
            'mad': float(np.nanmedian(np.abs(data[:self.split_idx] - np.nanmedian(data[:self.split_idx]))) + 1e-6)
        } for name, data in zip(
            ['ret', 'ret5', 'vol_chg', 'v_ret', 'trend', 'atr', 'rsi', 'clv', 'rs', 'mom', 'vix'],
            [ret, ret5, vol_chg, v_ret, trend, atr_norm, rsi_norm, clv, rs_20, mom, vix_norm]
        )}

        # ==================== 設置 Walk-Forward 窗口 ====================
        self._setup_walk_forward_windows()
        
        # 輸出信息
        print(f"✅ {self.symbol} Data Ready!")
        print(f"   Total: {len(df)} days | Train: {self.split_idx} | Test: {len(df) - self.split_idx}")
        print(f"   Features: {FEATURES}")
        if self.wf_windows:
            print(f"   Walk-Forward Windows: {len(self.wf_windows)}")
        
        return self
    
    def _setup_walk_forward_windows(self):
        """
        設置 Walk-Forward 滾動驗證窗口
        
        自動計算可用窗口數，充分利用歷史數據
        每個窗口: [Train: N年] → [Val: M年]
        
        如果最後有不足1年的數據，會合併到最後一個驗證窗口中
        """
        dates_naive = self.dates.dt.tz_localize(None) if self.dates.dt.tz is not None else self.dates
        
        start_dt = pd.to_datetime(self.start_date)
        end_dt = pd.to_datetime(self.end_date)
        
        # 計算可用年數
        total_years = (end_dt - start_dt).days / 365.25
        
        # 自動計算最大窗口數（如果 WALK_FORWARD_WINDOWS = 0）
        if WALK_FORWARD_WINDOWS == 0:
            # 計算能創建多少個不重疊的驗證窗口
            max_windows = int((total_years - TRAIN_YEARS) / VAL_YEARS)
            num_windows = max(1, min(max_windows, 8))  # 最多8個窗口，避免過多
        else:
            num_windows = WALK_FORWARD_WINDOWS
        
        # 從訓練結束日期往前推算窗口
        for i in range(num_windows):
            # 驗證期結束
            val_end = end_dt - pd.DateOffset(years=i * VAL_YEARS)
            # 驗證期開始
            val_start = val_end - pd.DateOffset(years=VAL_YEARS)
            # 訓練期開始
            train_start = val_start - pd.DateOffset(years=TRAIN_YEARS)
            
            # 確保訓練開始日期不早於數據開始日期
            if train_start < start_dt:
                continue
            
            # 轉換為索引
            train_start_idx = (dates_naive >= train_start).argmax()
            val_start_idx = (dates_naive >= val_start).argmax()
            val_end_idx = (dates_naive >= val_end).argmax()
            
            if val_end_idx > val_start_idx > train_start_idx:
                self.wf_windows.append({
                    'train_start': train_start_idx,
                    'train_end': val_start_idx,
                    'val_start': val_start_idx,
                    'val_end': val_end_idx,
                    'train_period': f"{dates_naive.iloc[train_start_idx].date()} ~ {dates_naive.iloc[val_start_idx-1].date()}",
                    'val_period': f"{dates_naive.iloc[val_start_idx].date()} ~ {dates_naive.iloc[val_end_idx-1].date()}"
                })
        
        # 反轉順序（從早到晚）
        self.wf_windows = self.wf_windows[::-1]
        
        # ==================== 處理剩餘數據 ====================
        # 如果最後一個驗證窗口結束後還有數據，合併到最後一個窗口
        if self.wf_windows:
            last_window = self.wf_windows[-1]
            last_val_end_idx = last_window['val_end']
            
            # 檢查是否還有剩餘數據（在 split_idx 之前）
            if last_val_end_idx < self.split_idx:
                remaining_days = self.split_idx - last_val_end_idx
                
                # 如果剩餘數據超過 60 天（約2個月），就合併到最後一個窗口
                if remaining_days > 60:
                    # 更新最後一個窗口的驗證期結束時間
                    last_window['val_end'] = self.split_idx
                    last_window['val_period'] = (
                        f"{dates_naive.iloc[last_window['val_start']].date()} ~ "
                        f"{dates_naive.iloc[self.split_idx-1].date()}"
                    )
                    print(f"   📌 Extended last Val window to include {remaining_days} extra days")


# ==================== 因子挖掘器 V2（防過擬合版） ====================
class DeepQuantMinerV2:
    """
    增強版因子挖掘器：
    1. Walk-Forward 驗證
    2. 多目標獎勵
    3. 複雜度懲罰
    4. Ensemble 策略
    """
    def __init__(self, engine: USDataEngineV2, signal_threshold=DEFAULT_SIGNAL_THRESHOLD, 
                 hysteresis=DEFAULT_HYSTERESIS, use_walk_forward=True, 
                 use_amp=True, use_compile=True):
        self.engine = engine
        self.signal_threshold = signal_threshold
        self.hysteresis = hysteresis
        self.use_walk_forward = use_walk_forward
        
        self.model = AlphaGPT().to(DEVICE)
        
        # ==================== 🆕 torch.compile 加速（PyTorch 2.0+） ====================
        self.use_compile = use_compile and hasattr(torch, 'compile') and DEVICE.type == 'cuda' and sys.platform != 'win32'
        if self.use_compile:
            try:
                # 使用 reduce-overhead 模式，適合小模型和頻繁調用
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print("✅ torch.compile enabled (reduce-overhead mode)")
            except Exception as e:
                print(f"⚠️ torch.compile failed: {e}, falling back to eager mode")
                self.use_compile = False
        else:
            if sys.platform == 'win32':
                print("⚠️ torch.compile disabled on Windows (Triton not supported)")
                
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=2e-4, weight_decay=1e-4)
        
        # ==================== 🆕 混合精度訓練 (AMP) ====================
        self.use_amp = use_amp and DEVICE.type == 'cuda'
        if self.use_amp:
            self.scaler = GradScaler()
            print("✅ AMP (Automatic Mixed Precision) enabled")
        else:
            self.scaler = None
        
        # 最佳公式追蹤
        self.best_score = -10.0
        self.best_formula_tokens = None
        
        # Top-K 公式（用於 Ensemble）
        self.top_k_formulas: List[Dict] = []
        
        # 訓練歷史
        self.history = []

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
        
        # ⚠️ 禁止 ABS 作為根節點（防止 RL 用 ABS 逃避做空）
        # 當 step == 0 時，禁止選擇 ABS
        if step == 0:
            # 找到 ABS 在 VOCAB 中的位置
            abs_idx = None
            for i, (name, _, _) in enumerate(OPS_CONFIG):
                if name == 'ABS':
                    abs_idx = len(FEATURES) + i
                    break
            if abs_idx is not None:
                mask[:, abs_idx] = float('-inf')
        
        return mask

    def solve_one(self, tokens, feat_data=None):
        """解析並執行因子公式"""
        if feat_data is None:
            feat_data = self.engine.feat_data
            
        stack = []
        try:
            for t in reversed(tokens):
                if t < len(FEATURES):
                    stack.append(feat_data[t].unsqueeze(0))
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
                if final.dim() == 2 and final.shape[0] == 1:
                    final = final.squeeze(0)
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

    def _calc_realistic_pnl(self, pos_signal, start_idx, end_idx):
        """
        計算正確的收益（向量化版本，GPU 加速）
        
        - 新建倉當天：Open-to-Close (開盤買入 → 收盤)
        - 持倉中：Close-to-Close (昨收 → 今收)
        - 平倉當天：Close-to-Open (昨收 → 今開賣出)
        """
        ctc_ret = self.engine.benchmark_ret[start_idx:end_idx]  # Close-to-Close
        otc_ret = self.engine.otc_ret[start_idx:end_idx]        # Open-to-Close
        cto_ret = self.engine.cto_ret[start_idx:end_idx]        # Close-to-Open
        
        # T 的信號決定 T+1 的倉位
        effective_pos = torch.zeros_like(pos_signal)
        effective_pos[1:] = pos_signal[:-1]
        
        # 計算昨天的倉位
        yesterday_pos = torch.zeros_like(effective_pos)
        yesterday_pos[1:] = effective_pos[:-1]
        
        # 判斷各種情況（向量化）
        is_holding = (effective_pos != 0) & (effective_pos == yesterday_pos)  # 持倉中
        is_opening = (effective_pos != 0) & (yesterday_pos == 0)              # 新建倉
        is_closing = (effective_pos == 0) & (yesterday_pos != 0)              # 平倉
        is_flipping = (effective_pos != 0) & (yesterday_pos != 0) & (effective_pos != yesterday_pos)  # 翻倉
        
        # 計算每日收益（向量化）
        pnl = torch.zeros_like(pos_signal)
        
        # 持倉中：昨收 → 今收
        pnl = torch.where(is_holding, effective_pos * ctc_ret, pnl)
        
        # 新建倉：今開 → 今收 - 成本
        pnl = torch.where(is_opening, effective_pos * otc_ret - COST_RATE, pnl)
        
        # 平倉：昨收 → 今開 - 成本
        pnl = torch.where(is_closing, yesterday_pos * cto_ret - COST_RATE, pnl)
        
        # 翻倉：平舊倉 + 建新倉 - 雙重成本
        pnl = torch.where(is_flipping, 
                          yesterday_pos * cto_ret + effective_pos * otc_ret - 2 * COST_RATE, 
                          pnl)
        
        return pnl, effective_pos

    def _calc_realistic_pnl_batch(self, pos_signals: torch.Tensor, start_idx: int, end_idx: int):
        """
        🆕 批量計算 PnL（完全向量化，GPU 加速）
        
        Args:
            pos_signals: [B, T] 的批量倉位信號
            start_idx, end_idx: 數據範圍
        
        Returns:
            pnl: [B, T] 的批量 PnL
            effective_pos: [B, T] 的有效倉位
        """
        B, T = pos_signals.shape
        
        ctc_ret = self.engine.benchmark_ret[start_idx:end_idx]  # [T]
        otc_ret = self.engine.otc_ret[start_idx:end_idx]        # [T]
        cto_ret = self.engine.cto_ret[start_idx:end_idx]        # [T]
        
        # 擴展為 [B, T]
        ctc_ret = ctc_ret.unsqueeze(0).expand(B, -1)
        otc_ret = otc_ret.unsqueeze(0).expand(B, -1)
        cto_ret = cto_ret.unsqueeze(0).expand(B, -1)
        
        # T 的信號決定 T+1 的倉位
        effective_pos = torch.zeros_like(pos_signals)
        effective_pos[:, 1:] = pos_signals[:, :-1]
        
        # 計算昨天的倉位
        yesterday_pos = torch.zeros_like(effective_pos)
        yesterday_pos[:, 1:] = effective_pos[:, :-1]
        
        # 判斷各種情況（批量向量化）
        is_holding = (effective_pos != 0) & (effective_pos == yesterday_pos)
        is_opening = (effective_pos != 0) & (yesterday_pos == 0)
        is_closing = (effective_pos == 0) & (yesterday_pos != 0)
        is_flipping = (effective_pos != 0) & (yesterday_pos != 0) & (effective_pos != yesterday_pos)
        
        # 計算每日收益（批量向量化）
        pnl = torch.zeros_like(pos_signals)
        
        pnl = torch.where(is_holding, effective_pos * ctc_ret, pnl)
        pnl = torch.where(is_opening, effective_pos * otc_ret - COST_RATE, pnl)
        pnl = torch.where(is_closing, yesterday_pos * cto_ret - COST_RATE, pnl)
        pnl = torch.where(is_flipping, 
                          yesterday_pos * cto_ret + effective_pos * otc_ret - 2 * COST_RATE, 
                          pnl)
        
        return pnl, effective_pos
    
    def _calc_metrics_for_period(self, factor, start_idx, end_idx):
        """計算指定時間段的策略指標"""
        daily_ret = self.engine.benchmark_ret[start_idx:end_idx]
        f = factor[start_idx:end_idx]
        
        if torch.isnan(f).all() or (f == 0).all() or f.numel() < 20:
            return None
        
        sig = torch.tanh(f)
        threshold = self.signal_threshold
        
        # 離散倉位
        pos_signal = torch.zeros_like(sig)
        pos_signal[sig > threshold] = 1
        pos_signal[sig < -threshold] = -1
        
        # 使用正確的收益計算
        pnl, effective_pos = self._calc_realistic_pnl(pos_signal, start_idx, end_idx)
        
        if pnl.numel() < 10:
            return None
        
        # 換手率（用於統計）
        turnover = torch.abs(effective_pos - torch.roll(effective_pos, 1))
        turnover[0] = torch.abs(effective_pos[0])
        
        # 計算指標
        equity = (1 + pnl).cumprod(dim=0)
        total_ret = (equity[-1] - 1).item()
        n_days = len(pnl)
        ann_ret = (equity[-1].item() ** (252 / n_days) - 1) if n_days > 0 else 0
        
        mu = pnl.mean().item()
        std = pnl.std().item() + 1e-6
        
        # Sharpe
        sharpe = (ann_ret - 0.02) / (std * np.sqrt(252) + 1e-6)
        
        # Sortino
        downside = pnl[pnl < 0]
        if downside.numel() > 5:
            down_std = downside.std().item() + 1e-6
            sortino = mu / down_std * 15.87
        else:
            sortino = mu / std * 15.87
        
        # Max Drawdown
        equity_np = equity.cpu().numpy()
        dd = 1 - equity_np / np.maximum.accumulate(equity_np)
        max_dd = np.max(dd)
        
        # Win Rate
        win_rate = (pnl > 0).float().mean().item()
        
        # 換手率
        avg_turnover = turnover.mean().item()
        
        return {
            'sortino': sortino,
            'sharpe': sharpe,
            'total_ret': total_ret,
            'ann_ret': ann_ret,
            'max_dd': max_dd,
            'win_rate': win_rate,
            'avg_turnover': avg_turnover,
            'mu': mu,
        }

    def backtest_single_window(self, factors, train_start, train_end):
        """
        🆕 向量化批量回測（GPU 加速版）
        
        相比原版優化：
        1. 批量離散倉位計算（GPU 端）
        2. 批量 PnL 計算
        3. 批量指標計算（減少 CPU 轉換）
        """
        B = factors.shape[0]
        if B == 0: 
            return torch.tensor([], device=DEVICE)
        
        # ==================== 預熱期處理 ====================
        warmup_offset = MAX_LOOKBACK
        effective_start = train_start + warmup_offset
        
        if effective_start >= train_end or (train_end - effective_start) < 30:
            warmup_offset = min(10, (train_end - train_start) // 4)
            effective_start = train_start + warmup_offset
        
        T = train_end - effective_start
        
        # 獲取這個時間段的收益數據
        daily_ret = self.engine.benchmark_ret[effective_start:train_end]  # [T]
        spy_ret_period = self.engine.spy_ret[effective_start:train_end]   # [T]
        
        # ==================== 批量處理因子 ====================
        f_batch = factors[:, effective_start:train_end]  # [B, T]
        
        # 檢查無效因子（向量化）
        is_all_nan = torch.isnan(f_batch).all(dim=1)  # [B]
        is_all_zero = (f_batch == 0).all(dim=1)       # [B]
        invalid_mask = is_all_nan | is_all_zero
        
        # 初始化獎勵
        rewards = torch.full((B,), -2.0, device=DEVICE)
        
        # 只處理有效的因子
        valid_mask = ~invalid_mask
        if not valid_mask.any():
            return torch.clamp(rewards, -3, 5)
        
        valid_indices = torch.where(valid_mask)[0]
        f_valid = f_batch[valid_mask]  # [B_valid, T]
        B_valid = f_valid.shape[0]
        
        # ==================== 批量信號和倉位計算（GPU 端） ====================
        sig_batch = torch.tanh(f_valid)  # [B_valid, T]
        
        # ==================== 批量離散倉位（帶滯後，與回測一致） ====================
        # 使用帶滯後的版本，確保訓練/回測行為一致
        pos_batch = discretize_position_gpu_batch(
            sig_batch, self.signal_threshold, self.hysteresis, use_hysteresis=True
        )  # [B_valid, T]
        
        # ==================== 批量 PnL 計算 ====================
        pnl_batch, effective_pos_batch = self._calc_realistic_pnl_batch(pos_batch, effective_start, train_end)
        
        # ==================== 批量指標計算（全在 GPU 上） ====================
        # 平均收益
        mu_batch = pnl_batch.mean(dim=1)  # [B_valid]
        std_batch = pnl_batch.std(dim=1) + 1e-6  # [B_valid]
        
        # 累積權益曲線
        equity_batch = (1 + pnl_batch).cumprod(dim=1)  # [B_valid, T]
        final_equity = equity_batch[:, -1]  # [B_valid]
        total_ret_batch = final_equity - 1  # [B_valid]
        
        # 年化收益
        n_days = T
        ann_ret_batch = torch.where(
            final_equity > 0,
            final_equity.pow(252.0 / n_days) - 1,
            torch.zeros_like(final_equity)
        )
        
        # Sharpe（批量）
        sharpe_batch = (ann_ret_batch - 0.02) / (std_batch * np.sqrt(252) + 1e-6)
        
        # Sortino（真實向量化版本 - 只懲罰下行波動）
        # 計算下行波動率：只取負收益的平方均值的根
        downside_pnl = torch.clamp(pnl_batch, max=0)  # 將正收益置為 0
        downside_std = torch.sqrt((downside_pnl ** 2).mean(dim=1)) + 1e-6
        sortino_batch = mu_batch / downside_std * 15.87
        
        # Max Drawdown（批量計算）
        running_max = torch.cummax(equity_batch, dim=1)[0]
        drawdown_batch = 1 - equity_batch / (running_max + 1e-8)
        max_dd_batch = drawdown_batch.max(dim=1)[0]  # [B_valid]
        
        # Return Score
        return_score_batch = torch.tanh(total_ret_batch * 5)
        
        # ==================== Alpha 計算（批量） ====================
        # 擴展 daily_ret 和 spy_ret 為 [B_valid, T]
        daily_ret_expanded = daily_ret.unsqueeze(0).expand(B_valid, -1)
        spy_ret_expanded = spy_ret_period.unsqueeze(0).expand(B_valid, -1)
        
        # Alpha vs SPY
        daily_excess_spy = pnl_batch - spy_ret_expanded
        alpha_spy_batch = daily_excess_spy.sum(dim=1)
        alpha_spy_score_batch = torch.tanh(alpha_spy_batch * 3)
        
        # Alpha vs Target
        daily_excess_target = pnl_batch - daily_ret_expanded
        alpha_target_batch = daily_excess_target.sum(dim=1)
        alpha_target_score_batch = torch.tanh(alpha_target_batch * 3)
        
        # ==================== 換手率（批量） ====================
        pos_shift = torch.roll(effective_pos_batch, 1, dims=1)
        pos_shift[:, 0] = 0
        turnover_batch = torch.abs(effective_pos_batch - pos_shift)
        avg_turnover_batch = turnover_batch.mean(dim=1)
        
        # 交易比率
        trade_ratio_batch = (effective_pos_batch != 0).float().mean(dim=1)
        
        # ==================== 批量獎勵計算 ====================
        reward_batch = torch.zeros(B_valid, device=DEVICE)
        
        # 1. Sortino
        sortino_clipped = torch.clamp(sortino_batch, -3, 5)
        reward_batch += REWARD_WEIGHTS['sortino'] * sortino_clipped
        
        # 2. Sharpe
        sharpe_clipped = torch.clamp(sharpe_batch, -2, 3)
        reward_batch += REWARD_WEIGHTS['sharpe'] * sharpe_clipped
        
        # 3. Return
        reward_batch += REWARD_WEIGHTS['return'] * return_score_batch * 2
        
        # 4. Alpha vs SPY
        reward_batch += REWARD_WEIGHTS['alpha_spy'] * alpha_spy_score_batch * 2
        
        # 5. Alpha vs Target
        reward_batch += REWARD_WEIGHTS['alpha_target'] * alpha_target_score_batch * 2
        
        # 跑輸 Buy & Hold 懲罰
        underperform_penalty = torch.where(
            alpha_target_batch < 0,
            torch.abs(alpha_target_batch) * 3,
            torch.zeros_like(alpha_target_batch)
        )
        reward_batch -= underperform_penalty
        
        # 6. Max Drawdown 懲罰
        dd_penalty = torch.where(
            max_dd_batch > 0.15,
            (max_dd_batch - 0.15) * 5,
            torch.zeros_like(max_dd_batch)
        )
        reward_batch -= REWARD_WEIGHTS['max_dd'] * dd_penalty
        
        # ==================== 懲罰項（批量） ====================
        # 負收益懲罰
        reward_batch = torch.where(mu_batch < 0, torch.full_like(reward_batch, -2.0), reward_batch)
        
        # 換手率過高懲罰
        turnover_penalty = torch.where(
            avg_turnover_batch > 0.25,
            (avg_turnover_batch - 0.25) * 3,
            torch.zeros_like(avg_turnover_batch)
        )
        reward_batch -= turnover_penalty
        
        # 幾乎不交易懲罰
        reward_batch = torch.where(trade_ratio_batch < 0.1, torch.full_like(reward_batch, -2.0), reward_batch)
        
        # ==================== 寫回獎勵 ====================
        rewards[valid_indices] = reward_batch

        return torch.clamp(rewards, -3, 5)

    def backtest_walk_forward(self, factors, token_seqs):
        """Walk-Forward 回測：在多個時間窗口驗證"""
        B = factors.shape[0]
        
        # ==================== 1. 計算基礎收益 (Raw Rewards) ====================
        if not self.engine.wf_windows or not self.use_walk_forward:
            # 標準模式：單窗口
            raw_rewards = self.backtest_single_window(factors, 0, self.engine.split_idx)
        else:
            # Walk-Forward 模式：多窗口聚合
            all_rewards = []
            
            for window in self.engine.wf_windows:
                # 在訓練期計算獎勵
                train_rewards = self.backtest_single_window(
                    factors, window['train_start'], window['train_end']
                )
                all_rewards.append(train_rewards)
            
            # 計算平均獎勵（所有窗口都要好才是真的好）
            stacked = torch.stack(all_rewards, dim=0)
            
            # 使用最小值（最保守策略）+ 平均值的加權組合
            min_rewards = stacked.min(dim=0)[0]
            mean_rewards = stacked.mean(dim=0)
            
            # 80% 最小值 + 20% 平均值（確保穩健性）
            raw_rewards = 0.8 * min_rewards + 0.2 * mean_rewards
        
        # ==================== 2. 統一應用複雜度懲罰（無論哪種模式都會執行）====================
        complexity_penalties = torch.zeros(B, device=DEVICE)
        for i in range(B):
            tokens = token_seqs[i].cpu().tolist()
            # 只計算操作符數量（特徵 token < len(FEATURES)，操作符 token >= len(FEATURES)）
            num_operators = sum(1 for t in tokens if t >= len(FEATURES))
            # 每多一個操作符懲罰 0.1
            complexity_penalties[i] = num_operators * 0.1
        
        final_rewards = raw_rewards - REWARD_WEIGHTS['complexity'] * complexity_penalties
        
        return torch.clamp(final_rewards, -3, 5)

    def _validate_on_oos(self, tokens):
        """在樣本外（測試集）驗證公式"""
        factor = self.solve_one(tokens)
        if factor is None:
            return None
        
        # 在驗證窗口上測試
        if self.engine.wf_windows and self.use_walk_forward:
            val_results = []
            for window in self.engine.wf_windows:
                metrics = self._calc_metrics_for_period(
                    factor, window['val_start'], window['val_end']
                )
                if metrics:
                    val_results.append(metrics)
            
            if not val_results:
                return None
            
            # 返回平均指標
            avg_metrics = {
                key: np.mean([r[key] for r in val_results])
                for key in val_results[0].keys()
            }
            avg_metrics['all_positive'] = all(r['total_ret'] > 0 for r in val_results)
            avg_metrics['window_results'] = val_results
            return avg_metrics
        else:
            # 使用標準測試集
            return self._calc_metrics_for_period(
                factor, self.engine.split_idx, len(factor)
            )

    def _update_top_k(self, tokens, score, train_metrics, val_metrics=None):
        """更新 Top-K 公式列表"""
        formula_str = self.decode(tokens)
        
        entry = {
            'tokens': tokens,
            'score': score,
            'formula': formula_str,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }
        
        # 檢查是否已存在相同公式
        existing_formulas = [f['formula'] for f in self.top_k_formulas]
        if formula_str in existing_formulas:
            return
        
        self.top_k_formulas.append(entry)
        
        # 按 score 排序，保留 Top K
        self.top_k_formulas.sort(key=lambda x: x['score'], reverse=True)
        self.top_k_formulas = self.top_k_formulas[:TOP_K_FORMULAS]
    
    def _print_topk_status(self, iteration):
        """打印 TopK 列表狀態"""
        if not self.top_k_formulas:
            return
        
        tqdm.write(f"\n{'='*100}")
        tqdm.write(f"📋 Iter {iteration} - Top {len(self.top_k_formulas)} Formulas:")
        tqdm.write(f"{'='*100}")
        
        for i, entry in enumerate(self.top_k_formulas):
            tm = entry.get('train_metrics', {})
            vm = entry.get('val_metrics', {})
            
            sortino = tm.get('sortino', 0) if tm else 0
            train_ret = tm.get('total_ret', 0) if tm else 0
            val_ret = vm.get('total_ret', 0) if vm else 0
            
            formula_short = entry['formula'][:60] + '...' if len(entry['formula']) > 60 else entry['formula']
            
            tqdm.write(f"  {i+1:2}. Score={entry['score']:.2f} | Sortino={sortino:.2f} | "
                      f"Train={train_ret:.1%} | Val={val_ret:.1%} | {formula_short}")
        
        tqdm.write(f"{'='*100}\n")

    def train(self):
        """
        🆕 訓練模型（支援 AMP 混合精度加速）
        """
        mode_str = "Walk-Forward" if self.use_walk_forward else "Standard"
        print(f"🚀 Training AlphaGPT V2 ({mode_str} Mode) for {self.engine.symbol}...")
        print(f"   MAX_SEQ_LEN={MAX_SEQ_LEN} | BATCH={BATCH_SIZE} | ITER={TRAIN_ITERATIONS}")
        print(f"   Features: {FEATURES}")
        print(f"   Acceleration: torch.compile={'✅' if self.use_compile else '❌'} | AMP={'✅' if self.use_amp else '❌'}")
        
        if self.use_walk_forward and self.engine.wf_windows:
            print(f"   Walk-Forward Windows:")
            for i, w in enumerate(self.engine.wf_windows):
                print(f"      [{i+1}] Train: {w['train_period']} | Val: {w['val_period']}")
        
        # 預先創建 arity_tens（避免每次迭代重複創建）
        arity_tens = torch.zeros(VOCAB_SIZE, dtype=torch.long, device=DEVICE)
        for k, v in OP_ARITY_MAP.items(): 
            arity_tens[k] = v
        
        pbar = tqdm(range(TRAIN_ITERATIONS))

        for iteration in pbar:
            B = BATCH_SIZE
            open_slots = torch.ones(B, dtype=torch.long, device=DEVICE)
            log_probs, tokens = [], []
            curr_inp = torch.zeros((B, 1), dtype=torch.long, device=DEVICE)

            # ==================== 🆕 AMP 前向傳播 ====================
            # 兼容不同 PyTorch 版本的 autocast API
            if AMP_DEVICE_TYPE_SUPPORTED:
                amp_context = autocast(device_type='cuda', enabled=self.use_amp)
            else:
                amp_context = autocast(enabled=self.use_amp)
            
            with amp_context:
                for step in range(MAX_SEQ_LEN):
                    logits, val = self.model(curr_inp)
                    mask = self.get_strict_mask(open_slots, step)
                    
                    # 確保 logits 是 float32（Categorical 需要）
                    logits_fp32 = logits.float() if self.use_amp else logits
                    dist = Categorical(logits=(logits_fp32 + mask))
                    action = dist.sample()

                    log_probs.append(dist.log_prob(action))
                    tokens.append(action)
                    curr_inp = torch.cat([curr_inp, action.unsqueeze(1)], dim=1)

                    is_op = action >= len(FEATURES)
                    delta = torch.full((B,), -1, device=DEVICE)
                    op_delta = arity_tens[action] - 1
                    delta = torch.where(is_op, op_delta, delta)
                    delta[open_slots == 0] = 0
                    open_slots += delta

            seqs = torch.stack(tokens, dim=1)

            with torch.no_grad():
                f_vals, valid_mask = self.solve_batch(seqs)
                valid_idx = torch.where(valid_mask)[0]
                rewards = torch.full((B,), -1.0, device=DEVICE)

                if len(valid_idx) > 0:
                    # 使用 Walk-Forward 或標準回測
                    bt_scores = self.backtest_walk_forward(f_vals[valid_idx], seqs[valid_idx])
                    rewards[valid_idx] = bt_scores

                    # 只處理 Top 分數的公式（按分數排序，取前 N 個檢查）
                    topk_updated = False
                    min_topk_score = self.top_k_formulas[-1]['score'] if self.top_k_formulas else -999
                    
                    # 按分數排序，找出候選公式
                    sorted_indices = torch.argsort(bt_scores, descending=True)
                    candidates_checked = 0
                    max_candidates_per_iter = min(100, max(30, B // 10))
                    
                    for sorted_idx in sorted_indices:
                        if candidates_checked >= max_candidates_per_iter:
                            break
                        
                        score = bt_scores[sorted_idx].item()
                        
                        if score <= 0:
                            break
                        if len(self.top_k_formulas) >= TOP_K_FORMULAS and score <= min_topk_score:
                            break
                        
                        tokens_list = seqs[valid_idx[sorted_idx]].cpu().tolist()
                        formula_str = self.decode(tokens_list)
                        
                        existing_formulas = [f['formula'] for f in self.top_k_formulas]
                        if formula_str in existing_formulas:
                            continue
                        
                        candidates_checked += 1
                        
                        factor = self.solve_one(tokens_list)
                        if factor is not None:
                            train_metrics = self._calc_metrics_for_period(
                                factor, 0, self.engine.split_idx
                            )
                            if train_metrics:
                                val_metrics = self._validate_on_oos(tokens_list)
                                self._update_top_k(tokens_list, score, train_metrics, val_metrics)
                                topk_updated = True
                                if self.top_k_formulas:
                                    min_topk_score = self.top_k_formulas[-1]['score']
                    
                    # 更新全局最佳
                    best_sub_idx = torch.argmax(bt_scores)
                    current_best_score = bt_scores[best_sub_idx].item()
                    
                    if current_best_score > self.best_score:
                        self.best_score = current_best_score
                        self.best_formula_tokens = seqs[valid_idx[best_sub_idx]].cpu().tolist()
                        
                        formula_str = self.decode(self.best_formula_tokens)
                        
                        factor = self.solve_one(self.best_formula_tokens)
                        train_metrics = self._calc_metrics_for_period(
                            factor, 0, self.engine.split_idx
                        ) if factor is not None else None
                        val_metrics = self._validate_on_oos(self.best_formula_tokens)
                        
                        self.history.append({
                            'iteration': iteration + 1,
                            'score': self.best_score,
                            'train_metrics': train_metrics,
                            'val_metrics': val_metrics,
                            'formula_tokens': self.best_formula_tokens,
                            'formula_str': formula_str
                        })
                    
                    if topk_updated and (iteration + 1) % 10 == 0:
                        self._print_topk_status(iteration + 1)

            # ==================== 🆕 AMP 反向傳播 ====================
            adv = rewards - rewards.mean()
            
            # 確保 log_probs 是 float32
            log_probs_tensor = torch.stack(log_probs, 1).float()
            loss = -(log_probs_tensor.sum(1) * adv).mean()

            self.opt.zero_grad()
            
            if self.use_amp and self.scaler is not None:
                # 使用 GradScaler 進行混合精度訓練
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()

            pbar.set_postfix({
                'Valid': f"{len(valid_idx)/B:.1%}", 
                'Best': f"{self.best_score:.2f}",
                'TopK': len(self.top_k_formulas)
            })
        
        self._print_training_summary()

    def decode(self, tokens=None):
        if tokens is None: 
            tokens = self.best_formula_tokens
        if tokens is None: 
            return "N/A"
        stream = list(tokens)
        def _parse():
            if not stream: return ""
            t = stream.pop(0)
            if t < len(FEATURES): 
                return FEATURES[t]
            args = [_parse() for _ in range(OP_ARITY_MAP[t])]
            return f"{VOCAB[t]}({','.join(args)})"
        try: 
            return _parse()
        except: 
            return "Invalid"

    def _print_training_summary(self):
        """打印訓練總結"""
        print("\n" + "=" * 100)
        print("📊 TRAINING SUMMARY")
        print("=" * 100)
        
        if not self.history:
            print("⚠️  No valid formula found!")
            return
        
        print(f"\n🏆 Best Formula: {self.decode()}")
        print(f"   Score: {self.best_score:.4f}")
        
        if self.history[-1].get('train_metrics'):
            m = self.history[-1]['train_metrics']
            print(f"   Train: Sortino={m['sortino']:.2f} | Sharpe={m['sharpe']:.2f} | "
                  f"Return={m['total_ret']:.1%} | MaxDD={m['max_dd']:.1%}")
        
        if self.history[-1].get('val_metrics'):
            m = self.history[-1]['val_metrics']
            all_pos = m.get('all_positive', False)
            print(f"   Val:   Sortino={m['sortino']:.2f} | Sharpe={m['sharpe']:.2f} | "
                  f"Return={m['total_ret']:.1%} | AllPositive={all_pos}")
        
        # 打印 Top-K
        if self.top_k_formulas:
            print(f"\n📋 Top {len(self.top_k_formulas)} Formulas (Ensemble Candidates):")
            print("-" * 100)
            for i, entry in enumerate(self.top_k_formulas):
                tm = entry.get('train_metrics', {})
                vm = entry.get('val_metrics', {})
                train_ret = tm.get('total_ret', 0) if tm else 0
                val_ret = vm.get('total_ret', 0) if vm else 0
                print(f"   [{i+1}] Score={entry['score']:.2f} | Train={train_ret:.1%} | "
                      f"Val={val_ret:.1%} | {entry['formula']}")
        
        print("=" * 100)

    def get_ensemble_signal(self, factor_list=None):
        """獲取 Ensemble 信號（多公式平均）"""
        if factor_list is None:
            # 使用 Top-K 公式
            factor_list = []
            for entry in self.top_k_formulas:
                f = self.solve_one(entry['tokens'])
                if f is not None:
                    factor_list.append(f)
        
        if not factor_list:
            return None
        
        # 取平均
        stacked = torch.stack(factor_list, dim=0)
        avg_factor = stacked.mean(dim=0)
        return avg_factor
    
    def calculate_score(self, factor):
        """
        計算因子的 Score（使用與訓練相同的多目標獎勵函數）
        
        用於評估 Ensemble 或任意因子的表現
        """
        if factor is None:
            return None
        
        # 使用 Walk-Forward 窗口計算
        if self.use_walk_forward and self.engine.wf_windows:
            window_scores = []
            window_metrics = []
            
            for window in self.engine.wf_windows:
                train_start = window['train_start']
                train_end = window['train_end']
                
                # 計算該窗口的指標
                metrics = self._calc_score_for_period(factor, train_start, train_end)
                if metrics:
                    window_scores.append(metrics['score'])
                    window_metrics.append(metrics)
            
            if not window_scores:
                return None
            
            # 80% 最小值 + 20% 平均值
            min_score = min(window_scores)
            mean_score = sum(window_scores) / len(window_scores)
            final_score = 0.8 * min_score + 0.2 * mean_score
            
            return {
                'score': final_score,
                'min_score': min_score,
                'mean_score': mean_score,
                'window_scores': window_scores,
                'window_metrics': window_metrics,
            }
        else:
            # 標準模式：使用整個訓練集
            metrics = self._calc_score_for_period(factor, 0, self.engine.split_idx)
            if metrics:
                return {
                    'score': metrics['score'],
                    'min_score': metrics['score'],
                    'mean_score': metrics['score'],
                    'window_scores': [metrics['score']],
                    'window_metrics': [metrics],
                }
            return None
    
    def _calc_score_for_period(self, factor, start_idx, end_idx):
        """計算指定時間段的 Score（多目標獎勵）"""
        daily_ret = self.engine.benchmark_ret[start_idx:end_idx]
        f = factor[start_idx:end_idx]
        
        if torch.isnan(f).all() or (f == 0).all() or f.numel() < 20:
            return None
        
        sig = torch.tanh(f)
        threshold = self.signal_threshold
        
        # 離散倉位
        pos_signal = torch.zeros_like(sig)
        pos_signal[sig > threshold] = 1
        pos_signal[sig < -threshold] = -1
        
        # 使用正確的收益計算
        pnl, effective_pos = self._calc_realistic_pnl(pos_signal, start_idx, end_idx)
        
        # 換手率（用於統計）
        turnover = torch.abs(effective_pos - torch.roll(effective_pos, 1))
        turnover[0] = torch.abs(effective_pos[0])
        
        if pnl.numel() < 10:
            return None
        
        mu = pnl.mean().item()
        std = pnl.std().item() + 1e-6
        
        # Sortino
        downside = pnl[pnl < 0]
        if downside.numel() > 5:
            down_std = downside.std().item() + 1e-6
            sortino = mu / down_std * 15.87
        else:
            sortino = mu / (std + 1e-6) * 15.87
        
        # Sharpe
        equity = (1 + pnl).cumprod(dim=0)
        n_days = len(pnl)
        ann_ret = (equity[-1].item() ** (252 / n_days) - 1) if n_days > 0 else 0
        sharpe = (ann_ret - 0.02) / (std * np.sqrt(252) + 1e-6)
        
        # Max Drawdown
        equity_np = equity.cpu().numpy()
        dd = 1 - equity_np / np.maximum.accumulate(equity_np)
        max_dd = float(np.max(dd))
        
        # Return Score
        total_ret = equity[-1].item() - 1
        return_score = float(np.tanh(total_ret * 5))
        
        # ==================== 改進的 Alpha 計算（每日超額收益累加） ====================
        # Alpha vs SPY - 使用每日超額收益的累加（反映每天的超額能力）
        spy_ret_period = self.engine.spy_ret[start_idx:end_idx]
        daily_excess_spy = pnl - spy_ret_period  # 每日超額收益
        alpha_spy = daily_excess_spy.sum().item()  # 累計超額收益
        alpha_spy_score = float(np.tanh(alpha_spy * 3))
        
        # 同時保留總收益差（用於報告）
        spy_equity = (1 + spy_ret_period).cumprod(dim=0)
        spy_total_ret = spy_equity[-1].item() - 1 if spy_equity.numel() > 0 else 0
        
        # Alpha vs Target - 使用每日超額收益的累加
        daily_excess_target = pnl - daily_ret  # 每日超額收益
        alpha_target = daily_excess_target.sum().item()  # 累計超額收益
        alpha_target_score = float(np.tanh(alpha_target * 3))
        
        # 同時保留總收益差（用於報告）
        target_equity = (1 + daily_ret).cumprod(dim=0)
        target_total_ret = target_equity[-1].item() - 1 if target_equity.numel() > 0 else 0
        
        # ==================== 計算 Score ====================
        score = 0.0
        
        # 1. Sortino
        score += REWARD_WEIGHTS['sortino'] * max(-3, min(5, sortino))
        
        # 2. Sharpe
        score += REWARD_WEIGHTS['sharpe'] * max(-2, min(3, sharpe))
        
        # 3. Return
        score += REWARD_WEIGHTS['return'] * return_score * 2
        
        # 4. Alpha vs SPY（超越大盤）
        score += REWARD_WEIGHTS['alpha_spy'] * alpha_spy_score * 2
        
        # 5. Alpha vs Target（超越目標標的）🆕
        score += REWARD_WEIGHTS['alpha_target'] * alpha_target_score * 2
        
        # ⚠️ 硬性懲罰：跑輸 Buy & Hold 直接扣分
        if alpha_target < 0:
            underperform_penalty = abs(alpha_target) * 3
            score -= underperform_penalty
        
        # 6. MaxDD 懲罰
        if max_dd > 0.15:
            score -= REWARD_WEIGHTS['max_dd'] * (max_dd - 0.15) * 5
        
        # 懲罰項
        if mu < 0:
            score = -2.0
        
        avg_turnover = turnover.mean().item()
        if avg_turnover > 0.25:
            score -= (avg_turnover - 0.25) * 3
        
        trade_ratio = (effective_pos != 0).float().mean().item()
        if trade_ratio < 0.1:
            score = -2.0
        
        return {
            'score': max(-3, min(5, score)),
            'sortino': sortino,
            'sharpe': sharpe,
            'total_ret': total_ret,
            'ann_ret': ann_ret,
            'max_dd': max_dd,
            'alpha_spy': alpha_spy,           # 超額收益 vs SPY
            'alpha_target': alpha_target,     # 🆕 超額收益 vs 目標標的
            'spy_ret': spy_total_ret,         # SPY 收益
            'target_ret': target_total_ret,   # 目標標的收益
            'avg_turnover': avg_turnover,
            'trade_ratio': trade_ratio,
        }


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


def final_reality_check_v2(miner: DeepQuantMinerV2, engine: USDataEngineV2, output_folder: str):
    """最終回測報告 V2"""
    threshold = miner.signal_threshold
    hysteresis = miner.hysteresis
    
    print("\n" + "="*80)
    print(f"🔬 FINAL REALITY CHECK V2 - {engine.symbol} (Out-of-Sample)")
    print("="*80)

    if miner.best_formula_tokens is None:
        print("❌ No valid formula found!")
        return
    
    formula_str = miner.decode()
    print(f"📜 Best Formula: {formula_str}")
    print(f"📊 Signal Threshold: ±{threshold} (Hysteresis: {hysteresis})")

    # ==================== 準備數據 ====================
    split = engine.split_idx
    test_dates = engine.dates[split:]
    test_ctc_ret = engine.benchmark_ret[split:].cpu().numpy()  # Close-to-Close
    test_otc_ret = engine.otc_ret[split:].cpu().numpy()        # Open-to-Close
    test_cto_ret = engine.cto_ret[split:].cpu().numpy()        # Close-to-Open

    def run_backtest(factor, label):
        # 計算整個數據期的信號（包含訓練期），以獲取 split-1 那天的仓位
        f_all = factor.cpu().numpy()
        signal_all = np.tanh(f_all)
        pos_all = discretize_position(signal_all, threshold, hysteresis)
        
        # 測試期信號
        f_test = f_all[split:]
        signal = signal_all[split:]
        pos_signal = pos_all[split:]
        
        # 設置 effective_pos，第 0 天使用 split-1 那天的仓位
        effective_pos = np.zeros(len(pos_signal), dtype=np.int32)
        effective_pos[0] = pos_all[split - 1] if split > 0 else 0  # ← 前一天的仓位
        effective_pos[1:] = pos_signal[:-1]
        
        # 使用正確的收益計算（注意：必須是 float64，不能用 zeros_like(pos_signal) 因為那是 int32）
        daily_ret = np.zeros(len(pos_signal), dtype=np.float64)
        
        # 處理第 0 天（T-1 收盤產生信號 → T 開盤買入）
        # 買入價 = Open[0]，收益 = Open-to-Close
        if effective_pos[0] != 0:
            daily_ret[0] = effective_pos[0] * test_otc_ret[0] - COST_RATE  # 建倉成本
        
        for i in range(1, len(pos_signal)):
            today_pos = effective_pos[i]
            yesterday_pos = effective_pos[i - 1]
            
            if today_pos == 0:
                if yesterday_pos != 0:
                    # 平倉：昨收 → 今開
                    daily_ret[i] = yesterday_pos * test_cto_ret[i] - COST_RATE
            else:
                if yesterday_pos == 0:
                    # 新建倉：今開 → 今收
                    daily_ret[i] = today_pos * test_otc_ret[i] - COST_RATE
                elif yesterday_pos == today_pos:
                    # 持倉中：昨收 → 今收
                    daily_ret[i] = today_pos * test_ctc_ret[i]
                else:
                    # 翻倉：平舊倉 + 建新倉
                    daily_ret[i] = yesterday_pos * test_cto_ret[i] + today_pos * test_otc_ret[i] - 2 * COST_RATE
        
        equity = (1 + daily_ret).cumprod()
        metrics = calc_metrics(daily_ret, equity)
        
        return {
            'label': label,
            'metrics': metrics,
            'equity': equity,
            'position': effective_pos,
            'signal': signal,
        }

    # ==================== 回測所有 Top K 公式 ====================
    top_results = []
    for i, entry in enumerate(miner.top_k_formulas[:TOP_SAVE_COUNT]):
        factor_i = miner.solve_one(entry['tokens'])
        if factor_i is not None:
            result_i = run_backtest(factor_i, f"Top{i+1}")
            result_i['formula'] = entry['formula']
            result_i['score'] = entry['score']
            top_results.append(result_i)
    
    # Best (Top 1)
    result_single = top_results[0] if top_results else None
    if result_single is None:
        print("❌ No valid formula found!")
        return
    
    # Ensemble
    factor_ensemble = miner.get_ensemble_signal()
    result_ensemble = run_backtest(factor_ensemble, "Ensemble") if factor_ensemble is not None else None

    # Buy & Hold（與策略相同：start_date 開盤買入）
    # - 第 0 天：Open[0] 買入 → Close[0] 持有（Open-to-Close）
    # - 第 1 天起：持倉（Close-to-Close）
    # 
    # 這樣策略和 Buy & Hold 的買入價完全一致（都是 Open[0]）
    bh_ret = np.zeros(len(test_ctc_ret), dtype=np.float64)
    if len(bh_ret) > 0:
        # 第 0 天：Open 買入 → Close
        bh_ret[0] = test_otc_ret[0]
        # 第 1 天起：持倉（Close → Close）
        if len(bh_ret) > 1:
            bh_ret[1:] = test_ctc_ret[1:]
    bench_equity = (1 + bh_ret).cumprod()
    bench_total_ret = bench_equity[-1] - 1

    # ==================== 打印結果 ====================
    print("-" * 80)
    print(f"📅 Test Period: {test_dates.iloc[0].date()} ~ {test_dates.iloc[-1].date()}")
    print(f"📊 Buy & Hold: {bench_total_ret:.2%}")
    print("-" * 80)
    
    # 打印所有 Top 公式的結果
    header_parts = [f"{'指標':<18}"]
    for r in top_results:
        header_parts.append(f"{r['label']:>12}")
    if result_ensemble:
        header_parts.append(f"{'Ensemble':>12}")
    print(" ".join(header_parts))
    print("-" * 80)
    
    m_s = result_single['metrics']
    
    def print_metric_row(name, key, is_pct=True):
        parts = [f"{name:<18}"]
        for r in top_results:
            val = r['metrics'][key]
            if is_pct:
                parts.append(f"{val:>11.2%}")
            else:
                parts.append(f"{val:>11.2f}")
        if result_ensemble:
            val = result_ensemble['metrics'][key]
            if is_pct:
                parts.append(f"{val:>11.2%}")
            else:
                parts.append(f"{val:>11.2f}")
        print(" ".join(parts))
    
    print_metric_row("📈 Total Return", 'total_ret', True)
    print_metric_row("📈 Ann. Return", 'ann_ret', True)
    print_metric_row("📊 Volatility", 'vol', True)
    print_metric_row("⭐ Sharpe", 'sharpe', False)
    print_metric_row("📉 Max Drawdown", 'max_dd', True)
    print_metric_row("🎯 Calmar", 'calmar', False)
    print_metric_row("✅ Win Rate", 'win_rate', True)
    print("-" * 80)

    # 倉位統計 (Top 1)
    pos = result_single['position']
    hold_ratio = (pos == 0).sum() / len(pos) * 100
    print(f"📊 Position (Top1): Long {(pos==1).sum()} | Short {(pos==-1).sum()} | Hold {(pos==0).sum()} ({hold_ratio:.1f}%)")

    # ==================== 繪圖：Top 公式對比 ====================
    plt.style.use('bmh')
    
    # 6行子圖: 淨值曲線 | Top1 Position | Top2 Position | Top3 Position | Ensemble Position | Drawdown
    fig, axes = plt.subplots(6, 1, figsize=(14, 18), gridspec_kw={'height_ratios': [3, 1.2, 1.2, 1.2, 1.2, 2]})
    
    # 顏色配置
    colors = ['#2E86AB', '#28A745', '#F39C12', '#E74C3C', '#9B59B6', '#1ABC9C']

    # 第1圖：淨值曲線 - 所有 Top 公式對比
    ax1 = axes[0]
    
    for i, r in enumerate(top_results):
        m = r['metrics']
        label = f"{r['label']}: {r['formula'][:25]}... | Sharpe {m['sharpe']:.2f}"
        ax1.plot(test_dates, r['equity'], label=label, linewidth=2 if i == 0 else 1.5, 
                 color=colors[i % len(colors)], alpha=1.0 if i == 0 else 0.7)
    
    if result_ensemble:
        m_e = result_ensemble['metrics']
        ax1.plot(test_dates, result_ensemble['equity'], 
                 label=f'Ensemble (Top{len(miner.top_k_formulas)}) | Sharpe {m_e["sharpe"]:.2f}', 
                 linewidth=2, color='#34495E', linestyle='--')
    
    ax1.plot(test_dates, bench_equity, label=f'{engine.symbol} Buy & Hold ({bench_total_ret:.1%})', 
             alpha=0.5, linewidth=1.5, color='#A23B72', linestyle=':')
    
    ax1.set_title(f'{engine.symbol} AlphaGPT V2 - Top Formulas Comparison | Test: {test_dates.iloc[0].date()} ~ {test_dates.iloc[-1].date()}', fontsize=14)
    ax1.set_ylabel('Cumulative Return')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 第2-4圖：Top 1-3 的 Position + Signal
    def plot_position_with_signal(ax, position, signal, label, color, formula_short, threshold):
        """繪製 Position 區域 + Signal 曲線
        
        - Signal: 公式的實際輸出值（連續曲線，通常在 -1 到 1 之間）
        - Position: 離散化後的倉位方向（+1, 0, -1 的區域填充）
        """
        # 繪製 Signal 曲線（實際公式結果）
        ax.plot(test_dates, signal, color=color, alpha=0.8, linewidth=1.2, label='Signal')
        
        # 繪製 Position 區域（半透明填充）
        ax.fill_between(test_dates, position, step='mid', alpha=0.3, color=color, label='Position')
        
        # 基準線
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(y=1, color='green', linestyle=':', linewidth=0.5, alpha=0.5)
        ax.axhline(y=-1, color='red', linestyle=':', linewidth=0.5, alpha=0.5)
        
        # 閾值線（顯示信號轉換為倉位的臨界點）
        ax.axhline(y=threshold, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.axhline(y=-threshold, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        
        long_cnt = (position == 1).sum()
        short_cnt = (position == -1).sum()
        hold_cnt = (position == 0).sum()
        hold_pct = hold_cnt / len(position) * 100
        
        ax.set_title(f'{label}: {formula_short} | L:{long_cnt} S:{short_cnt} H:{hold_cnt}({hold_pct:.0f}%)', fontsize=10)
        ax.set_ylabel('Signal/Pos')
        ax.set_ylim(-1.5, 1.5)
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
    
    # Top 1 Position + Signal
    plot_position_with_signal(axes[1], top_results[0]['position'], top_results[0]['signal'], 
                              'Top1', colors[0], top_results[0]['formula'][:35] + '...', threshold)
    
    # Top 2 Position + Signal
    if len(top_results) > 1:
        plot_position_with_signal(axes[2], top_results[1]['position'], top_results[1]['signal'],
                                  'Top2', colors[1], top_results[1]['formula'][:35] + '...', threshold)
    else:
        axes[2].text(0.5, 0.5, 'No Top2 Formula', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Top2 Position')
    
    # Top 3 Position + Signal
    if len(top_results) > 2:
        plot_position_with_signal(axes[3], top_results[2]['position'], top_results[2]['signal'],
                                  'Top3', colors[2], top_results[2]['formula'][:35] + '...', threshold)
    else:
        axes[3].text(0.5, 0.5, 'No Top3 Formula', ha='center', va='center', transform=axes[3].transAxes)
        axes[3].set_title('Top3 Position')
    
    # 第5圖：Ensemble Position + Signal
    if result_ensemble:
        plot_position_with_signal(axes[4], result_ensemble['position'], result_ensemble['signal'],
                                  'Ensemble', '#34495E', f'Mean of Top {len(miner.top_k_formulas)} formulas', threshold)
    else:
        axes[4].text(0.5, 0.5, 'No Ensemble', ha='center', va='center', transform=axes[4].transAxes)
        axes[4].set_title('Ensemble Position')

    # 第6圖：回撤對比
    ax6 = axes[5]
    for i, r in enumerate(top_results[:3]):  # 最多顯示 Top 3 的回撤
        dd = 1 - r['equity'] / np.maximum.accumulate(r['equity'])
        ax6.fill_between(test_dates, -dd * 100, alpha=0.3, color=colors[i % len(colors)], 
                         label=f'{r["label"]} MaxDD: {r["metrics"]["max_dd"]:.1%}')
    if result_ensemble:
        dd_e = 1 - result_ensemble['equity'] / np.maximum.accumulate(result_ensemble['equity'])
        ax6.fill_between(test_dates, -dd_e * 100, alpha=0.3, color='#34495E',
                         label=f'Ensemble MaxDD: {result_ensemble["metrics"]["max_dd"]:.1%}')
    ax6.set_title('Drawdown Comparison (%)')
    ax6.set_ylabel('Drawdown %')
    ax6.set_xlabel('Date')
    ax6.legend(loc='lower left', fontsize=8)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_folder, 'strategy_performance_v2.png')
    plt.savefig(output_file, dpi=150)
    print(f"📈 Chart saved to '{output_file}'")
    plt.close()

    # ==================== 保存策略 ====================
    strategy_file = os.path.join(output_folder, 'best_strategy_v2.json')
    with open(strategy_file, 'w') as f:
        json.dump({
            'version': '2.0',
            'symbol': engine.symbol,
            'signal_threshold': threshold,
            'hysteresis': hysteresis,
            'formula_tokens': miner.best_formula_tokens,
            'formula_readable': formula_str,
            'train_score': float(miner.best_score),
            'train_period': f"{engine.start_date} ~ {engine.end_date}",
            'test_period': f"{test_dates.iloc[0].date()} ~ {test_dates.iloc[-1].date()}",
            'benchmark_return': float(bench_total_ret),
            'norm_params': engine.norm_params,
            'features': FEATURES,
            'test_metrics': {
                'single': {k: float(v) for k, v in m_s.items()},
                'ensemble': {k: float(v) for k, v in result_ensemble['metrics'].items()} if result_ensemble else None,
            },
            'top_k_formulas': [
                {
                    'formula': entry['formula'],
                    'tokens': entry['tokens'],
                    'score': float(entry['score']),
                }
                for entry in miner.top_k_formulas
            ],
            'walk_forward_windows': engine.wf_windows,
        }, f, indent=2, default=str)
    print(f"💾 Strategy saved to '{strategy_file}'")
    
    # ==================== 生成 Report.txt ====================
    report_file = os.path.join(output_folder, 'report.txt')
    test_start_str = str(test_dates.iloc[0].date())
    test_end_str = str(test_dates.iloc[-1].date())
    
    report_lines = [
        "=" * 80,
        f"🔬 FINAL REALITY CHECK V2 - {engine.symbol} (Out-of-Sample)",
        "=" * 80,
        f"📜 Best Formula: {formula_str}",
        f"📊 Signal Threshold: ±{threshold}",
        "-" * 80,
        f"📅 Train Period : {engine.start_date} ~ {engine.end_date}",
        f"📅 Test Period  : {test_start_str} ~ {test_end_str}",
        "-" * 80,
        f"📊 Buy & Hold   : {bench_total_ret:.2%}",
        f"📊 Position Stats: Long {(pos==1).sum()} | Short {(pos==-1).sum()} | Hold {(pos==0).sum()} ({hold_ratio:.1f}%)",
        "-" * 80,
    ]
    
    # 添加指標表格
    report_lines.append(f"{'指標':<20} {'Single Best':<18} {'Ensemble':<18}")
    report_lines.append("-" * 80)
    
    metrics_rows = [
        ('📈 Total Return', 'total_ret', True),
        ('📈 Ann. Return', 'ann_ret', True),
        ('📊 Volatility', 'vol', True),
        ('⭐ Sharpe', 'sharpe', False),
        ('📉 Max Drawdown', 'max_dd', True),
        ('🎯 Calmar', 'calmar', False),
        ('✅ Win Rate', 'win_rate', True),
        ('💰 Profit Factor', 'profit_factor', False),
    ]
    
    for name, key, is_pct in metrics_rows:
        val_s = m_s.get(key, 0)
        val_e = result_ensemble['metrics'].get(key, 0) if result_ensemble else 0
        if is_pct:
            report_lines.append(f"{name:<20} {val_s:>16.2%} {val_e:>16.2%}")
        else:
            report_lines.append(f"{name:<20} {val_s:>16.2f} {val_e:>16.2f}")
    
    report_lines.append("-" * 80)
    
    # 添加 Top K 公式
    report_lines.append("")
    report_lines.append(f"📋 Top {len(miner.top_k_formulas)} Formulas (Ensemble Candidates):")
    report_lines.append("-" * 80)
    for i, entry in enumerate(miner.top_k_formulas):
        tm = entry.get('train_metrics', {})
        vm = entry.get('val_metrics', {})
        train_ret = tm.get('total_ret', 0) if tm else 0
        val_ret = vm.get('total_ret', 0) if vm else 0
        report_lines.append(f"  [{i+1}] Score={entry['score']:.2f} | Train={train_ret:.1%} | Val={val_ret:.1%} | {entry['formula']}")
    
    report_lines.append("-" * 80)
    report_lines.append("")
    report_lines.append(f"Output Folder: {output_folder}")
    report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"📝 Report saved to '{report_file}'")
    
    # ==================== 保存 Top N 公式為獨立策略 ====================
    top_save = min(TOP_SAVE_COUNT, len(miner.top_k_formulas))
    if top_save > 0:
        print(f"\n📁 Saving Top {top_save} formulas as separate strategies...")
        
        for rank, entry in enumerate(miner.top_k_formulas[:top_save], 1):
            # 創建獨立文件夾
            formula_short = entry['formula'][:30].replace('/', '_').replace('(', '').replace(')', '').replace(',', '_')
            sub_folder = os.path.join(output_folder, f"top{rank}_{formula_short}")
            os.makedirs(sub_folder, exist_ok=True)
            
            # 計算該公式的測試集指標
            factor_i = miner.solve_one(entry['tokens'])
            if factor_i is not None:
                result_i = run_backtest(factor_i, f"Top{rank}")
                m_i = result_i['metrics']
                
                # 保存策略 JSON
                strategy_i_file = os.path.join(sub_folder, 'best_strategy_v2.json')
                with open(strategy_i_file, 'w') as f:
                    json.dump({
                        'version': '2.0',
                        'symbol': engine.symbol,
                        'rank': rank,
                        'signal_threshold': threshold,
                        'hysteresis': hysteresis,
                        'formula_tokens': entry['tokens'],
                        'formula_readable': entry['formula'],
                        'train_score': float(entry['score']),
                        'train_period': f"{engine.start_date} ~ {engine.end_date}",
                        'test_period': f"{test_start_str} ~ {test_end_str}",
                        'benchmark_return': float(bench_total_ret),
                        'norm_params': engine.norm_params,
                        'features': FEATURES,
                        'test_metrics': {k: float(v) for k, v in m_i.items()},
                        'train_metrics': entry.get('train_metrics', {}),
                        'val_metrics': entry.get('val_metrics', {}),
                    }, f, indent=2, default=str)
                
                # 保存該公式的 report.txt
                report_i_file = os.path.join(sub_folder, 'report.txt')
                report_i_lines = [
                    "=" * 70,
                    f"🏆 Top {rank} Formula - {engine.symbol}",
                    "=" * 70,
                    f"📜 Formula: {entry['formula']}",
                    f"📊 Score: {entry['score']:.4f}",
                    f"📊 Threshold: ±{threshold}",
                    "-" * 70,
                    f"📅 Test Period: {test_start_str} ~ {test_end_str}",
                    "-" * 70,
                    f"📈 Total Return : {m_i['total_ret']:.2%}",
                    f"📈 Ann. Return  : {m_i['ann_ret']:.2%}",
                    f"⭐ Sharpe       : {m_i['sharpe']:.2f}",
                    f"📉 Max Drawdown : {m_i['max_dd']:.2%}",
                    f"✅ Win Rate     : {m_i['win_rate']:.2%}",
                    "-" * 70,
                    f"📊 Buy & Hold   : {bench_total_ret:.2%}",
                    f"📊 Alpha        : {m_i['total_ret'] - bench_total_ret:.2%}",
                    "-" * 70,
                ]
                with open(report_i_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(report_i_lines))
                
                print(f"   ✅ Top {rank}: {entry['formula'][:40]}... → {sub_folder}")
    
    # ==================== 保存 Ensemble 策略 ====================
    if result_ensemble and len(miner.top_k_formulas) > 1:
        print(f"\n📁 Saving Ensemble strategy...")
        
        # ===== 計算 Ensemble 的 Score（使用訓練過程的評估方法）=====
        print(f"   🔄 Calculating Ensemble Score using training evaluation...")
        ensemble_score_result = miner.calculate_score(factor_ensemble)
        
        if ensemble_score_result:
            ensemble_score = ensemble_score_result['score']
            ensemble_min_score = ensemble_score_result['min_score']
            ensemble_mean_score = ensemble_score_result['mean_score']
            window_scores = ensemble_score_result['window_scores']
            
            print(f"\n   " + "=" * 60)
            print(f"   📊 ENSEMBLE SCORE RESULTS")
            print(f"   " + "=" * 60)
            print(f"   Final Score  : {ensemble_score:.4f}")
            print(f"   Min Score    : {ensemble_min_score:.4f}")
            print(f"   Mean Score   : {ensemble_mean_score:.4f}")
            if len(window_scores) > 1:
                print(f"   Window Scores: {', '.join([f'{s:.2f}' for s in window_scores])}")
            
            # 與 Top1 對比
            top1_score = miner.top_k_formulas[0]['score'] if miner.top_k_formulas else 0
            print(f"   " + "-" * 60)
            print(f"   Top1 Score   : {top1_score:.4f}")
            print(f"   Ensemble vs Top1: {ensemble_score - top1_score:+.4f} ({'+' if ensemble_score > top1_score else ''}{(ensemble_score/top1_score - 1)*100:.1f}%)")
            print(f"   " + "=" * 60)
        else:
            ensemble_score = 0.0
            ensemble_min_score = 0.0
            ensemble_mean_score = 0.0
            window_scores = []
            print(f"   ⚠️ Could not calculate Ensemble Score")
        
        # 創建 Ensemble 文件夾
        ensemble_folder = os.path.join(output_folder, f"ensemble_top{len(miner.top_k_formulas)}")
        os.makedirs(ensemble_folder, exist_ok=True)
        
        # 構建 Ensemble 的 formula_readable
        ensemble_formulas = []
        ensemble_tokens_list = []
        for i, entry in enumerate(miner.top_k_formulas):
            ensemble_formulas.append(f"[{i+1}] {entry['formula']}")
            ensemble_tokens_list.append(entry['tokens'])
        
        ensemble_formula_readable = "ENSEMBLE(\n  " + ",\n  ".join(ensemble_formulas) + "\n)"
        
        m_e = result_ensemble['metrics']
        
        # 保存 Ensemble 策略 JSON
        ensemble_strategy_file = os.path.join(ensemble_folder, 'best_strategy_v2.json')
        with open(ensemble_strategy_file, 'w') as f:
            json.dump({
                'version': '2.0',
                'type': 'ensemble',
                'symbol': engine.symbol,
                'signal_threshold': threshold,
                'hysteresis': hysteresis,
                'ensemble_count': len(miner.top_k_formulas),
                'train_score': float(ensemble_score),
                'train_score_min': float(ensemble_min_score),
                'train_score_mean': float(ensemble_mean_score),
                'train_score_windows': [float(s) for s in window_scores],
                'formula_readable': ensemble_formula_readable,
                'formula_tokens_list': ensemble_tokens_list,
                'component_formulas': [
                    {
                        'rank': i + 1,
                        'formula': entry['formula'],
                        'tokens': entry['tokens'],
                        'score': float(entry['score']),
                        'weight': 1.0 / len(miner.top_k_formulas),  # 等權重
                    }
                    for i, entry in enumerate(miner.top_k_formulas)
                ],
                'train_period': f"{engine.start_date} ~ {engine.end_date}",
                'test_period': f"{test_start_str} ~ {test_end_str}",
                'benchmark_return': float(bench_total_ret),
                'norm_params': engine.norm_params,
                'features': FEATURES,
                'test_metrics': {k: float(v) for k, v in m_e.items()},
            }, f, indent=2, default=str)
        
        # 保存 Ensemble 的 report.txt
        ensemble_report_file = os.path.join(ensemble_folder, 'report.txt')
        ensemble_report_lines = [
            "=" * 80,
            f"🎯 ENSEMBLE Strategy - {engine.symbol}",
            "=" * 80,
            f"📊 Ensemble of {len(miner.top_k_formulas)} formulas (equal weight)",
            f"📊 Threshold: ±{threshold}",
            "",
            "=" * 80,
            "📊 ENSEMBLE SCORE (Training Evaluation)",
            "=" * 80,
            f"   Final Score  : {ensemble_score:.4f}",
            f"   Min Score    : {ensemble_min_score:.4f}",
            f"   Mean Score   : {ensemble_mean_score:.4f}",
        ]
        if len(window_scores) > 1:
            ensemble_report_lines.append(f"   Window Scores: {', '.join([f'{s:.2f}' for s in window_scores])}")
        
        # 與各單公式 Score 對比
        ensemble_report_lines.extend([
            "-" * 80,
            "📊 Score Comparison (Ensemble vs Single):",
        ])
        for i, entry in enumerate(miner.top_k_formulas[:5]):
            diff = ensemble_score - entry['score']
            ensemble_report_lines.append(f"   vs Top{i+1}: {entry['score']:.2f} → Diff: {diff:+.2f}")
        
        ensemble_report_lines.extend([
            "",
            "-" * 80,
            "📜 Component Formulas:",
        ])
        for i, entry in enumerate(miner.top_k_formulas):
            weight_pct = 100.0 / len(miner.top_k_formulas)
            ensemble_report_lines.append(f"   [{i+1}] ({weight_pct:.1f}%) Score={entry['score']:.2f} | {entry['formula']}")
        
        ensemble_report_lines.extend([
            "",
            "=" * 80,
            "📊 TEST SET RESULTS",
            "=" * 80,
            f"📅 Test Period: {test_start_str} ~ {test_end_str}",
            "-" * 80,
            f"📈 Total Return : {m_e['total_ret']:.2%}",
            f"📈 Ann. Return  : {m_e['ann_ret']:.2%}",
            f"📊 Volatility   : {m_e['vol']:.2%}",
            f"⭐ Sharpe       : {m_e['sharpe']:.2f}",
            f"📉 Max Drawdown : {m_e['max_dd']:.2%}",
            f"🎯 Calmar       : {m_e['calmar']:.2f}",
            f"✅ Win Rate     : {m_e['win_rate']:.2%}",
            "-" * 80,
            f"📊 Buy & Hold   : {bench_total_ret:.2%}",
            f"📊 Alpha        : {m_e['total_ret'] - bench_total_ret:.2%}",
            "-" * 80,
        ])
        
        with open(ensemble_report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(ensemble_report_lines))
        
        print(f"   ✅ Ensemble: {len(miner.top_k_formulas)} formulas → {ensemble_folder}")


# ==================== 主程序 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='AlphaGPT V2 - Anti-Overfitting Edition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
核心改進：
  1. Walk-Forward 滾動驗證
  2. 多目標獎勵（Sortino + Sharpe + Return - MaxDD - Complexity）
  3. 更多因子（ATR, RSI, CLV, RS, MOM）
  4. Ensemble 策略
  5. 複雜度懲罰

使用範例：
  python times_us_v2.py --symbol SPY --walk_forward
  python times_us_v2.py --symbol NVDA --iterations 500
        """
    )
    parser.add_argument('--symbol', type=str, default=DEFAULT_SYMBOL, 
                        help='Stock/ETF symbol (default: SPY)')
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
    parser.add_argument('--threshold', type=float, default=DEFAULT_SIGNAL_THRESHOLD,
                        help='Signal threshold (default: 0.1)')
    parser.add_argument('--hysteresis', type=float, default=DEFAULT_HYSTERESIS,
                        help='Hysteresis for position changes to avoid frequent trading (default: 0.05)')
    parser.add_argument('--walk_forward', action='store_true',
                        help='Enable Walk-Forward validation')
    parser.add_argument('--no_walk_forward', action='store_true',
                        help='Disable Walk-Forward validation')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: random)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: auto-generated in output/)')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable AMP (Automatic Mixed Precision) training')
    parser.add_argument('--no-compile', action='store_true',
                        help='Disable torch.compile optimization')
    args = parser.parse_args()

    # 設置隨機種子（多進程並行時每個進程使用不同種子）
    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        print(f"🎲 Random seed set to: {args.seed}")

    # 更新配置
    TRAIN_ITERATIONS = args.iterations
    BATCH_SIZE = args.batch_size
    
    # 確定是否使用 Walk-Forward
    use_wf = args.walk_forward or (not args.no_walk_forward)

    # 創建輸出文件夾（支持外部指定）
    if args.output_dir:
        # 使用外部指定的輸出目錄
        output_folder = args.output_dir
        os.makedirs(output_folder, exist_ok=True)
    else:
        # 自動生成輸出目錄
        suffix = "_WF" if use_wf else ""
        output_folder = create_output_folder(args.symbol, suffix)
    
    print("="*80)
    print("🚀 AlphaGPT V2 - Anti-Overfitting Edition")
    print("="*80)
    print(f"Symbol        : {args.symbol}")
    print(f"Train Period  : {args.start} ~ {args.end}")
    print(f"Test Period   : {args.end} ~ {args.test_end}")
    print(f"Walk-Forward  : {'Enabled' if use_wf else 'Disabled'}")
    print(f"Threshold     : ±{args.threshold} (Hysteresis: {args.hysteresis})")
    print(f"Iterations    : {args.iterations}")
    print(f"Device        : {DEVICE}")
    print(f"Output        : {output_folder}")
    print("="*80)

    # 加載數據
    engine = USDataEngineV2(args.symbol, args.start, args.end, args.test_end)
    engine.load()

    # 訓練（支援 AMP 和 torch.compile 加速）
    use_amp = not getattr(args, 'no_amp', False)
    use_compile = not getattr(args, 'no_compile', False)
    
    miner = DeepQuantMinerV2(
        engine, 
        signal_threshold=args.threshold, 
        hysteresis=args.hysteresis, 
        use_walk_forward=use_wf,
        use_amp=use_amp,
        use_compile=use_compile
    )
    miner.train()

    # 回測報告
    final_reality_check_v2(miner, engine, output_folder)

    # 清理緩存
    if os.path.exists(engine.cache_path):
        os.remove(engine.cache_path)
        print(f"🗑️  Deleted cache file: {engine.cache_path}")
    
    print(f"\n✅ All outputs saved to: {output_folder}")
