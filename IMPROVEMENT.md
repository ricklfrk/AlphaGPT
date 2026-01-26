# 🚀 AlphaGPT 改進指南

本文檔詳細說明 AlphaGPT 項目的潛在改進方向，包括原因分析、實現方法和優先級建議。

---

## 📋 目錄

1. [改進總覽](#改進總覽)
2. [訓練算法改進](#1-訓練算法改進)
3. [模型架構改進](#2-模型架構改進)
4. [因子與算子擴展](#3-因子與算子擴展)
5. [回測系統改進](#4-回測系統改進)
6. [風控系統改進](#5-風控系統改進)
7. [工程優化](#6-工程優化)
8. [實驗與驗證](#7-實驗與驗證)
9. [改進優先級](#改進優先級)

---

## 改進總覽

```
                            AlphaGPT 改進方向
                                   │
         ┌─────────────┬───────────┼───────────┬─────────────┐
         │             │           │           │             │
    🎯 訓練算法    🧠 模型架構   📊 因子算子   📈 回測系統   🔧 工程優化
         │             │           │           │             │
    ┌────┴────┐   ┌────┴────┐  ┌───┴───┐  ┌────┴────┐   ┌────┴────┐
    PPO    課程  注意力  位置  新因子  新算子  成本   驗證   並行   緩存
    Entropy 學習  機制   編碼  多週期  金融類  滑點   WF    GPU    數據
```

---

## 1. 訓練算法改進

### 1.1 升級到 PPO (Proximal Policy Optimization)

#### 🔍 為什麼？

| 問題 | REINFORCE (當前) | PPO (改進) |
|------|-----------------|------------|
| 訓練穩定性 | ❌ 方差大，不穩定 | ✅ Clip 機制限制更新幅度 |
| 樣本效率 | ❌ 每個樣本只用一次 | ✅ 可多次重用樣本 |
| 超參敏感 | ❌ 學習率敏感 | ✅ 更魯棒 |

#### 📝 實現方法

```python
# 文件：model_core/ppo_trainer.py

import torch
import torch.nn as nn
from torch.distributions import Categorical

class PPOTrainer:
    """
    PPO (Proximal Policy Optimization) 訓練器
    
    相比 REINFORCE 的優勢：
    1. Clipped objective 防止策略更新過大
    2. 可以多次重用同一批數據
    3. 訓練更穩定
    """
    
    def __init__(
        self,
        model,
        lr=3e-4,
        clip_epsilon=0.2,      # PPO clip 範圍
        value_coef=0.5,        # Value loss 權重
        entropy_coef=0.01,     # Entropy bonus 權重
        max_grad_norm=0.5,     # 梯度裁剪
        ppo_epochs=4,          # 每批數據訓練幾輪
    ):
        self.model = model
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    def compute_gae(self, rewards, values, gamma=0.99, lam=0.95):
        """
        計算 Generalized Advantage Estimation (GAE)
        
        GAE 平衡了偏差和方差：
        - λ=0: 高偏差，低方差（TD error）
        - λ=1: 低偏差，高方差（Monte Carlo）
        - λ=0.95: 常用的平衡點
        """
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)
        
        return torch.tensor(advantages)
    
    def update(self, states, actions, old_log_probs, rewards, values):
        """
        PPO 更新步驟
        """
        # 計算優勢
        advantages = self.compute_gae(rewards, values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 計算回報（用於 Value function 訓練）
        returns = advantages + values
        
        # 多輪更新（PPO 的關鍵優勢）
        for _ in range(self.ppo_epochs):
            # 前向傳播
            new_logits, new_values = self.model(states)
            dist = Categorical(logits=new_logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # 計算 ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio, 
                1 - self.clip_epsilon, 
                1 + self.clip_epsilon
            ) * advantages
            
            # 損失函數
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(new_values.squeeze(), returns)
            entropy_loss = -entropy
            
            total_loss = (
                policy_loss 
                + self.value_coef * value_loss 
                + self.entropy_coef * entropy_loss
            )
            
            # 更新
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
        }
```

#### 🔧 集成到現有代碼

```python
# 修改 model_core/engine.py

from .ppo_trainer import PPOTrainer

class AlphaEngine:
    def __init__(self, use_ppo=True):
        self.model = AlphaGPT().to(DEVICE)
        
        if use_ppo:
            self.trainer = PPOTrainer(self.model)
        else:
            self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
```

---

### 1.2 添加 Entropy Bonus

#### 🔍 為什麼？

- **問題**：模型可能過早收斂到局部最優，只生成少數幾種公式
- **解決**：Entropy bonus 鼓勵探索，保持策略的多樣性

#### 📝 實現方法（最簡單的改進）

```python
# 在 times_us.py 或 model_core/engine.py 中

# 原始代碼
loss = -(torch.stack(log_probs, 1).sum(1) * adv).mean()

# 改進代碼
policy_loss = -(torch.stack(log_probs, 1).sum(1) * adv).mean()

# 計算 entropy
entropy = 0
for step_logits in all_logits:  # 需要保存每步的 logits
    dist = Categorical(logits=step_logits)
    entropy += dist.entropy().mean()
entropy /= len(all_logits)

# 最終損失（entropy bonus 為負，因為我們要最大化 entropy）
ENTROPY_COEF = 0.01  # 可調參數
loss = policy_loss - ENTROPY_COEF * entropy
```

---

### 1.3 課程學習 (Curriculum Learning)

#### 🔍 為什麼？

- **問題**：一開始就搜索長公式，搜索空間太大
- **解決**：從短公式開始，逐漸增加複雜度

#### 📝 實現方法

```python
# 文件：model_core/curriculum.py

class CurriculumScheduler:
    """
    課程學習調度器
    
    思路：
    - 階段1：只生成長度 4 的公式（簡單）
    - 階段2：長度增加到 6
    - 階段3：完整長度 8
    
    好處：
    1. 先學會生成有效的短公式
    2. 逐漸擴展到更複雜的公式
    3. 加速收斂
    """
    
    def __init__(
        self,
        initial_len=4,
        final_len=8,
        warmup_steps=100,
        total_steps=400,
    ):
        self.initial_len = initial_len
        self.final_len = final_len
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
    
    def get_max_len(self, step):
        """根據訓練步數返回當前最大公式長度"""
        if step < self.warmup_steps:
            return self.initial_len
        
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)
        
        current_len = self.initial_len + (self.final_len - self.initial_len) * progress
        return int(current_len)
    
    def get_difficulty(self, step):
        """返回當前難度係數 [0, 1]"""
        return min(1.0, step / self.total_steps)


# 使用方式
scheduler = CurriculumScheduler()

for step in range(TRAIN_ITERATIONS):
    max_len = scheduler.get_max_len(step)
    
    # 生成公式時使用動態長度
    for t in range(max_len):  # 而不是固定的 MAX_SEQ_LEN
        ...
```

---

## 2. 模型架構改進

### 2.1 Rotary Position Embedding (RoPE)

#### 🔍 為什麼？

| 方面 | 絕對位置編碼（當前） | RoPE（改進） |
|------|-------------------|-------------|
| 外推能力 | ❌ 超出訓練長度效果差 | ✅ 更好的長度泛化 |
| 相對位置 | ❌ 不直接建模 | ✅ 自然編碼相對位置 |
| 計算效率 | ✅ 簡單 | ✅ 同樣高效 |

#### 📝 實現方法

```python
# 文件：model_core/rope.py

import torch
import torch.nn as nn
import math

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    
    論文：RoFormer: Enhanced Transformer with Rotary Position Embedding
    
    原理：通過旋轉矩陣編碼位置信息，使得 q·k 自然包含相對位置
    """
    
    def __init__(self, dim, max_seq_len=512, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # 計算頻率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # 預計算 cos 和 sin
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def forward(self, x, seq_len):
        """
        Args:
            x: [batch, seq_len, dim]
            seq_len: 序列長度
        """
        return (
            self.cos_cached[:seq_len].unsqueeze(0),
            self.sin_cached[:seq_len].unsqueeze(0)
        )


def rotate_half(x):
    """將向量分成兩半並旋轉"""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """應用 RoPE 到 query 和 key"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RoPEAttention(nn.Module):
    """帶有 RoPE 的注意力層"""
    
    def __init__(self, d_model, nhead, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len)
    
    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        # 投影
        q = self.q_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        
        # 應用 RoPE
        cos, sin = self.rotary(x, T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # 注意力計算
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        
        return self.o_proj(out)
```

---

### 2.2 QK-Norm（Query-Key Normalization）

#### 🔍 為什麼？

- **問題**：深層 Transformer 中，Q 和 K 的範數可能失控
- **解決**：對 Q 和 K 進行 L2 歸一化，穩定注意力分數

#### 📝 實現方法

```python
# 已在 model_core/alphagpt.py 中實現
# 這裡展示如何使用

class QKNorm(nn.Module):
    """Query-Key Normalization"""
    
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        # 可學習的縮放因子
        self.scale = nn.Parameter(torch.ones(1) * (d_model ** -0.5))
    
    def forward(self, q, k):
        # L2 歸一化
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)
        
        # 應用縮放
        return q_norm * self.scale, k_norm * self.scale
```

---

## 3. 因子與算子擴展

### 3.1 新增因子

#### 🔍 為什麼？

- **當前因子**：主要是價格和成交量相關
- **缺失**：波動率結構、流動性、市場微觀結構

#### 📝 實現方法

```python
# 文件：新增到 times_us.py 的因子計算部分

def compute_advanced_features(df):
    """計算高級因子"""
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    volume = df['Volume'].values
    
    features = {}
    
    # ========== 波動率因子 ==========
    
    # 1. 已實現波動率 (Realized Volatility)
    ret = np.log(close[1:] / close[:-1])
    rv = pd.Series(ret**2).rolling(20).sum().fillna(0).values
    features['RV'] = np.sqrt(rv * 252)  # 年化
    
    # 2. Parkinson 波動率（用 High-Low）
    hl_ratio = np.log(high / low)
    parkinson = pd.Series(hl_ratio**2).rolling(20).mean().fillna(0).values
    features['PARKINSON'] = np.sqrt(parkinson / (4 * np.log(2)) * 252)
    
    # 3. 波動率偏度 (Vol of Vol)
    ret_series = pd.Series(ret)
    rolling_vol = ret_series.rolling(5).std()
    vol_of_vol = rolling_vol.rolling(20).std().fillna(0).values
    features['VOV'] = vol_of_vol
    
    # ========== 流動性因子 ==========
    
    # 4. Amihud 非流動性指標
    abs_ret = np.abs(ret)
    dollar_volume = close[1:] * volume[1:]
    amihud = abs_ret / (dollar_volume + 1e-9)
    features['AMIHUD'] = pd.Series(amihud).rolling(20).mean().fillna(0).values
    
    # 5. 換手率變化
    avg_volume = pd.Series(volume).rolling(60).mean()
    turnover_ratio = volume / (avg_volume + 1e-9)
    features['TURNOVER'] = turnover_ratio.fillna(1).values
    
    # ========== 價格結構因子 ==========
    
    # 6. 跳空缺口
    gap = (df['Open'].values[1:] - close[:-1]) / close[:-1]
    features['GAP'] = np.concatenate([[0], gap])
    
    # 7. 價格偏度 (Price Skewness)
    skew = pd.Series(ret).rolling(20).skew().fillna(0).values
    features['SKEW'] = skew
    
    # 8. 價格峰度 (Price Kurtosis)
    kurt = pd.Series(ret).rolling(20).kurt().fillna(0).values
    features['KURT'] = kurt
    
    # ========== 動量因子 ==========
    
    # 9. 信息離散度 (Information Discreteness)
    # 衡量收益是連續的還是跳躍的
    sign_ret = np.sign(ret)
    info_disc = pd.Series(sign_ret).rolling(20).sum().fillna(0).values / 20
    features['INFO_DISC'] = info_disc
    
    # 10. 動量反轉
    mom_5 = pd.Series(close).pct_change(5).fillna(0).values
    mom_20 = pd.Series(close).pct_change(20).fillna(0).values
    features['MOM_REV'] = mom_5 - mom_20  # 短期動量 - 長期動量
    
    return features


# 更新 FEATURES 列表
FEATURES = [
    # 原有因子
    'RET', 'RET5', 'RET20', 'VOL_CHG', 'V_RET', 'TREND', 'ATR', 'RSI',
    # 新增因子
    'RV', 'PARKINSON', 'VOV',           # 波動率
    'AMIHUD', 'TURNOVER',               # 流動性
    'GAP', 'SKEW', 'KURT',              # 價格結構
    'INFO_DISC', 'MOM_REV',             # 動量
]
```

---

### 3.2 新增算子

#### 🔍 為什麼？

- **當前算子**：主要是基礎運算和簡單時序
- **缺失**：橫截面排名、條件邏輯、更多時序變換

#### 📝 實現方法

```python
# 文件：新增到 times_us.py 的算子部分

# ========== 新增時序算子 ==========

@torch.jit.script
def _ts_rank(x: torch.Tensor, d: int) -> torch.Tensor:
    """
    時序排名：當前值在過去 d 天中的百分位排名
    
    用途：識別極端值，減少異常值影響
    """
    if d <= 1: return torch.ones_like(x) * 0.5
    B, T = x.shape
    pad = torch.zeros((B, d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    
    # 計算排名
    sorted_idx = windows.argsort(dim=-1)
    ranks = sorted_idx.argsort(dim=-1)
    current_rank = ranks[..., -1]  # 當前值的排名
    
    return current_rank.float() / (d - 1)  # 歸一化到 [0, 1]


@torch.jit.script
def _ts_corr(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    """
    時序相關性：過去 d 天 x 和 y 的相關係數
    
    用途：檢測因子之間的動態關係
    """
    if d <= 2: return torch.zeros_like(x)
    B, T = x.shape
    
    pad = torch.zeros((B, d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    y_pad = torch.cat([pad, y], dim=1)
    
    x_win = x_pad.unfold(1, d, 1)
    y_win = y_pad.unfold(1, d, 1)
    
    x_mean = x_win.mean(dim=-1, keepdim=True)
    y_mean = y_win.mean(dim=-1, keepdim=True)
    
    x_centered = x_win - x_mean
    y_centered = y_win - y_mean
    
    cov = (x_centered * y_centered).mean(dim=-1)
    x_std = x_centered.std(dim=-1) + 1e-6
    y_std = y_centered.std(dim=-1) + 1e-6
    
    return cov / (x_std * y_std)


@torch.jit.script
def _ts_decay_exp(x: torch.Tensor, d: int, decay: float = 0.9) -> torch.Tensor:
    """
    指數衰減加權平均
    
    用途：給近期數據更高權重
    """
    if d <= 1: return x
    B, T = x.shape
    
    pad = torch.zeros((B, d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    
    # 指數衰減權重
    weights = torch.tensor([decay ** i for i in range(d-1, -1, -1)], 
                          device=x.device, dtype=x.dtype)
    weights = weights / weights.sum()
    
    return (windows * weights).sum(dim=-1)


@torch.jit.script
def _ts_argmax(x: torch.Tensor, d: int) -> torch.Tensor:
    """
    最大值出現的位置（距今天數）
    
    用途：識別趨勢拐點
    """
    if d <= 1: return torch.zeros_like(x)
    B, T = x.shape
    
    pad = torch.full((B, d - 1), float('-inf'), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    
    argmax = windows.argmax(dim=-1)
    return (d - 1 - argmax).float() / (d - 1)  # 歸一化


# ========== 條件算子 ==========

@torch.jit.script
def _op_if_else(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    條件選擇：if condition > 0 then x else y
    
    用途：實現條件邏輯
    """
    mask = (condition > 0).float()
    return mask * x + (1 - mask) * y


@torch.jit.script
def _op_clip(x: torch.Tensor, low: float = -3.0, high: float = 3.0) -> torch.Tensor:
    """
    裁剪：限制在 [low, high] 範圍內
    
    用途：去除異常值
    """
    return torch.clamp(x, low, high)


# ========== 更新 OPS_CONFIG ==========

OPS_CONFIG = [
    # 基礎運算（原有）
    ('ADD', lambda x, y: x + y, 2),
    ('SUB', lambda x, y: x - y, 2),
    ('MUL', lambda x, y: x * y, 2),
    ('DIV', lambda x, y: x / (y + 1e-6 * torch.sign(y)), 2),
    
    # 數學函數（原有）
    ('NEG', lambda x: -x, 1),
    ('ABS', lambda x: torch.abs(x), 1),
    ('SIGN', lambda x: torch.sign(x), 1),
    
    # 時序算子（原有）
    ('DELTA5', lambda x: _ts_delta(x, 5), 1),
    ('DELTA10', lambda x: _ts_delta(x, 10), 1),
    ('MA10', lambda x: _ts_decay_linear(x, 10), 1),
    ('MA20', lambda x: _ts_decay_linear(x, 20), 1),
    ('STD20', lambda x: _ts_zscore(x, 20), 1),
    ('MAX20', lambda x: _ts_max(x, 20), 1),
    ('MIN20', lambda x: _ts_min(x, 20), 1),
    
    # 新增時序算子
    ('RANK20', lambda x: _ts_rank(x, 20), 1),           # 時序排名
    ('DECAY_EXP', lambda x: _ts_decay_exp(x, 10), 1),   # 指數衰減
    ('ARGMAX20', lambda x: _ts_argmax(x, 20), 1),       # 最大值位置
    
    # 新增條件算子
    ('IFELSE', _op_if_else, 3),                         # 條件選擇
    ('CLIP', lambda x: _op_clip(x), 1),                 # 裁剪
    
    # 新增相關性算子（需要兩個輸入）
    ('CORR20', lambda x, y: _ts_corr(x, y, 20), 2),     # 時序相關
]
```

---

## 4. 回測系統改進

### 4.1 Walk-Forward 驗證

#### 🔍 為什麼？

| 方法 | 單次劃分（當前） | Walk-Forward（改進） |
|------|-----------------|---------------------|
| 過擬合檢測 | ❌ 只看一個測試集 | ✅ 多個測試期 |
| 時間一致性 | ❌ 未驗證 | ✅ 模擬真實交易 |
| 結果可靠性 | ⚠️ 可能是運氣 | ✅ 統計顯著 |

#### 📝 實現方法

```python
# 文件：model_core/walk_forward.py

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class WFResult:
    """Walk-Forward 單期結果"""
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_sharpe: float
    test_sharpe: float
    test_return: float
    test_max_dd: float


class WalkForwardValidator:
    """
    Walk-Forward 驗證器
    
    原理：
    ┌──────────────┬────────┐
    │   訓練期 1    │ 測試 1  │
    └──────────────┴────────┘
         ┌──────────────┬────────┐
         │   訓練期 2    │ 測試 2  │
         └──────────────┴────────┘
              ┌──────────────┬────────┐
              │   訓練期 3    │ 測試 3  │
              └──────────────┴────────┘
    
    好處：
    1. 模擬真實的「用過去預測未來」
    2. 多個測試期，結果更可靠
    3. 檢測策略是否隨時間衰減
    """
    
    def __init__(
        self,
        train_window: int = 252 * 3,   # 3年訓練
        test_window: int = 252,         # 1年測試
        step_size: int = 126,           # 半年滾動一次
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
    
    def generate_splits(self, total_len: int) -> List[Tuple[int, int, int, int]]:
        """
        生成訓練/測試劃分
        
        Returns:
            List of (train_start, train_end, test_start, test_end)
        """
        splits = []
        start = 0
        
        while start + self.train_window + self.test_window <= total_len:
            train_start = start
            train_end = start + self.train_window
            test_start = train_end
            test_end = min(test_start + self.test_window, total_len)
            
            splits.append((train_start, train_end, test_start, test_end))
            start += self.step_size
        
        return splits
    
    def validate(
        self,
        miner,  # DeepQuantMiner instance
        engine,  # DataEngine instance
    ) -> List[WFResult]:
        """
        執行 Walk-Forward 驗證
        """
        total_len = engine.feat_data.shape[1]
        splits = self.generate_splits(total_len)
        results = []
        
        print(f"🔄 Walk-Forward Validation: {len(splits)} folds")
        
        for i, (tr_s, tr_e, te_s, te_e) in enumerate(splits):
            print(f"\n--- Fold {i+1}/{len(splits)} ---")
            print(f"Train: {tr_s} ~ {tr_e} | Test: {te_s} ~ {te_e}")
            
            # 在訓練期上訓練
            miner.train_on_subset(tr_s, tr_e)
            
            # 在測試期上評估
            train_metrics = miner.evaluate_on_subset(tr_s, tr_e)
            test_metrics = miner.evaluate_on_subset(te_s, te_e)
            
            result = WFResult(
                train_start=str(engine.dates.iloc[tr_s].date()),
                train_end=str(engine.dates.iloc[tr_e].date()),
                test_start=str(engine.dates.iloc[te_s].date()),
                test_end=str(engine.dates.iloc[te_e].date()),
                train_sharpe=train_metrics['sharpe'],
                test_sharpe=test_metrics['sharpe'],
                test_return=test_metrics['total_return'],
                test_max_dd=test_metrics['max_drawdown'],
            )
            results.append(result)
            
            print(f"Train Sharpe: {result.train_sharpe:.2f} | "
                  f"Test Sharpe: {result.test_sharpe:.2f}")
        
        return results
    
    def summary(self, results: List[WFResult]) -> dict:
        """生成驗證摘要"""
        test_sharpes = [r.test_sharpe for r in results]
        test_returns = [r.test_return for r in results]
        
        return {
            'num_folds': len(results),
            'avg_test_sharpe': np.mean(test_sharpes),
            'std_test_sharpe': np.std(test_sharpes),
            'min_test_sharpe': np.min(test_sharpes),
            'max_test_sharpe': np.max(test_sharpes),
            'win_rate': np.mean([s > 0 for s in test_sharpes]),
            'avg_test_return': np.mean(test_returns),
            't_statistic': np.mean(test_sharpes) / (np.std(test_sharpes) / np.sqrt(len(test_sharpes)) + 1e-6),
        }
    
    def print_report(self, results: List[WFResult]):
        """打印驗證報告"""
        summary = self.summary(results)
        
        print("\n" + "="*60)
        print("📊 WALK-FORWARD VALIDATION REPORT")
        print("="*60)
        
        print(f"\nFolds: {summary['num_folds']}")
        print(f"Avg Test Sharpe: {summary['avg_test_sharpe']:.2f} ± {summary['std_test_sharpe']:.2f}")
        print(f"Min/Max Sharpe: {summary['min_test_sharpe']:.2f} / {summary['max_test_sharpe']:.2f}")
        print(f"Win Rate: {summary['win_rate']:.1%}")
        print(f"T-Statistic: {summary['t_statistic']:.2f}")
        
        # 顯著性判斷
        if summary['t_statistic'] > 2.0:
            print("✅ Statistically Significant (t > 2.0)")
        else:
            print("⚠️ Not Statistically Significant (t < 2.0)")
        
        print("\n--- Per-Fold Details ---")
        for i, r in enumerate(results):
            print(f"Fold {i+1}: {r.test_start} ~ {r.test_end} | "
                  f"Sharpe: {r.test_sharpe:.2f} | Return: {r.test_return:.1%}")
```

---

### 4.2 真實交易成本模型

#### 🔍 為什麼？

- **當前**：固定成本率 `COST_RATE = 0.0001`
- **問題**：實際成本與交易量、流動性相關

#### 📝 實現方法

```python
# 文件：model_core/cost_model.py

class RealisticCostModel:
    """
    真實交易成本模型
    
    成本組成：
    1. 佣金 (Commission): 固定比例
    2. 滑點 (Slippage): 與交易量/流動性相關
    3. 市場衝擊 (Market Impact): 大額交易會移動價格
    4. 買賣價差 (Spread): bid-ask spread
    """
    
    def __init__(
        self,
        commission_rate: float = 0.0001,  # 萬一佣金
        base_slippage: float = 0.0001,    # 基礎滑點
        impact_coef: float = 0.1,         # 市場衝擊係數
        spread_rate: float = 0.0001,      # 買賣價差
    ):
        self.commission_rate = commission_rate
        self.base_slippage = base_slippage
        self.impact_coef = impact_coef
        self.spread_rate = spread_rate
    
    def compute_cost(
        self,
        trade_value: float,        # 交易金額
        daily_volume: float,       # 當日成交量（金額）
        volatility: float = 0.02,  # 波動率
    ) -> float:
        """
        計算單次交易成本
        
        Args:
            trade_value: 交易金額
            daily_volume: 當日成交量
            volatility: 日波動率
        
        Returns:
            總成本率
        """
        # 1. 佣金（固定）
        commission = self.commission_rate
        
        # 2. 滑點（與波動率相關）
        slippage = self.base_slippage * (volatility / 0.02)
        
        # 3. 市場衝擊（Square-root model）
        # 參考：Almgren & Chriss 模型
        participation_rate = trade_value / (daily_volume + 1e-9)
        market_impact = self.impact_coef * volatility * np.sqrt(participation_rate)
        
        # 4. 買賣價差
        spread = self.spread_rate
        
        total_cost = commission + slippage + market_impact + spread
        
        # 限制最大成本（防止異常）
        return min(total_cost, 0.05)  # 最大 5%
    
    def compute_batch_costs(
        self,
        positions: np.ndarray,        # [T] 持倉序列
        prices: np.ndarray,           # [T] 價格
        volumes: np.ndarray,          # [T] 成交量
        trade_capital: float = 100000,# 交易資金
    ) -> np.ndarray:
        """批量計算成本"""
        T = len(positions)
        costs = np.zeros(T)
        
        # 計算換手
        turnover = np.abs(positions - np.roll(positions, 1))
        turnover[0] = np.abs(positions[0])
        
        # 計算波動率
        returns = np.diff(np.log(prices + 1e-9), prepend=0)
        volatility = pd.Series(returns).rolling(20).std().fillna(0.02).values
        
        for t in range(T):
            if turnover[t] > 0:
                trade_value = trade_capital * turnover[t]
                daily_vol = volumes[t] * prices[t]
                costs[t] = self.compute_cost(trade_value, daily_vol, volatility[t])
        
        return costs * turnover  # 只有換手時才有成本
```

---

## 5. 風控系統改進

### 5.1 動態倉位管理

```python
# 文件：risk/position_sizing.py

class DynamicPositionSizer:
    """
    動態倉位管理
    
    根據波動率和信號強度調整倉位
    """
    
    def __init__(
        self,
        target_vol: float = 0.15,      # 目標年化波動率
        max_position: float = 1.0,      # 最大倉位
        min_position: float = 0.0,      # 最小倉位
        signal_scale: float = 1.0,      # 信號縮放
    ):
        self.target_vol = target_vol
        self.max_position = max_position
        self.min_position = min_position
        self.signal_scale = signal_scale
    
    def compute_position(
        self,
        signal: float,           # 模型信號 [-1, 1]
        current_vol: float,      # 當前波動率（年化）
        regime: str = 'normal',  # 市場狀態
    ) -> float:
        """
        計算目標倉位
        
        公式：position = signal × (target_vol / current_vol) × regime_scale
        """
        # 波動率調整
        vol_scalar = self.target_vol / (current_vol + 0.01)
        vol_scalar = np.clip(vol_scalar, 0.5, 2.0)  # 限制範圍
        
        # 市場狀態調整
        regime_scales = {
            'normal': 1.0,
            'high_vol': 0.5,   # 高波動時減倉
            'crisis': 0.25,    # 危機時大幅減倉
            'trend': 1.2,      # 趨勢市場加倉
        }
        regime_scale = regime_scales.get(regime, 1.0)
        
        # 計算倉位
        position = signal * self.signal_scale * vol_scalar * regime_scale
        
        # 限制範圍
        return np.clip(position, -self.max_position, self.max_position)
    
    def detect_regime(self, returns: np.ndarray, window: int = 60) -> str:
        """檢測市場狀態"""
        if len(returns) < window:
            return 'normal'
        
        recent_vol = np.std(returns[-window:]) * np.sqrt(252)
        recent_return = np.sum(returns[-window:])
        
        if recent_vol > 0.3:
            return 'crisis'
        elif recent_vol > 0.2:
            return 'high_vol'
        elif abs(recent_return) > 0.15 and recent_vol < 0.15:
            return 'trend'
        else:
            return 'normal'
```

### 5.2 止損與風險限制

```python
# 文件：risk/stop_loss.py

class RiskLimiter:
    """
    風險限制器
    
    實現多種止損機制
    """
    
    def __init__(
        self,
        max_drawdown: float = 0.15,       # 最大回撤限制
        daily_loss_limit: float = 0.03,   # 日虧損限制
        trailing_stop: float = 0.10,      # 追蹤止損
    ):
        self.max_drawdown = max_drawdown
        self.daily_loss_limit = daily_loss_limit
        self.trailing_stop = trailing_stop
        
        self.peak_equity = 1.0
        self.daily_pnl = 0.0
    
    def check_limits(self, current_equity: float, daily_return: float) -> dict:
        """
        檢查是否觸發風險限制
        
        Returns:
            {'should_close': bool, 'reason': str}
        """
        # 更新峰值
        self.peak_equity = max(self.peak_equity, current_equity)
        self.daily_pnl = daily_return
        
        # 1. 最大回撤檢查
        drawdown = 1 - current_equity / self.peak_equity
        if drawdown > self.max_drawdown:
            return {
                'should_close': True,
                'reason': f'Max Drawdown ({drawdown:.1%} > {self.max_drawdown:.1%})'
            }
        
        # 2. 日虧損檢查
        if daily_return < -self.daily_loss_limit:
            return {
                'should_close': True,
                'reason': f'Daily Loss Limit ({daily_return:.1%} < -{self.daily_loss_limit:.1%})'
            }
        
        # 3. 追蹤止損
        if drawdown > self.trailing_stop and drawdown > 0.05:
            return {
                'should_close': True,
                'reason': f'Trailing Stop ({drawdown:.1%})'
            }
        
        return {'should_close': False, 'reason': None}
    
    def reset_daily(self):
        """每日重置"""
        self.daily_pnl = 0.0
```

---

## 6. 工程優化

### 6.1 GPU 批量執行

```python
# 文件：model_core/batch_vm.py

class BatchStackVM:
    """
    批量執行虛擬機
    
    優化：一次執行多個公式，利用 GPU 並行
    """
    
    def __init__(self, features_list, ops_config):
        self.feat_offset = len(features_list)
        self.op_map = {i + self.feat_offset: cfg[1] for i, cfg in enumerate(ops_config)}
        self.arity_map = {i + self.feat_offset: cfg[2] for i, cfg in enumerate(ops_config)}
    
    @torch.no_grad()
    def execute_batch(
        self,
        formulas: torch.Tensor,    # [B, max_len] 公式批次
        features: torch.Tensor,    # [F, T] 特徵
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量執行公式
        
        Returns:
            results: [B, T] 每個公式的結果
            valid_mask: [B] 公式是否有效
        """
        B, max_len = formulas.shape
        T = features.shape[1]
        
        results = torch.zeros(B, T, device=features.device)
        valid_mask = torch.ones(B, dtype=torch.bool, device=features.device)
        
        # 並行處理（可進一步優化為純向量化）
        for i in range(B):
            result = self._execute_single(formulas[i], features)
            if result is not None:
                results[i] = result
            else:
                valid_mask[i] = False
        
        return results, valid_mask
    
    def _execute_single(self, formula, features):
        """執行單個公式"""
        stack = []
        try:
            for token in formula:
                token = int(token)
                if token < self.feat_offset:
                    stack.append(features[token])
                elif token in self.op_map:
                    arity = self.arity_map[token]
                    if len(stack) < arity:
                        return None
                    args = [stack.pop() for _ in range(arity)]
                    args.reverse()
                    result = self.op_map[token](*args)
                    result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
                    stack.append(result)
            
            return stack[0] if len(stack) == 1 else None
        except:
            return None
```

### 6.2 數據緩存系統

```python
# 文件：utils/cache.py

import os
import hashlib
import pickle
from functools import wraps

class DataCache:
    """
    數據緩存系統
    
    自動緩存計算結果，加速重複運行
    """
    
    def __init__(self, cache_dir: str = '.cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """生成緩存 key"""
        content = f"{func_name}_{str(args)}_{str(sorted(kwargs.items()))}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, key: str):
        """獲取緩存"""
        path = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def set(self, key: str, value):
        """設置緩存"""
        path = os.path.join(self.cache_dir, f"{key}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(value, f)
    
    def cached(self, func):
        """裝飾器：自動緩存函數結果"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = self._get_key(func.__name__, args, kwargs)
            result = self.get(key)
            if result is not None:
                print(f"📦 Cache hit: {func.__name__}")
                return result
            
            result = func(*args, **kwargs)
            self.set(key, result)
            return result
        return wrapper


# 使用示例
cache = DataCache()

@cache.cached
def compute_features(symbol: str, start: str, end: str):
    """計算特徵（結果會被緩存）"""
    # 耗時計算...
    return features
```

---

## 7. 實驗與驗證

### 7.1 消融實驗

```python
# 文件：experiments/ablation.py

def run_ablation_study(base_config: dict):
    """
    消融實驗：測試每個組件的貢獻
    
    實驗設計：
    1. 完整模型 (baseline)
    2. 移除每個因子
    3. 移除每個算子
    4. 改變模型大小
    """
    results = {}
    
    # 1. Baseline
    results['baseline'] = train_and_evaluate(base_config)
    
    # 2. 因子消融
    for feature in FEATURES:
        config = base_config.copy()
        config['features'] = [f for f in FEATURES if f != feature]
        results[f'without_{feature}'] = train_and_evaluate(config)
    
    # 3. 算子消融
    for op_name, _, _ in OPS_CONFIG:
        config = base_config.copy()
        config['ops'] = [o for o in OPS_CONFIG if o[0] != op_name]
        results[f'without_{op_name}'] = train_and_evaluate(config)
    
    # 4. 模型大小
    for d_model in [32, 64, 128]:
        config = base_config.copy()
        config['d_model'] = d_model
        results[f'd_model_{d_model}'] = train_and_evaluate(config)
    
    return results


def print_ablation_report(results: dict):
    """打印消融報告"""
    baseline = results['baseline']['test_sharpe']
    
    print("="*60)
    print("ABLATION STUDY RESULTS")
    print("="*60)
    print(f"Baseline Sharpe: {baseline:.2f}")
    print("-"*60)
    
    # 按影響程度排序
    impacts = []
    for name, metrics in results.items():
        if name == 'baseline':
            continue
        impact = baseline - metrics['test_sharpe']
        impacts.append((name, metrics['test_sharpe'], impact))
    
    impacts.sort(key=lambda x: x[2], reverse=True)
    
    for name, sharpe, impact in impacts:
        direction = "↓" if impact > 0 else "↑"
        print(f"{name:30s} | Sharpe: {sharpe:.2f} | Impact: {direction} {abs(impact):.2f}")
```

### 7.2 超參數搜索

```python
# 文件：experiments/hyperparam_search.py

import optuna

def objective(trial):
    """Optuna 目標函數"""
    
    # 搜索空間
    config = {
        'd_model': trial.suggest_categorical('d_model', [32, 64, 128]),
        'n_layer': trial.suggest_int('n_layer', 1, 4),
        'n_head': trial.suggest_categorical('n_head', [2, 4, 8]),
        'lr': trial.suggest_loguniform('lr', 1e-4, 1e-2),
        'max_seq_len': trial.suggest_int('max_seq_len', 4, 12),
        'entropy_coef': trial.suggest_loguniform('entropy_coef', 1e-3, 1e-1),
    }
    
    # 訓練並評估
    metrics = train_and_evaluate(config)
    
    return metrics['test_sharpe']


def run_hyperparam_search(n_trials: int = 100):
    """運行超參數搜索"""
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print("Best params:", study.best_params)
    print("Best Sharpe:", study.best_value)
    
    return study
```

---

## 改進優先級

### 🔴 高優先級（建議立即實施）

| 改進 | 難度 | 預期收益 | 實施時間 |
|------|------|----------|----------|
| Walk-Forward 驗證 | 中 | ⭐⭐⭐ 大幅減少過擬合 | 2-3小時 |
| Entropy Bonus | 低 | ⭐⭐ 改善探索 | 10分鐘 |
| 真實交易成本 | 低 | ⭐⭐ 更準確的回測 | 1小時 |

### 🟡 中優先級（建議短期實施）

| 改進 | 難度 | 預期收益 | 實施時間 |
|------|------|----------|----------|
| 新增因子 | 低 | ⭐⭐ 更豐富的信息 | 2小時 |
| 課程學習 | 中 | ⭐⭐ 加速收斂 | 1-2小時 |
| PPO 升級 | 中 | ⭐⭐ 訓練更穩定 | 3-4小時 |
| 動態倉位 | 中 | ⭐⭐ 更好的風險控制 | 2小時 |

### 🟢 低優先級（建議長期實施）

| 改進 | 難度 | 預期收益 | 實施時間 |
|------|------|----------|----------|
| RoPE | 高 | ⭐ 模型較小，收益有限 | 4小時 |
| 批量 VM | 高 | ⭐ 當前速度足夠 | 6小時 |
| 超參數搜索 | 中 | ⭐ 需要大量計算 | 1天 |

---

## 快速開始

### 第一步：添加 Entropy Bonus（5分鐘）

```bash
# 編輯 times_us.py，找到 loss 計算部分
# 按照 1.2 節的方法修改
```

### 第二步：實現 Walk-Forward（2小時）

```bash
# 創建 model_core/walk_forward.py
# 按照 4.1 節的代碼實現
# 在 times_us.py 中集成
```

### 第三步：驗證改進效果

```bash
python times_us.py --symbol SPY --iterations 500
# 比較改進前後的 test_sharpe
```

---

## 📚 參考資料

- [PPO 論文](https://arxiv.org/abs/1707.06347)
- [RoPE 論文](https://arxiv.org/abs/2104.09864)
- [Walk-Forward Analysis](https://www.investopedia.com/terms/w/walk-forward-testing.asp)
- [Almgren-Chriss Market Impact Model](https://www.math.nyu.edu/~almgren/papers/optliq.pdf)

---

<p align="center">
  <b>🚀 持續改進，追求卓越 🚀</b>
</p>
