# 🧠 AlphaGPT

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License">
</p>

**AlphaGPT** 是一套基於深度學習的**自動化因子挖掘系統**。它不直接預測價格，而是使用 Transformer 模型自動生成可解釋的量化因子公式，並通過強化學習優化公式生成器。

> 🎯 核心理念：**生成公式 → 執行公式 → 回測評分 → 優化生成器**

---

## ✨ 特點

- 🤖 **自動因子挖掘** - Transformer 自動生成因子公式，無需手動設計
- 📊 **可解釋性強** - 輸出的公式是人類可讀的數學表達式
- 🔄 **強化學習優化** - 使用 Policy Gradient 根據回測收益優化模型
- 🏗️ **模塊化設計** - 策略研究與交易執行清晰分離
- 🌐 **多市場支持** - 支持加密貨幣、A股、美股

---

## 📦 安裝

### 環境要求

- Python 3.10+
- CUDA 11.8 / 12.1 / 12.4（推薦，用於 GPU 加速）
- NVIDIA GPU（推薦 RTX 3060 或更高）

### 安裝步驟

```bash
# 克隆項目
git clone https://github.com/imbue-bit/AlphaGPT.git
cd AlphaGPT

# Step 1: 安裝 PyTorch（選擇對應的 CUDA 版本）

# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4（最新）
pip install torch --index-url https://download.pytorch.org/whl/cu124

# CPU 版本（無 GPU）
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Step 2: 安裝其他依賴
pip install -r requirements.txt

# Step 3: 安裝可選依賴（A股回測）
pip install -r requirements-optional.txt
```

### 驗證 CUDA 安裝

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

### 依賴說明

| 依賴包 | 用途 |
|--------|------|
| `torch` (CUDA) | 深度學習框架（GPU 加速） |
| `pandas`, `numpy` | 數據處理 |
| `yfinance` | 美股數據 |
| `matplotlib` | 圖表繪製 |
| `sqlalchemy`, `asyncpg` | 數據庫連接 |
| `aiohttp` | 異步 HTTP |
| `solana`, `solders` | Solana 區塊鏈交互 |
| `streamlit`, `plotly` | 可視化看板 |
| `tushare` | A股數據（可選）|

---

## 🚀 快速開始

### 互動式命令行介面（推薦新手）

```bash
# 啟動互動式 CLI
python alphagpt_cli.py
```

CLI 提供以下功能：
1. 🚀 **訓練新策略** - 設置參數並訓練
2. 📊 **回測策略 (V2)** - 使用正確的持倉邏輯回測
3. 📡 **生成交易信號** - 掃描多個股票
4. 📋 **查看已有策略** - 列出所有策略
5. 🧹 **清理輸出文件** - 管理文件

### 美股因子挖掘

```bash
# 訓練 SPY 策略（使用默認參數，隔夜策略 T1OT2O）
python times_us.py --symbol SPY

# 使用日內策略 (T+1 Open → T+1 Close)
python times_us.py --symbol SPY --mode T1OTC --iterations 600

# 使用隔夜策略 (T+1 Open → T+2 Open)
python times_us.py --symbol NVDA --mode T1OT2O --iterations 600 --batch_size 2048

# 自定義時間範圍（訓練：2018~2023，測試：2023~2025）
python times_us.py --symbol AAPL --mode T1OTC --start 2018-01-01 --end 2023-01-01 --test_end 2025-01-01

# 查看生成的信號
python signal_generator.py --strategy output/SPY_T1OTC_xxx/best_strategy.json
```

**預測目標模式：**
| 模式 | 說明 | 適用場景 |
|------|------|----------|
| `T1OTC` | T+1 Open-to-Close (日內策略) | 日內波動大、避免隔夜風險 |
| `T1OT2O` | T+1 Open-to-T+2 Open (隔夜策略) | 捕捉隔夜跳空、趨勢延續 |

**輸出示例：**
```
✅ NVDA Data Ready!
   Total: 1523 days | Train: 1258 | Test: 265
   Train Period: 2020-01-02 ~ 2024-12-31
   Test Period : 2025-01-02 ~ 2026-01-23
   Target Mode: T1OTC (T+1 Open → T+1 Close)
```

### A股因子挖掘

```bash
# 需要先在 tushare.pro 註冊獲取 Token
# 修改 times.py 中的 TS_TOKEN

python times.py
```

### 加密貨幣實盤（進階）

```bash
# 1. 配置環境變量
cp .env.example .env
# 編輯 .env 填入 API Keys

# 2. 啟動數據管線
python -m data_pipeline.run_pipeline

# 3. 訓練模型
python -m model_core.engine

# 4. 啟動策略執行
python -m strategy_manager.runner
```

---

## 🏗️ 項目結構

```
AlphaGPT/
├── model_core/          # 🧠 策略挖掘核心
│   ├── alphagpt.py      # Transformer 模型定義
│   ├── factors.py       # 因子計算模塊
│   ├── ops.py           # 算子定義
│   ├── vm.py            # StackVM 虛擬機（執行公式）
│   ├── backtest.py      # 回測引擎
│   ├── engine.py        # 訓練引擎
│   └── config.py        # 模型配置
│
├── data_pipeline/       # 📊 數據管線
│   ├── fetcher.py       # 數據抓取（Birdeye/DexScreener）
│   ├── processor.py     # 數據處理
│   ├── db_manager.py    # 數據庫管理
│   └── config.py        # 數據配置
│
├── strategy_manager/    # 📈 策略執行
│   ├── runner.py        # 策略主循環
│   ├── portfolio.py     # 持倉管理
│   ├── risk.py          # 風控引擎
│   └── config.py        # 策略配置
│
├── execution/           # 💱 交易執行
│   ├── trader.py        # 交易執行器
│   ├── jupiter.py       # Jupiter 聚合器
│   ├── rpc_handler.py   # Solana RPC
│   └── config.py        # 執行配置
│
├── dashboard/           # 📺 可視化看板
│   ├── app.py           # Streamlit 主應用
│   ├── visualizer.py    # 圖表組件
│   └── data_service.py  # 數據服務
│
├── lord/                # 🔬 研究實驗
│   └── experiment.py    # LoRD 正則化實驗
│
├── times.py              # 🇨🇳 A股回測腳本
├── times_us.py           # 🇺🇸 美股回測腳本
├── signal_generator.py   # 📡 實時信號生成器
├── backtest_strategy_v2.py # 📊 V2 策略回測腳本 (推薦)
├── backtest_strategy.py  # 📊 舊版策略回測腳本
├── alphagpt_cli.py       # 🖥️ 互動式命令行介面
│
├── requirements.txt      # 核心依賴
└── requirements-optional.txt  # 可選依賴
```

---

## 🧠 核心概念

### 1. 因子（Features）

因子是從原始行情數據計算出的特徵，用於描述市場狀態：

| 因子 | 說明 | 計算方式 |
|------|------|----------|
| `RET` | 日收益率 | `log(close / close_prev)` |
| `RET5` | 5日收益率 | `close / close_5d_ago - 1` |
| `VOL_CHG` | 量變化 | `volume / MA(volume, 20) - 1` |
| `TREND` | 趨勢 | `close / MA(close, 60) - 1` |
| `ATR` | 波動率 | 14日平均真實波幅 |
| `RSI` | 相對強弱 | 14日RSI標準化 |

### 2. 算子（Operators）

算子是對因子進行變換的函數：

| 算子 | 類型 | 說明 |
|------|------|------|
| `ADD`, `SUB`, `MUL`, `DIV` | 二元 | 基本運算 |
| `NEG`, `ABS`, `SIGN` | 一元 | 數學函數 |
| `MA10`, `MA20` | 一元 | 移動平均 |
| `DELTA5`, `DELTA10` | 一元 | N日變化 |
| `STD20` | 一元 | Z-Score 標準化 |
| `MAX20`, `MIN20` | 一元 | 滾動極值 |

### 3. 公式（Formula）

公式是因子和算子的組合，例如：

```
ADD(MA20(RET), MUL(VOL_CHG, TREND))
```

這個公式表示：20日動量 + (量變化 × 趨勢)

### 4. StackVM 虛擬機

StackVM 使用堆棧計算執行公式，類似逆波蘭表達式：

```python
# 公式 Token: [ADD, MA20, RET, MUL, VOL_CHG, TREND]
# 執行過程:
# 1. TREND → stack: [TREND]
# 2. VOL_CHG → stack: [TREND, VOL_CHG]
# 3. MUL → stack: [TREND * VOL_CHG]
# 4. RET → stack: [TREND * VOL_CHG, RET]
# 5. MA20 → stack: [TREND * VOL_CHG, MA20(RET)]
# 6. ADD → stack: [MA20(RET) + TREND * VOL_CHG]
```

---

## 📈 使用教程

### 美股回測

```bash
# 基本用法（使用默認隔夜策略 T1OT2O）
python times_us.py --symbol SPY

# 使用日內策略
python times_us.py --symbol SPY --mode T1OTC

# 自定義參數
python times_us.py \
    --symbol QQQ \
    --mode T1OTC \
    --start 2018-01-01 \
    --end 2023-01-01 \
    --test_end 2025-01-01 \
    --iterations 600

# 使用 V2 回測腳本測試已有策略（推薦）
python backtest_strategy_v2.py \
    --strategy output/NVDA_T1OT2O_xxx/best_strategy.json \
    --tickers NVDA,SPY,QQQ \
    --period 1y \
    --capital 100000 \
    --export --plot

# 指定時間範圍回測
python backtest_strategy_v2.py \
    --strategy output/xxx/best_strategy.json \
    --tickers NVDA \
    --start 2024-01-01 \
    --end 2026-01-01
```

**訓練命令行參數：**

| 參數 | 說明 | 默認值 |
|------|------|--------|
| `--symbol` | 股票代碼 | SPY |
| `--mode` | 預測目標模式 (T1OTC/T1OT2O) | T1OT2O |
| `--threshold` | 信號閾值 (0~1) | 0.1 |
| `--start` | 數據開始日期 | 2015-01-01 |
| `--end` | 訓練/測試分界點 | 2024-01-01 |
| `--test_end` | 數據結束日期 | 2025-01-01 |
| `--iterations` | 訓練迭代次數 | 400 |
| `--batch_size` | 批次大小 | 1024 |

**信號閾值說明：**
```
閾值用於將連續信號 [-1, +1] 離散化為 {-1, 0, +1}：
  - signal >  threshold → 倉位 = +1 (做多)
  - signal < -threshold → 倉位 = -1 (做空)
  - |signal| < threshold → 倉位 = 0 (觀望)

較高閾值：減少交易次數，只在信號強烈時交易
較低閾值：增加交易次數，對弱信號也做出反應
```

**支持的標的：**

| 類型 | 代碼示例 |
|------|----------|
| 大盤 ETF | SPY, VOO, IVV |
| 科技 ETF | QQQ, XLK |
| 小盤 ETF | IWM, VB |
| 行業 ETF | XLF, XLE, XLV |
| 個股 | AAPL, MSFT, NVDA, TSLA |

### 信號生成

```bash
# 查看單個策略的信號（使用策略默認模式）
python signal_generator.py --strategy output/SPY_T1OTC_xxx/best_strategy.json

# 指定模式生成信號
python signal_generator.py --strategy output/SPY_best_strategy.json --mode T1OTC

# 掃描多個股票
python signal_generator.py --symbols SPY,QQQ,AAPL,MSFT,NVDA --mode T1OT2O

# 監控模式（每分鐘更新）
python signal_generator.py --strategy output/SPY_T1OTC_xxx/best_strategy.json --monitor
```

### 策略回測 (V2 推薦)

使用 `backtest_strategy_v2.py` 對已訓練的策略進行回測：

```bash
# 基本用法
python backtest_strategy_v2.py --strategy output/NVDA_T1OT2O_xxx/best_strategy.json --tickers NVDA

# 指定時間範圍
python backtest_strategy_v2.py \
    --strategy output/NVDA_T1OT2O_xxx/best_strategy.json \
    --tickers SPY,QQQ,AAPL \
    --start 2024-01-01 \
    --end 2026-01-01 \
    --capital 100000 \
    --export --plot

# 調整信號閾值
python backtest_strategy_v2.py \
    --strategy output/xxx/best_strategy.json \
    --tickers NVDA \
    --threshold 0.2 \
    --period 1y
```

**V2 回測器特點：**
- ✅ **正確的持倉邏輯**：信號不變則持倉不變，不強制每天換倉
- ✅ **與訓練一致**：回測結果與訓練報告完全一致
- ✅ **詳細交易記錄**：輸出 today_position, tomorrow_position, action 等字段

**回測參數：**

| 參數 | 說明 | 默認值 |
|------|------|--------|
| `--strategy` | 策略 JSON 文件 | 必填 |
| `--tickers` | 測試標的（逗號分隔） | SPY |
| `--period` | 回測週期 (如 1y, 6mo) | 1y |
| `--start` | 開始日期 (YYYY-MM-DD) | - |
| `--end` | 結束日期 (YYYY-MM-DD) | - |
| `--capital` | 初始資金 | 100000 |
| `--threshold` | 覆蓋信號閾值 | 策略原始值 |
| `--export` | 導出交易記錄 CSV | - |
| `--plot` | 生成回測圖表 | - |

**V2 回測邏輯說明：**
```
時間線：
  T日收盤 → 計算信號 → 決定 T+1 的倉位
  T+1開盤 → 根據信號調整倉位
  持有直到信號改變（不強制每天換倉）

收益計算：
  daily_ret[T] = today_position[T] × (Close[T] - Close[T-1]) / Close[T-1]
```

### 在代碼中使用

```python
from signal_generator import SignalGenerator

# 加載策略（使用策略默認模式）
gen = SignalGenerator(strategy_file='output/SPY_T1OTC_xxx/best_strategy.json')

# 或者覆蓋模式
gen = SignalGenerator(
    strategy_file='output/SPY_best_strategy.json',
    override_mode='T1OTC'
)

# 獲取信號
signal = gen.get_signal('AAPL')

print(f"方向: {signal['direction']}")           # BUY / SELL / HOLD
print(f"強度: {signal['signal_strength']}")     # -1 到 1
print(f"價格: ${signal['price']:.2f}")
print(f"模式: {signal['target_mode']}")         # T1OTC / T1OT2O
print(f"指令: {signal['trade_instruction']}")   # 具體交易指令

# 批量掃描
results = gen.scan_multiple(['AAPL', 'MSFT', 'NVDA'])
```

---

## ⚙️ 配置說明

### 美股配置 (`times_us.py`)

```python
DEFAULT_SYMBOL = 'SPY'           # 默認標的
START_DATE = '2015-01-01'        # 數據開始日期
END_DATE = '2024-01-01'          # 訓練/測試分界點 ← 關鍵！
TEST_END_DATE = '2025-01-01'     # 數據結束日期
BATCH_SIZE = 1024                # 批次大小
TRAIN_ITERATIONS = 400           # 訓練輪數
MAX_SEQ_LEN = 8                  # 公式最大長度
COST_RATE = 0.0005               # 交易成本

# 預測目標模式
TARGET_MODE_T1OTC = 'T1OTC'      # T+1 Open → T+1 Close (日內策略)
TARGET_MODE_T1OT2O = 'T1OT2O'    # T+1 Open → T+2 Open (隔夜策略)
```

**⚠️ 日期劃分邏輯：**
- **訓練集**：`START_DATE` ~ `END_DATE`（模型學習期間）
- **測試集**：`END_DATE` ~ `TEST_END_DATE`（樣本外驗證）

這種設計確保測試集是真正的「樣本外」數據，避免數據洩漏。

**📊 預測目標模式說明：**
| 模式 | 目標收益計算 | 說明 |
|------|-------------|------|
| T1OTC | `(Close[T+1] - Open[T+1]) / Open[T+1]` | 日內策略，避免隔夜風險 |
| T1OT2O | `(Open[T+2] - Open[T+1]) / Open[T+1]` | 隔夜策略，捕捉隔夜跳空 |

### A股配置 (`times.py`)

```python
TS_TOKEN = 'your_tushare_token'  # Tushare API Token
INDEX_CODE = '511260.SH'         # 標的代碼
COST_RATE = 0.0005               # 交易成本（萬五）
```

### 加密貨幣配置 (`.env`)

```bash
# Birdeye API
BIRDEYE_API_KEY=your_api_key

# Solana RPC
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
WALLET_PRIVATE_KEY=your_private_key

# Database
DATABASE_URL=postgresql://user:pass@localhost/alphagpt
```

---

## 📊 回測指標

| 指標 | 說明 |
|------|------|
| **Ann. Return** | 年化收益率 |
| **Sharpe Ratio** | 夏普比率 (風險調整收益) |
| **Sortino Ratio** | 索提諾比率 (下行風險調整) |
| **Max Drawdown** | 最大回撤 |
| **Calmar Ratio** | 卡瑪比率 (收益/最大回撤) |
| **Win Rate** | 勝率 |
| **Profit Factor** | 盈虧比 |

---

## 🔬 技術細節

### AlphaGPT 模型架構

```
Input (Token Sequence)
    ↓
Token Embedding + Position Embedding
    ↓
┌─────────────────────────────┐
│   Looped Transformer        │
│   ├─ QK-Norm Attention      │
│   ├─ RMSNorm                │
│   └─ SwiGLU FFN             │
│   (Loop × 3 per layer)      │
└─────────────────────────────┘
    ↓
RMSNorm
    ↓
┌─────────────────────────────┐
│   MTP Head (Multi-Task)     │
│   ├─ Actor (Token Logits)   │
│   └─ Critic (Value)         │
└─────────────────────────────┘
```

### 訓練流程

1. **採樣階段**: Transformer 逐步生成公式 Token
2. **執行階段**: StackVM 執行公式，生成因子信號
3. **回測階段**: 計算策略收益作為獎勵
4. **更新階段**: Policy Gradient 更新模型參數
5. **正則化**: LoRD (Low-Rank Decay) 防止過擬合

### LoRD 正則化

LoRD 使用 Newton-Schulz 迭代計算最小奇異向量，對注意力權重進行低秩衰減：

```python
# Newton-Schulz iteration
for _ in range(num_iterations):
    A = Y.T @ Y
    Y = 0.5 * Y @ (3.0 * I - A)

# Low-rank decay
W -= decay_rate * Y
```

---

## 📁 輸出文件

### 目錄結構

```
output/
├── NVDA_T1OT2O_20260127_150000/       # 訓練輸出（帶時間戳）
│   ├── best_strategy.json              # 策略文件
│   ├── strategy_performance.png        # 訓練結果圖表
│   └── report.txt                      # 詳細報告
│
├── SPY_T1OT2O_20260127_160000/        # 另一次訓練
│   └── ...
│
├── NVDA_backtest_v2.png               # V2 回測圖表
├── NVDA_trade_log_v2.csv              # V2 交易記錄
│
└── data_cache_NVDA.parquet            # 數據緩存
```

| 文件類型 | 說明 |
|----------|------|
| `best_strategy.json` | 最優策略公式（含模式、閾值、歸一化參數） |
| `strategy_performance.png` | 回測曲線圖（訓練時生成） |
| `report.txt` | 詳細回測報告 |
| `*_backtest_v2.png` | V2 回測曲線圖 |
| `*_trade_log_v2.csv` | V2 交易記錄（詳細字段見下表） |
| `data_cache_*.parquet` | 數據緩存 |

### V2 Trade Log 字段說明

| 字段 | 說明 | 時間點 |
|------|------|--------|
| `date` | 日期 | T 日 |
| `price` | 收盤價 | T 日 |
| `today_position` | 今天實際持有的倉位 | T 日（由 T-1 信號決定）|
| `action` | 今天開盤的操作 | T 日開盤 |
| `signal` | 今天收盤後的信號 | T 日收盤（決定 T+1 倉位）|
| `tomorrow_position` | 明天的倉位 | T+1 日（由今天信號決定）|
| `daily_ret_pct` | 今天收益率 (%) | T 日 |
| `equity` | 累計淨值 | T 日 |

**Action 類型：**
| Action | 說明 |
|--------|------|
| `HOLD` | 持倉不變 |
| `BUY` | 從空倉買入做多 |
| `SHORT` | 從空倉賣空做空 |
| `CLOSE` | 平倉回到空倉 |
| `COVER & BUY` | 回補空頭並做多 |
| `SELL & SHORT` | 賣出多頭並做空 |

### 策略文件格式

```json
{
  "symbol": "SPY",
  "target_mode": "T1OTC",
  "target_mode_description": "T+1 Open-to-Close (日內策略...)",
  "formula_tokens": [8, 10, 0],
  "formula_readable": "ADD(MA10(RET),RET)",
  "train_sortino": 1.85,
  "train_period": "2015-01-01 ~ 2024-01-01",
  "test_period": "2024-01-02 ~ 2025-01-23",
  "discrete": {
    "sharpe": 1.42,
    "ann_return": 0.156,
    "max_drawdown": 0.12,
    "win_rate": 0.52,
    "profit_factor": 1.35
  },
  "continuous": {
    "sharpe": 1.28,
    "ann_return": 0.132,
    "max_drawdown": 0.14,
    "win_rate": 0.51,
    "profit_factor": 1.22
  }
}
```

---

## ✅ 訓練與回測一致性

**V2 系統確保訓練報告與回測結果完全一致：**

| 組件 | 信號計算 | 收益計算 |
|------|---------|---------|
| `times_us.py` (訓練) | 每天計算信號，shift 1天決定倉位 | Close-to-Close 日收益 |
| `backtest_strategy_v2.py` | 每天計算信號，shift 1天決定倉位 | Close-to-Close 日收益 |
| `signal_generator.py` | 相同的特徵計算和公式執行 | - |

**驗證一致性：**
```bash
# 運行一致性測試
python test_signal_consistency.py output/xxx/best_strategy.json
```

**關鍵改進（相比舊版）：**
1. ✅ 修復了 Look-Ahead Bias：歸一化參數從訓練集保存並在回測中復用
2. ✅ 正確的持倉邏輯：信號不變則持倉不變，不強制每天換倉
3. ✅ 統一的收益計算：使用 Close-to-Close 日收益

---

## ⚠️ 風險提示

> **重要：本項目僅供研究學習，不構成投資建議！**

1. **回測 ≠ 實盤** - 過去表現不代表未來收益
2. **過擬合風險** - 生成的公式可能對歷史數據過擬合
3. **市場變化** - 市場結構變化可能導致策略失效
4. **交易成本** - 實際滑點和手續費可能高於模擬
5. **流動性風險** - 小市值標的可能難以按預期價格成交

### 建議

- 📝 先用模擬盤驗證至少 1-3 個月
- 💰 實盤初期用 5-10% 資金測試
- 🛡️ 設置止損，單筆虧損不超過總資金 2%
- 🔄 定期重新訓練策略（如每季度）
- 📊 分散投資，不要只交易單一標的

---

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

---

## 📄 許可證

本項目採用 [Apache-2.0 License](LICENSE) 開源。

---

## 🔗 相關鏈接

- [Tushare 數據平台](https://tushare.pro) - A股數據
- [yfinance 文檔](https://github.com/ranaroussi/yfinance) - 美股數據
- [Birdeye API](https://docs.birdeye.so) - 加密貨幣數據
- [Jupiter Aggregator](https://jup.ag) - Solana DEX 聚合器

---

## 📈 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=imbue-bit/AlphaGPT&type=date&legend=top-left)](https://www.star-history.com/#imbue-bit/AlphaGPT&type=date&legend=top-left)

---

<p align="center">
  <b>🚀 Happy Trading! 🚀</b>
</p>
