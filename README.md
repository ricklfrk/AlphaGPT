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
- CUDA 11.8+（可選，用於 GPU 加速）

### 安裝步驟

```bash
# 克隆項目
git clone https://github.com/imbue-bit/AlphaGPT.git
cd AlphaGPT

# 安裝核心依賴
pip install -r requirements.txt

# 安裝可選依賴（A股/美股回測）
pip install -r requirements-optional.txt
```

### 依賴說明

| 依賴包 | 用途 |
|--------|------|
| `torch` | 深度學習框架 |
| `pandas`, `numpy` | 數據處理 |
| `sqlalchemy`, `asyncpg` | 數據庫連接 |
| `aiohttp` | 異步 HTTP |
| `solana`, `solders` | Solana 區塊鏈交互 |
| `streamlit`, `plotly` | 可視化看板 |
| `yfinance` | 美股數據（可選）|
| `tushare` | A股數據（可選）|

---

## 🚀 快速開始

### 美股因子挖掘（推薦新手）

```bash
# 訓練 SPY 策略
python times_us.py --symbol SPY

# 查看生成的信號
python signal_generator.py --strategy SPY_best_strategy.json
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
├── times.py             # 🇨🇳 A股回測腳本
├── times_us.py          # 🇺🇸 美股回測腳本
├── signal_generator.py  # 📡 實時信號生成器
│
├── requirements.txt     # 核心依賴
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
# 基本用法
python times_us.py --symbol SPY

# 自定義參數
python times_us.py \
    --symbol QQQ \
    --start 2018-01-01 \
    --end 2023-01-01 \
    --test_end 2025-01-01 \
    --iterations 600
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
# 查看單個策略的信號
python signal_generator.py --strategy SPY_best_strategy.json

# 掃描多個股票
python signal_generator.py --symbols SPY,QQQ,AAPL,MSFT,NVDA

# 監控模式（每分鐘更新）
python signal_generator.py --strategy SPY_best_strategy.json --monitor
```

### 在代碼中使用

```python
from signal_generator import SignalGenerator

# 加載策略
gen = SignalGenerator(strategy_file='SPY_best_strategy.json')

# 獲取信號
signal = gen.get_signal('AAPL')

print(f"方向: {signal['direction']}")      # BUY / SELL / HOLD
print(f"強度: {signal['signal_strength']}")  # -1 到 1
print(f"價格: ${signal['price']:.2f}")

# 批量掃描
results = gen.scan_multiple(['AAPL', 'MSFT', 'NVDA'])
```

---

## ⚙️ 配置說明

### 美股配置 (`times_us.py`)

```python
DEFAULT_SYMBOL = 'SPY'           # 默認標的
START_DATE = '2015-01-01'        # 訓練開始
END_DATE = '2024-01-01'          # 訓練結束
TEST_END_DATE = '2025-01-01'     # 測試結束
BATCH_SIZE = 1024                # 批次大小
TRAIN_ITERATIONS = 400           # 訓練輪數
MAX_SEQ_LEN = 8                  # 公式最大長度
COST_RATE = 0.0001               # 交易成本
```

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

| 文件 | 說明 |
|------|------|
| `{SYMBOL}_best_strategy.json` | 最優策略公式 |
| `{SYMBOL}_strategy_performance.png` | 回測曲線圖 |
| `data_cache_{SYMBOL}.parquet` | 數據緩存 |
| `training_history.json` | 訓練歷史 |

### 策略文件格式

```json
{
  "symbol": "SPY",
  "formula_tokens": [8, 10, 0],
  "formula_readable": "ADD(MA10(RET),RET)",
  "train_sharpe": 1.85,
  "test_sharpe": 1.42,
  "test_ann_return": 0.156,
  "test_max_drawdown": 0.12
}
```

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
