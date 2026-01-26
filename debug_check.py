"""
Debug Script: Check Data and Formula Execution
"""

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import json

print("=" * 60)
print("DEBUG: Data and Formula Check")
print("=" * 60)

# ==================== 1. Check yfinance data mapping ====================
print("\n[1] YFINANCE DATA MAPPING CHECK")
print("-" * 60)

symbol = "SPY"
ticker = yf.Ticker(symbol)
df = ticker.history(period="5d", auto_adjust=True)

print(f"Symbol: {symbol}")
print(f"Raw DataFrame columns: {list(df.columns)}")
print(f"Raw DataFrame index name: {df.index.name}")
print(f"\nRaw DataFrame head:")
print(df.head())

df_reset = df.reset_index()
print(f"\nAfter reset_index() columns: {list(df_reset.columns)}")
print(df_reset.head())

# ==================== 2. Check feature calculation ====================
print("\n[2] FEATURE CALCULATION CHECK")
print("-" * 60)

# Get more data to calculate features
df = ticker.history(period="120d", auto_adjust=True)
df = df.reset_index()

close = df['Close'].values.astype(np.float32)
high = df['High'].values.astype(np.float32)
low = df['Low'].values.astype(np.float32)
vol = df['Volume'].values.astype(np.float32)

# Calculate RSI
delta = pd.Series(close).diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / (loss + 1e-6)
rsi = (100 - 100 / (1 + rs)).fillna(50).values.astype(np.float32)
rsi_norm = (rsi - 50) / 50  # Normalize to [-1, 1]

# Calculate RET
ret = np.zeros_like(close)
ret[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-6)

def robust_norm(x):
    x = x.astype(np.float32)
    median = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - median)) + 1e-6
    res = (x - median) / mad
    return np.clip(res, -5, 5).astype(np.float32)

ret_normed = robust_norm(ret)

print(f"RET (last 5 days raw): {ret[-5:]}")
print(f"RET (last 5 days normed): {ret_normed[-5:]}")
print(f"RSI (last 5 days raw): {rsi[-5:]}")
print(f"RSI (last 5 days normed): {rsi_norm[-5:]}")

# ==================== 3. Check formula execution ====================
print("\n[3] FORMULA EXECUTION CHECK")
print("-" * 60)

# Load strategy
try:
    with open("NVDA_best_strategy.json", "r") as f:
        strategy = json.load(f)
    
    formula_tokens = strategy['formula_tokens']
    formula_readable = strategy['formula_readable']
    
    print(f"Formula tokens: {formula_tokens}")
    print(f"Formula readable: {formula_readable}")
    
    # Define FEATURES and OPS
    FEATURES = ['RET', 'RET5', 'RET20', 'VOL_CHG', 'V_RET', 'TREND', 'ATR', 'RSI']
    
    # Print token mapping
    print(f"\nToken mapping:")
    for i, token in enumerate(formula_tokens):
        if token < len(FEATURES):
            print(f"  Token {token} -> Feature: {FEATURES[token]}")
        else:
            op_idx = token - len(FEATURES)
            OPS = ['ADD', 'SUB', 'MUL', 'DIV', 'NEG', 'ABS', 'SIGN', 'DELTA5', 'DELTA10', 
                   'MA10', 'MA20', 'STD20', 'MAX20', 'MIN20']
            if op_idx < len(OPS):
                print(f"  Token {token} -> Operator: {OPS[op_idx]}")
            else:
                print(f"  Token {token} -> Unknown")
    
except FileNotFoundError:
    print("ERROR: NVDA_best_strategy.json not found")

# ==================== 4. Manual formula execution ====================
print("\n[4] MANUAL FORMULA EXECUTION")
print("-" * 60)

# Formula: NEG(SIGN(SUB(RET, DIV(SIGN(RSI), RSI))))
# Manual calculation

RET = torch.from_numpy(ret_normed)
RSI = torch.from_numpy(rsi_norm.astype(np.float32))

print(f"RET tensor shape: {RET.shape}")
print(f"RSI tensor shape: {RSI.shape}")
print(f"RET last value: {RET[-1].item():.6f}")
print(f"RSI last value: {RSI[-1].item():.6f}")

# Step by step
step1_sign_rsi = torch.sign(RSI)
print(f"\nStep 1 - SIGN(RSI) last value: {step1_sign_rsi[-1].item():.6f}")

step2_div = step1_sign_rsi / (RSI + 1e-6)
print(f"Step 2 - DIV(SIGN(RSI), RSI) last value: {step2_div[-1].item():.6f}")

step3_sub = RET - step2_div
print(f"Step 3 - SUB(RET, ...) last value: {step3_sub[-1].item():.6f}")

step4_sign = torch.sign(step3_sub)
print(f"Step 4 - SIGN(...) last value: {step4_sign[-1].item():.6f}")

step5_neg = -step4_sign
print(f"Step 5 - NEG(...) last value: {step5_neg[-1].item():.6f}")

# Final signal
final_signal = torch.tanh(step5_neg[-1])
print(f"\nFinal signal (tanh): {final_signal.item():.6f}")
print(f"Signal strength: {final_signal.item():.2f}")

# ==================== 5. Check multiple symbols ====================
print("\n[5] MULTI-SYMBOL CHECK")
print("-" * 60)

for sym in ['SPY', 'QQQ', 'AAPL', 'MSFT']:
    ticker = yf.Ticker(sym)
    df = ticker.history(period="120d", auto_adjust=True)
    close = df['Close'].values.astype(np.float32)
    
    # RET
    ret = np.zeros_like(close)
    ret[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-6)
    ret_normed = robust_norm(ret)
    
    # RSI
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-6)
    rsi = (100 - 100 / (1 + rs)).fillna(50).values.astype(np.float32)
    rsi_norm = (rsi - 50) / 50
    
    # Calculate formula
    RET_t = torch.from_numpy(ret_normed)
    RSI_t = torch.from_numpy(rsi_norm.astype(np.float32))
    
    result = -torch.sign(RET_t - torch.sign(RSI_t) / (RSI_t + 1e-6))
    signal = torch.tanh(result[-1]).item()
    
    print(f"{sym}: RET={ret_normed[-1]:.4f}, RSI_norm={rsi_norm[-1]:.4f}, Signal={signal:.4f}")

print("\n" + "=" * 60)
print("Debug complete!")
print("=" * 60)
