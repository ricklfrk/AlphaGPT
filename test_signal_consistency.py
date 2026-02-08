"""
測試 signal_generator.py 和 backtest_strategy_v2.py 的信號是否一致
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# 導入兩個模塊
from signal_generator import SignalGenerator
from backtest_strategy_v2 import RealisticBacktester

def test_signal_consistency(strategy_file: str, ticker: str = "NVDA", lookback_days: int = 120):
    """比較兩個模塊生成的信號"""
    
    print("=" * 70)
    print("🔍 Testing Signal Consistency")
    print("=" * 70)
    print(f"Strategy: {strategy_file}")
    print(f"Ticker: {ticker}")
    print(f"Lookback: {lookback_days} days")
    print("-" * 70)
    
    # 1. 使用 SignalGenerator 獲取最新信號
    print("\n📊 Signal Generator Results:")
    sg = SignalGenerator(strategy_file=strategy_file)
    sg_result = sg.get_signal(ticker, lookback_days=lookback_days)
    
    if sg_result['error']:
        print(f"❌ Error: {sg_result['error']}")
        return
    
    print(f"   Date: {sg_result['date']}")
    print(f"   Price: {sg_result['price']:.2f}")
    print(f"   Signal: {sg_result['signal_strength']:.4f}")
    print(f"   Direction: {sg_result['direction']}")
    
    # 2. 使用 RealisticBacktester 獲取最新信號
    print("\n📊 Realistic Backtester Results:")
    bt = RealisticBacktester(strategy_file)
    
    # 下載數據
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days + 30)
    
    yf_ticker = yf.Ticker(ticker)
    df = yf_ticker.history(start=start_date, end=end_date, auto_adjust=True)
    df = df.reset_index()
    
    # 獲取最後一天的信號
    last_idx = len(df) - 1
    bt_signal, bt_position = bt.get_signal_at_day(df, last_idx)
    
    print(f"   Date: {df['Date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"   Price: {df['Close'].iloc[-1]:.2f}")
    print(f"   Signal: {bt_signal:.4f}")
    print(f"   Position: {bt_position}")
    
    # 3. 比較結果
    print("\n" + "=" * 70)
    print("📊 Comparison:")
    print("-" * 70)
    
    signal_diff = abs(sg_result['signal_strength'] - bt_signal)
    print(f"   Signal Generator: {sg_result['signal_strength']:.6f}")
    print(f"   Backtest V2:      {bt_signal:.6f}")
    print(f"   Difference:       {signal_diff:.6f}")
    
    if signal_diff < 0.0001:
        print("\n✅ Signals are CONSISTENT!")
    else:
        print(f"\n⚠️ Signals DIFFER by {signal_diff:.6f}")
        
        # 詳細診斷
        print("\n🔍 Detailed Diagnosis:")
        
        # 比較特徵
        print("\n   Comparing features at last day...")
        
        # SignalGenerator 的特徵
        sg_features = sg._compute_features(df)
        print(f"\n   Signal Generator Features (last day):")
        for i, name in enumerate(['RET', 'RET5', 'VOL_CHG', 'V_RET', 'TREND']):
            print(f"      {name}: {sg_features[i, -1].item():.6f}")
        
        # Backtester 的特徵
        bt_features = bt._compute_features_at_day(df, last_idx)
        print(f"\n   Backtest V2 Features (last day):")
        for i, name in enumerate(['RET', 'RET5', 'VOL_CHG', 'V_RET', 'TREND']):
            print(f"      {name}: {bt_features[i, -1].item():.6f}")
        
        # 特徵差異
        print(f"\n   Feature Differences:")
        for i, name in enumerate(['RET', 'RET5', 'VOL_CHG', 'V_RET', 'TREND']):
            diff = abs(sg_features[i, -1].item() - bt_features[i, -1].item())
            status = "✓" if diff < 0.0001 else "✗"
            print(f"      {name}: {diff:.6f} {status}")
    
    print("=" * 70)
    
    return {
        'sg_signal': sg_result['signal_strength'],
        'bt_signal': bt_signal,
        'difference': signal_diff
    }


def test_multiple_days(strategy_file: str, ticker: str = "NVDA", num_days: int = 10):
    """比較多天的信號"""
    
    print("\n" + "=" * 70)
    print(f"🔍 Testing {num_days} Days Signal Consistency")
    print("=" * 70)
    
    # 加載策略
    sg = SignalGenerator(strategy_file=strategy_file)
    bt = RealisticBacktester(strategy_file)
    
    # 下載數據
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    yf_ticker = yf.Ticker(ticker)
    df = yf_ticker.history(start=start_date, end=end_date, auto_adjust=True)
    df = df.reset_index()
    
    print(f"\nData range: {df['Date'].iloc[0].strftime('%Y-%m-%d')} ~ {df['Date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"Total days: {len(df)}")
    
    # 比較最後 N 天
    print(f"\n{'Date':<12} {'Price':>10} {'SG Signal':>12} {'BT Signal':>12} {'Diff':>10} {'Status':>8}")
    print("-" * 70)
    
    differences = []
    for i in range(max(60, len(df) - num_days), len(df)):
        date_str = df['Date'].iloc[i].strftime('%Y-%m-%d')
        price = df['Close'].iloc[i]
        
        # SignalGenerator: 使用截至第 i 天的數據
        df_slice = df.iloc[:i+1].copy()
        sg_features = sg._compute_features(df_slice)
        sg_factor = sg._execute_formula(sg_features)
        sg_signal = np.tanh(sg_factor[-1].item()) if sg_factor is not None else 0
        
        # Backtester: 使用 get_signal_at_day
        bt_signal, _ = bt.get_signal_at_day(df, i)
        
        diff = abs(sg_signal - bt_signal)
        differences.append(diff)
        status = "✓" if diff < 0.0001 else "✗"
        
        print(f"{date_str:<12} {price:>10.2f} {sg_signal:>12.6f} {bt_signal:>12.6f} {diff:>10.6f} {status:>8}")
    
    print("-" * 70)
    print(f"Average Difference: {np.mean(differences):.6f}")
    print(f"Max Difference: {np.max(differences):.6f}")
    
    if np.max(differences) < 0.0001:
        print("\n✅ All signals are CONSISTENT!")
    else:
        print(f"\n⚠️ Found {sum(d > 0.0001 for d in differences)} inconsistent days")


if __name__ == "__main__":
    import sys
    
    # 使用最新策略
    strategy_file = "output/NVDA_T1OT2O_20260127_114530/best_strategy.json"
    
    if len(sys.argv) > 1:
        strategy_file = sys.argv[1]
    
    # 測試單天
    test_signal_consistency(strategy_file)
    
    # 測試多天
    test_multiple_days(strategy_file, num_days=10)
