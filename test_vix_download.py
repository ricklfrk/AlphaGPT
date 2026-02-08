"""
Test VIX and SPY data download
"""
import yfinance as yf
import pandas as pd

def test_download(symbol, name):
    """Test downloading specified symbol"""
    print(f"\n{'='*60}")
    print(f"[TEST] {name} ({symbol})")
    print('='*60)
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start='2020-01-01', end='2026-01-27', auto_adjust=True)
        
        if df.empty:
            print(f"[FAIL] No data returned for {symbol}")
            return False
        
        print(f"[OK] Download successful!")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Date range: {df.index[0].date()} ~ {df.index[-1].date()}")
        print(f"\n[DATA] Latest 5 rows:")
        print(df.tail())
        
        print(f"\n[STATS] Statistics:")
        if 'Close' in df.columns:
            close = df['Close']
            print(f"   Min: {close.min():.2f}")
            print(f"   Max: {close.max():.2f}")
            print(f"   Mean: {close.mean():.2f}")
            print(f"   Current: {close.iloc[-1]:.2f}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("VIX & SPY Download Test")
    print("=" * 60)
    
    results = {}
    
    # Test VIX
    results['VIX'] = test_download('^VIX', 'VIX Index (Fear Gauge)')
    
    # Test SPY
    results['SPY'] = test_download('SPY', 'S&P 500 ETF')
    
    # Test QQQ
    results['QQQ'] = test_download('QQQ', 'Nasdaq 100 ETF')
    
    # Summary
    print(f"\n{'='*60}")
    print("[SUMMARY]")
    print('='*60)
    for symbol, success in results.items():
        status = "[OK]" if success else "[FAILED]"
        print(f"   {symbol}: {status}")
    
    if all(results.values()):
        print("\nAll downloads successful!")
    else:
        print("\nSome downloads failed!")
