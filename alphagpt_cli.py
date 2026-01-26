#!/usr/bin/env python
"""
AlphaGPT CLI - 互動式命令行控制介面

Usage:
    python alphagpt_cli.py
"""

import os
import sys
import subprocess
import json
from datetime import datetime

# Windows 編碼修復
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ==================== 配置 ====================
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 輔助函數 ====================
def clear_screen():
    """清屏"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_banner():
    """打印橫幅"""
    banner = """
╔═══════════════════════════════════════════════════════════════════╗
║     _    _       _           ____ ____ _____                      ║
║    / \\  | |_ __ | |__   __ _/ ___|  _ \\_   _|                    ║
║   / _ \\ | | '_ \\| '_ \\ / _` | |  _| |_) || |                     ║
║  / ___ \\| | |_) | | | | (_| | |_| |  __/ | |                      ║
║ /_/   \\_\\_| .__/|_| |_|\\__,_|\\____|_|    |_|                     ║
║           |_|                                                     ║
║                                                                   ║
║  🚀 AI-Powered Quantitative Factor Discovery System               ║
╚═══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_menu():
    """打印主菜單"""
    print("\n" + "=" * 60)
    print("📋 主菜單 - 請選擇功能")
    print("=" * 60)
    print("  1. 🚀 訓練新策略 (Train)")
    print("  2. 📊 回測策略 (Backtest)")
    print("  3. 📡 生成交易信號 (Signal)")
    print("  4. 📋 查看已有策略 (List)")
    print("  5. 🧹 清理輸出文件 (Clean)")
    print("  6. ❓ 幫助說明 (Help)")
    print("  0. 🚪 退出 (Exit)")
    print("=" * 60)


def get_input(prompt, default=None, required=True):
    """獲取用戶輸入"""
    if default:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "
    
    while True:
        value = input(prompt).strip()
        if not value:
            if default:
                return default
            elif required:
                print("⚠️  此項為必填，請重新輸入")
                continue
            else:
                return ""
        return value


def get_yes_no(prompt, default='n'):
    """獲取是/否輸入"""
    default_text = "Y/n" if default.lower() == 'y' else "y/N"
    value = input(f"{prompt} ({default_text}): ").strip().lower()
    if not value:
        return default.lower() == 'y'
    return value in ['y', 'yes', '是', '1']


def list_strategies():
    """列出所有可用的策略"""
    strategies = []
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith('_best_strategy.json'):
                filepath = os.path.join(OUTPUT_DIR, f)
                try:
                    with open(filepath, 'r') as file:
                        data = json.load(file)
                    strategies.append({
                        'file': f,
                        'symbol': data.get('symbol', 'Unknown'),
                        'formula': data.get('formula_readable', 'N/A'),
                        'train_sortino': data.get('train_sortino', 'N/A'),
                        'discrete_sharpe': data.get('discrete', {}).get('sharpe', 'N/A'),
                        'continuous_sharpe': data.get('continuous', {}).get('sharpe', 'N/A'),
                    })
                except:
                    pass
    return strategies


def select_strategy():
    """選擇策略文件"""
    strategies = list_strategies()
    
    if not strategies:
        print("\n❌ 沒有找到任何策略文件！")
        print("   請先使用選項 1 訓練一個策略")
        return None
    
    print("\n📋 可用的策略文件:")
    print("-" * 60)
    for i, s in enumerate(strategies, 1):
        sharpe = s.get('discrete_sharpe', 'N/A')
        if isinstance(sharpe, float):
            sharpe = f"{sharpe:.2f}"
        print(f"  {i}. {s['symbol']:<8} | Sharpe: {sharpe:<8} | {s['file']}")
    print("-" * 60)
    
    while True:
        choice = input(f"請選擇策略 (1-{len(strategies)}): ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(strategies):
                return os.path.join(OUTPUT_DIR, strategies[idx]['file'])
        except ValueError:
            pass
        print("⚠️  無效的選擇，請重新輸入")


# ==================== 功能模塊 ====================
def mode_train():
    """訓練模式"""
    print("\n" + "=" * 60)
    print("🚀 訓練新策略")
    print("=" * 60)
    
    # 獲取參數
    symbol = get_input("請輸入股票代碼 (例如: NVDA, SPY, AAPL)", default="SPY")
    start_date = get_input("請輸入訓練開始日期 (YYYY-MM-DD)", default="2015-01-01")
    end_date = get_input("請輸入訓練結束日期 (YYYY-MM-DD)", default="2024-01-01")
    test_end = get_input("請輸入測試結束日期 (YYYY-MM-DD)", default="2025-01-01")
    iterations = get_input("請輸入訓練迭代次數", default="400")
    batch_size = get_input("請輸入批次大小", default="1024")
    
    # 確認
    print("\n" + "-" * 60)
    print("📝 訓練配置確認:")
    print(f"   股票代碼    : {symbol}")
    print(f"   訓練期間    : {start_date} ~ {end_date}")
    print(f"   測試期間    : {end_date} ~ {test_end}")
    print(f"   迭代次數    : {iterations}")
    print(f"   批次大小    : {batch_size}")
    print("-" * 60)
    
    if not get_yes_no("確認開始訓練?", 'y'):
        print("已取消")
        return
    
    # 執行訓練
    print("\n🚀 開始訓練...")
    cmd = [
        sys.executable, 'times_us.py',
        '--symbol', symbol.upper(),
        '--start', start_date,
        '--end', end_date,
        '--test_end', test_end,
        '--iterations', iterations,
        '--batch_size', batch_size,
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n✅ 訓練完成！")
        strategy_file = os.path.join(OUTPUT_DIR, f'{symbol.upper()}_best_strategy.json')
        if os.path.exists(strategy_file):
            print(f"📁 策略已保存到: {strategy_file}")
    else:
        print(f"\n❌ 訓練失敗 (錯誤碼: {result.returncode})")
    
    input("\n按 Enter 鍵繼續...")


def mode_backtest():
    """回測模式"""
    print("\n" + "=" * 60)
    print("📊 回測策略")
    print("=" * 60)
    
    # 選擇策略
    strategy_path = select_strategy()
    if not strategy_path:
        input("\n按 Enter 鍵繼續...")
        return
    
    # 獲取參數
    tickers = get_input("請輸入要測試的股票代碼 (多個用逗號分隔, 例如: SPY,QQQ,AAPL)", default="SPY")
    period = get_input("請輸入回測週期 (例如: 6mo, 1y, 2y, ytd)", default="1y")
    capital = get_input("請輸入初始資金 (USD)", default="100000")
    
    export_logs = get_yes_no("是否導出交易記錄 (CSV)?", 'n')
    generate_charts = get_yes_no("是否生成圖表?", 'y')
    
    # 確認
    print("\n" + "-" * 60)
    print("📝 回測配置確認:")
    print(f"   策略文件    : {os.path.basename(strategy_path)}")
    print(f"   測試標的    : {tickers}")
    print(f"   回測週期    : {period}")
    print(f"   初始資金    : ${float(capital):,.2f}")
    print(f"   導出記錄    : {'是' if export_logs else '否'}")
    print(f"   生成圖表    : {'是' if generate_charts else '否'}")
    print("-" * 60)
    
    if not get_yes_no("確認開始回測?", 'y'):
        print("已取消")
        return
    
    # 執行回測
    print("\n📊 開始回測...")
    cmd = [
        sys.executable, 'backtest_strategy.py',
        '--strategy', strategy_path,
        '--tickers', tickers.upper(),
        '--period', period,
        '--capital', capital,
    ]
    
    if export_logs:
        cmd.append('--export')
    if generate_charts:
        cmd.append('--plot')
    
    subprocess.run(cmd)
    
    input("\n按 Enter 鍵繼續...")


def mode_signal():
    """信號生成模式"""
    print("\n" + "=" * 60)
    print("📡 生成交易信號")
    print("=" * 60)
    
    # 選擇策略
    strategy_path = select_strategy()
    if not strategy_path:
        input("\n按 Enter 鍵繼續...")
        return
    
    # 獲取參數
    symbols = get_input(
        "請輸入要掃描的股票代碼 (多個用逗號分隔)", 
        default="SPY,QQQ,AAPL,MSFT,NVDA,TSLA,AMD,GOOGL,META"
    )
    
    monitor_mode = get_yes_no("是否啟用監控模式 (持續更新)?", 'n')
    interval = 60
    if monitor_mode:
        interval = get_input("請輸入更新間隔 (秒)", default="60")
    
    # 執行
    print("\n📡 生成信號...")
    cmd = [
        sys.executable, 'signal_generator.py',
        '--strategy', strategy_path,
        '--symbols', symbols.upper(),
    ]
    
    if monitor_mode:
        cmd.extend(['--monitor', '--interval', str(interval)])
    
    subprocess.run(cmd)
    
    if not monitor_mode:
        input("\n按 Enter 鍵繼續...")


def mode_list():
    """列出策略和數據"""
    print("\n" + "=" * 60)
    print("📋 已有策略和數據")
    print("=" * 60)
    
    strategies = list_strategies()
    
    if strategies:
        print("\n📜 策略文件:")
        print("-" * 80)
        print(f"{'文件名':<35} {'代碼':<8} {'Sortino':<10} {'Sharpe(D/C)':<15}")
        print("-" * 80)
        
        for s in strategies:
            sortino = s.get('train_sortino', 'N/A')
            if isinstance(sortino, float):
                sortino = f"{sortino:.2f}"
            
            sharpe_d = s.get('discrete_sharpe', 'N/A')
            sharpe_c = s.get('continuous_sharpe', 'N/A')
            if isinstance(sharpe_d, float):
                sharpe_d = f"{sharpe_d:.2f}"
            if isinstance(sharpe_c, float):
                sharpe_c = f"{sharpe_c:.2f}"
            
            print(f"{s['file']:<35} {s['symbol']:<8} {sortino:<10} {sharpe_d}/{sharpe_c}")
        
        print("-" * 80)
        print("\n📜 公式詳情:")
        for s in strategies:
            formula = s.get('formula', 'N/A')
            print(f"  {s['symbol']}: {formula}")
    else:
        print("\n  ❌ 沒有找到任何策略文件")
        print("     請先使用選項 1 訓練一個策略")
    
    # 數據緩存
    print("\n📁 數據緩存:")
    print("-" * 50)
    caches = []
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            if f.startswith('data_cache_') and f.endswith('.parquet'):
                filepath = os.path.join(OUTPUT_DIR, f)
                size = os.path.getsize(filepath)
                print(f"  {f:<35} {size/1024:.1f} KB")
                caches.append(f)
    
    if not caches:
        print("  (無)")
    
    # 統計
    if os.path.exists(OUTPUT_DIR):
        all_files = os.listdir(OUTPUT_DIR)
        png_files = [f for f in all_files if f.endswith('.png')]
        csv_files = [f for f in all_files if f.endswith('.csv')]
        
        print(f"\n📊 輸出統計:")
        print(f"  圖表文件: {len(png_files)} 個")
        print(f"  日誌文件: {len(csv_files)} 個")
        print(f"  總計    : {len(all_files)} 個文件在 {OUTPUT_DIR}/")
    
    input("\n按 Enter 鍵繼續...")


def mode_clean():
    """清理輸出文件"""
    print("\n" + "=" * 60)
    print("🧹 清理輸出文件")
    print("=" * 60)
    
    if not os.path.exists(OUTPUT_DIR):
        print("\n  輸出目錄不存在")
        input("\n按 Enter 鍵繼續...")
        return
    
    files = os.listdir(OUTPUT_DIR)
    if not files:
        print("\n  輸出目錄已經是空的")
        input("\n按 Enter 鍵繼續...")
        return
    
    # 分類
    strategy_files = [f for f in files if f.endswith('_best_strategy.json')]
    cache_files = [f for f in files if f.endswith('.parquet')]
    chart_files = [f for f in files if f.endswith('.png')]
    log_files = [f for f in files if f.endswith('.csv')]
    
    print(f"\n📁 當前文件統計:")
    print(f"  策略文件: {len(strategy_files)} 個")
    print(f"  緩存文件: {len(cache_files)} 個")
    print(f"  圖表文件: {len(chart_files)} 個")
    print(f"  日誌文件: {len(log_files)} 個")
    
    print("\n請選擇要清理的內容:")
    print("  1. 只清理圖表和日誌 (保留策略和緩存)")
    print("  2. 清理緩存文件")
    print("  3. 清理圖表文件")
    print("  4. 清理日誌文件")
    print("  5. 清理全部文件 ⚠️")
    print("  0. 取消")
    
    choice = input("\n請選擇 (0-5): ").strip()
    
    to_delete = []
    
    if choice == '1':
        to_delete = chart_files + log_files
    elif choice == '2':
        to_delete = cache_files
    elif choice == '3':
        to_delete = chart_files
    elif choice == '4':
        to_delete = log_files
    elif choice == '5':
        to_delete = files
    else:
        print("已取消")
        input("\n按 Enter 鍵繼續...")
        return
    
    if not to_delete:
        print("沒有要刪除的文件")
        input("\n按 Enter 鍵繼續...")
        return
    
    print(f"\n將刪除 {len(to_delete)} 個文件")
    if not get_yes_no("確認刪除?", 'n'):
        print("已取消")
        input("\n按 Enter 鍵繼續...")
        return
    
    deleted = 0
    for f in to_delete:
        try:
            os.remove(os.path.join(OUTPUT_DIR, f))
            deleted += 1
        except Exception as e:
            print(f"  ⚠️ 無法刪除 {f}: {e}")
    
    print(f"\n✅ 已刪除 {deleted} 個文件")
    input("\n按 Enter 鍵繼續...")


def mode_help():
    """顯示幫助"""
    print("\n" + "=" * 60)
    print("❓ 幫助說明")
    print("=" * 60)
    
    print("""
📖 功能說明:

  1. 🚀 訓練新策略
     - 使用深度強化學習自動發現量化因子
     - 需要指定股票代碼、訓練期間、測試期間
     - 訓練完成後會保存策略到 output/ 目錄

  2. 📊 回測策略
     - 使用已訓練的策略對股票進行回測
     - 同時顯示離散倉位和連續倉位的結果
     - 可以生成圖表和交易記錄

  3. 📡 生成交易信號
     - 使用策略對當前市場生成買/賣信號
     - 支持監控模式（持續更新）
     - 可以同時掃描多個股票

  4. 📋 查看已有策略
     - 列出所有已訓練的策略
     - 顯示策略的公式和表現指標

  5. 🧹 清理輸出文件
     - 清理圖表、日誌、緩存等文件
     - 可以選擇性清理

📁 輸出目錄: output/

🔧 檔案說明:
  - *_best_strategy.json  : 策略配置文件
  - *_strategy_performance.png : 訓練結果圖表
  - *_backtest.png : 回測結果圖表
  - *_trade_log.csv : 交易記錄
  - data_cache_*.parquet : 數據緩存

💡 提示:
  - 首次使用請先訓練一個策略
  - 訓練需要一定時間（取決於迭代次數）
  - 建議使用 GPU 加速訓練（自動檢測）
  - 回測同時顯示離散(±1)和連續(信號強度)倉位結果
""")
    
    input("\n按 Enter 鍵繼續...")


# ==================== 主程序 ====================
def main():
    """主程序入口"""
    
    while True:
        clear_screen()
        print_banner()
        print_menu()
        
        choice = input("\n請輸入選項 (0-6): ").strip()
        
        if choice == '1':
            mode_train()
        elif choice == '2':
            mode_backtest()
        elif choice == '3':
            mode_signal()
        elif choice == '4':
            mode_list()
        elif choice == '5':
            mode_clean()
        elif choice == '6':
            mode_help()
        elif choice == '0':
            print("\n👋 再見！")
            break
        else:
            print("\n⚠️ 無效的選項，請輸入 0-6")
            input("按 Enter 鍵繼續...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 再見！")
        sys.exit(0)
