#!/usr/bin/env python
"""
AlphaGPT CLI - 互動式命令行控制介面

支持 V1 和 V2 訓練模式:
- V1: 標準訓練 (times_us.py)
- V2: 防過擬合版 (times_us_v2.py) - Walk-Forward + 多目標獎勵

Usage:
    python alphagpt_cli.py
"""

import os
import sys
import subprocess
import json
import time
import multiprocessing
import shutil
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# 延遲導入 matplotlib（只在需要時導入）
def get_plt():
    import matplotlib.pyplot as plt
    return plt

# Windows 編碼修復
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ==================== 配置 ====================
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 信號閾值配置 ====================
DEFAULT_SIGNAL_THRESHOLD = 0.1  # 默認閾值
DEFAULT_HYSTERESIS = 0.05       # 死區/滯後閾值，避免頻繁交易

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
║  📊 V2: Anti-Overfitting Edition                                  ║
╚═══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_menu():
    """打印主菜單"""
    print("\n" + "=" * 60)
    print("📋 主菜單 - 請選擇功能")
    print("=" * 60)
    print("  1. 🚀 訓練新策略 (多任務)")
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
    """列出所有可用的策略（包括子目錄）"""
    strategies = []
    if os.path.exists(OUTPUT_DIR):
        # 遍歷 output 目錄及其子目錄
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for f in files:
                if f in ['best_strategy.json', 'best_strategy_v2.json'] or f.endswith('_best_strategy.json'):
                    filepath = os.path.join(root, f)
                    try:
                        with open(filepath, 'r') as file:
                            data = json.load(file)
                        # 使用相對於 OUTPUT_DIR 的路徑
                        rel_path = os.path.relpath(filepath, OUTPUT_DIR)
                        
                        # 判斷版本
                        version = data.get('version', '1.0')
                        is_v2 = version.startswith('2') or 'v2' in f.lower()
                        
                        # 獲取指標
                        if is_v2:
                            test_metrics = data.get('test_metrics', {})
                            single_metrics = test_metrics.get('single', {})
                            sharpe = single_metrics.get('sharpe', 'N/A')
                            max_dd = single_metrics.get('max_dd', 'N/A')
                        else:
                            sharpe = data.get('discrete', {}).get('sharpe', 'N/A')
                            max_dd = data.get('discrete', {}).get('max_drawdown', 'N/A')
                        
                        strategies.append({
                            'file': rel_path,
                            'filepath': filepath,
                            'symbol': data.get('symbol', 'Unknown'),
                            'version': 'V2' if is_v2 else 'V1',
                            'signal_threshold': data.get('signal_threshold', DEFAULT_SIGNAL_THRESHOLD),
                            'formula': data.get('formula_readable', 'N/A'),
                            'train_score': data.get('train_sortino', data.get('train_score', 'N/A')),
                            'sharpe': sharpe,
                            'max_dd': max_dd,
                            'features': data.get('features', ['RET', 'RET5', 'VOL_CHG', 'V_RET', 'TREND']),
                        })
                    except:
                        pass
    # 按修改時間排序（最新的在前）
    strategies.sort(key=lambda x: os.path.getmtime(x['filepath']), reverse=True)
    return strategies


def select_strategy():
    """選擇策略文件"""
    strategies = list_strategies()
    
    if not strategies:
        print("\n❌ 沒有找到任何策略文件！")
        print("   請先使用選項 1 或 2 訓練一個策略")
        return None
    
    print("\n📋 可用的策略文件 (按時間排序，最新在前):")
    print("-" * 90)
    print(f"  {'#':<3} {'Ver':<4} {'Symbol':<8} {'Thresh':<7} {'Sharpe':<8} {'MaxDD':<8} {'Path':<40}")
    print("-" * 90)
    for i, s in enumerate(strategies, 1):
        sharpe = s.get('sharpe', 'N/A')
        if isinstance(sharpe, float):
            sharpe = f"{sharpe:.2f}"
        max_dd = s.get('max_dd', 'N/A')
        if isinstance(max_dd, float):
            max_dd = f"{max_dd:.1%}"
        thresh = s.get('signal_threshold', DEFAULT_SIGNAL_THRESHOLD)
        thresh_str = f"±{thresh}"
        # 截斷過長的路徑
        filepath = s['file']
        if len(filepath) > 38:
            filepath = "..." + filepath[-35:]
        print(f"  {i:<3} {s['version']:<4} {s['symbol']:<8} {thresh_str:<7} {sharpe:<8} {max_dd:<8} {filepath}")
    print("-" * 90)
    
    while True:
        choice = input(f"請選擇策略 (1-{len(strategies)}): ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(strategies):
                return strategies[idx]['filepath']
        except ValueError:
            pass
        print("⚠️  無效的選擇，請重新輸入")


# ==================== 功能模塊 ====================
def run_single_training(args_dict):
    """
    運行單個訓練任務
    
    Args:
        args_dict: 包含所有參數的字典
    
    Returns:
        (worker_id, return_code, output_folder)
    """
    worker_id = args_dict['worker_id']
    
    cmd = [
        sys.executable, 'times_us_v2.py',
        '--symbol', args_dict['symbol'],
        '--threshold', args_dict['threshold'],
        '--hysteresis', args_dict['hysteresis'],
        '--start', args_dict['start_date'],
        '--end', args_dict['end_date'],
        '--test_end', args_dict['test_end'],
        '--iterations', args_dict['iterations'],
        '--batch_size', args_dict['batch_size'],
        '--seed', str(args_dict['seed']),  # 不同的隨機種子
    ]
    
    if args_dict['use_walk_forward']:
        cmd.append('--walk_forward')
    else:
        cmd.append('--no_walk_forward')
    
    # 運行訓練
    try:
        result = subprocess.run(cmd, capture_output=False)
        return (worker_id, result.returncode, None)
    except Exception as e:
        return (worker_id, -1, str(e))


def mode_train_v2():
    """訓練模式 V2（防過擬合版）"""
    print("\n" + "=" * 70)
    print("🚀 訓練新策略 V2 (防過擬合版) ⭐推薦")
    print("=" * 70)
    print("""
V2 版本核心改進:
  1. Walk-Forward 滾動驗證 - 確保因子在不同時間段穩定
  2. 多目標獎勵 - Sortino + Sharpe + Return - MaxDD - Complexity
  3. 更多因子 - 11個 (含 ATR, RSI, CLV, RS, MOM, VIX)
  4. Ensemble 策略 - Top-K 公式信號平均
  5. 複雜度懲罰 - 奧卡姆剃刀原則
  6. 多次任務 - 不同種子探索更多可能性 ⭐NEW
""")
    
    # 獲取參數
    symbol = get_input("請輸入股票代碼 (例如: SPY, QQQ, NVDA)", default="SPY")
    
    start_date = get_input("請輸入訓練開始日期 (YYYY-MM-DD)", default="2015-01-01")
    end_date = get_input("請輸入訓練結束日期 (YYYY-MM-DD)", default="2024-01-01")
    test_end = get_input("請輸入測試結束日期 (YYYY-MM-DD)", default="2026-01-27")
    iterations = get_input("請輸入訓練迭代次數", default="300")
    batch_size = get_input("請輸入批次大小", default="512")
    
    # Walk-Forward 選項
    print("\n📊 Walk-Forward 滾動驗證:")
    print("   啟用後會在多個時間窗口訓練和驗證，確保因子穩健")
    use_walk_forward = get_yes_no("是否啟用 Walk-Forward?", 'y')
    
    # 信號閾值
    print(f"\n📊 信號閾值設置 (|signal| < threshold → 觀望/不交易)")
    threshold = get_input(f"請輸入信號閾值 (0~1之間)", default=str(DEFAULT_SIGNAL_THRESHOLD))
    
    # 滯後閾值（死區）
    print(f"\n📊 滯後閾值設置 (避免信號在閾值附近震盪時頻繁交易)")
    print(f"   開倉閾值 = threshold = {threshold}")
    print(f"   平倉閾值 = threshold - hysteresis")
    hysteresis = get_input(f"請輸入滯後值 (0~0.1之間, 0表示禁用)", default=str(DEFAULT_HYSTERESIS))
    
    # ==================== 多次任務選項 ====================
    print(f"\n🔄 多次任務訓練:")
    print("   使用不同的隨機種子執行多次訓練，探索更多可能性")
    print("   每次任務完成後再開始下一次（串行執行，GPU 100% 利用）")
    print("   最後自動整合所有結果，選出最佳策略")
    
    use_multi_task = get_yes_no("是否啟用多次任務訓練?", 'n')
    
    num_workers = 1
    if use_multi_task:
        num_workers = int(get_input(f"請輸入任務次數 (建議 2~6)", default="4"))
        num_workers = max(1, min(num_workers, 20))  # 限制範圍
    
    # 確認
    print("\n" + "-" * 70)
    print("📝 訓練配置確認 (V2 防過擬合版):")
    print(f"   股票代碼    : {symbol}")
    print(f"   信號閾值    : ±{threshold} (滯後: {hysteresis})")
    print(f"   訓練期間    : {start_date} ~ {end_date}")
    print(f"   測試期間    : {end_date} ~ {test_end}")
    print(f"   迭代次數    : {iterations}")
    print(f"   批次大小    : {batch_size}")
    print(f"   Walk-Forward: {'啟用' if use_walk_forward else '禁用'}")
    print(f"   因子數量    : 11 (含 ATR, RSI, CLV, RS, MOM, VIX)")
    print(f"   獎勵機制    : 多目標 (Sortino + Sharpe + Return - MaxDD)")
    if use_multi_task:
        print(f"   任務次數    : {num_workers} 次 🔄")
        print(f"   總探索量    : {num_workers} × {iterations} = {num_workers * int(iterations)} 輪")
    print("-" * 70)
    
    if not get_yes_no("確認開始訓練?", 'y'):
        print("已取消")
        return
    
    # ==================== 執行訓練 ====================
    if num_workers == 1:
        # 單進程模式
        print("\n🚀 開始訓練 (V2 單進程模式)...")
        cmd = [
            sys.executable, 'times_us_v2.py',
            '--symbol', symbol.upper(),
            '--threshold', threshold,
            '--hysteresis', hysteresis,
            '--start', start_date,
            '--end', end_date,
            '--test_end', test_end,
            '--iterations', iterations,
            '--batch_size', batch_size,
        ]
        
        if use_walk_forward:
            cmd.append('--walk_forward')
        else:
            cmd.append('--no_walk_forward')
        
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print("\n✅ 訓練完成！")
            print(f"📁 策略已保存到 output/ 子目錄中")
        else:
            print(f"\n❌ 訓練失敗 (錯誤碼: {result.returncode})")
        
        input("\n按 Enter 鍵繼續...")
        return  # 單進程模式結束後直接返回
    
    # ==================== 多次任務模式（串行執行）====================
    print(f"\n🔄 開始多次任務訓練 ({num_workers} 次任務，串行執行)...")
    print("=" * 70)
    
    # 1. 預先創建 MERGED 文件夾和各任務子文件夾
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    merged_folder = os.path.join(OUTPUT_DIR, f"{symbol.upper()}_v2_MERGED_{timestamp}")
    os.makedirs(merged_folder, exist_ok=True)
    
    task_folders = []
    for i in range(num_workers):
        task_folder = os.path.join(merged_folder, f"task_{i+1}")
        os.makedirs(task_folder, exist_ok=True)
        task_folders.append(task_folder)
    
    print(f"📁 輸出目錄: {merged_folder}")
    print(f"   └── task_1/ ~ task_{num_workers}/")
    
    # 2. 預加載數據緩存
    print("\n📦 正在預加載數據緩存...")
    cache_cmd = [
        sys.executable, '-c',
        f'''
import sys
sys.path.insert(0, ".")
from times_us_v2 import USDataEngineV2
engine = USDataEngineV2("{symbol.upper()}", "{start_date}", "{end_date}", "{test_end}")
engine.load()
print("✅ 數據緩存已生成")
'''
    ]
    subprocess.run(cache_cmd, capture_output=True)
    
    # 3. 準備任務參數
    base_seed = int(time.time()) % 10000
    task_args = []
    for i in range(num_workers):
        task_args.append({
            'task_id': i + 1,
            'symbol': symbol.upper(),
            'threshold': threshold,
            'hysteresis': hysteresis,
            'start_date': start_date,
            'end_date': end_date,
            'test_end': test_end,
            'iterations': iterations,
            'batch_size': batch_size,
            'use_walk_forward': use_walk_forward,
            'seed': base_seed + i * 1000,
            'output_dir': task_folders[i],
        })
    
    print(f"\n🎲 隨機種子: {base_seed}, {base_seed + 1000}, ..., {base_seed + (num_workers-1) * 1000}")
    print("-" * 70)
    
    # 4. 串行執行任務（帶中斷保護）
    start_time = time.time()
    results = []
    interrupted = False
    
    try:
        for i, args in enumerate(task_args):
            task_start = time.time()
            print(f"\n{'='*70}")
            print(f"🚀 任務 {args['task_id']}/{num_workers} 開始 (seed={args['seed']})")
            print(f"{'='*70}")
            
            cmd = [
                sys.executable, 'times_us_v2.py',
                '--symbol', args['symbol'],
                '--threshold', args['threshold'],
                '--hysteresis', args['hysteresis'],
                '--start', args['start_date'],
                '--end', args['end_date'],
                '--test_end', args['test_end'],
                '--iterations', args['iterations'],
                '--batch_size', args['batch_size'],
                '--seed', str(args['seed']),
                '--output_dir', args['output_dir'],
            ]
            if args['use_walk_forward']:
                cmd.append('--walk_forward')
            else:
                cmd.append('--no_walk_forward')
            
            # 執行任務（同步等待完成）
            result = subprocess.run(cmd)
            task_elapsed = time.time() - task_start
            
            if result.returncode == 0:
                print(f"\n✅ 任務 {args['task_id']} 完成！耗時: {task_elapsed/60:.1f} 分鐘")
                results.append((args['task_id'], 0, args['seed'], args['output_dir']))
            else:
                print(f"\n❌ 任務 {args['task_id']} 失敗 (code={result.returncode})")
                results.append((args['task_id'], result.returncode, args['seed'], args['output_dir']))
            
            # 顯示進度
            completed = i + 1
            remaining = num_workers - completed
            if remaining > 0:
                avg_time = (time.time() - start_time) / completed
                eta = avg_time * remaining
                print(f"\n📊 進度: {completed}/{num_workers} | 預計剩餘: {eta/60:.1f} 分鐘")
        
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 70)
        print(f"🏁 多次任務訓練完成！總耗時: {elapsed/60:.1f} 分鐘")
        print("=" * 70)
        
        # 統計結果
        success_count = sum(1 for _, code, _, _ in results if code == 0)
        print(f"   成功: {success_count}/{num_workers}")
        
        if success_count > 0:
            print("\n📊 整合訓練結果...")
            successful_tasks = [(tid, odir) for tid, code, _, odir in results if code == 0]
            finalize_merged_results(merged_folder, successful_tasks, symbol.upper())
    
    except KeyboardInterrupt:
        interrupted = True
        print("\n" + "=" * 70)
        print("🛑 檢測到中斷 (Ctrl+C)")
        print("=" * 70)
        
        # 串行執行時，當前任務會被終止，已完成的任務結果保留
        completed_count = len(results)
        print(f"   已完成任務: {completed_count}/{num_workers}")
        
        if completed_count > 0:
            print(f"💡 提示：已完成的 {completed_count} 個任務結果保存在 {merged_folder}")
            # 整合已完成的任務
            successful_tasks = [(tid, odir) for tid, code, _, odir in results if code == 0]
            if successful_tasks:
                print("\n📊 整合已完成的訓練結果...")
                finalize_merged_results(merged_folder, successful_tasks, symbol.upper())
    
    if not interrupted:
        input("\n按 Enter 鍵繼續...")
    else:
        input("\n訓練已中斷。按 Enter 鍵返回主菜單...")


def finalize_merged_results(merged_folder, successful_tasks, symbol):
    """
    整合多次任務訓練結果（基於已知的任務文件夾）
    
    Args:
        merged_folder: 預先創建的 MERGED 文件夾路徑
        successful_tasks: 成功完成的任務列表 [(task_id, output_dir), ...]
        symbol: 股票代碼
    
    Returns:
        str: 整合文件夾路徑
    """
    print("\n" + "=" * 70)
    print("📦 整合多次任務訓練結果")
    print("=" * 70)
    
    # 1. 從已知的任務文件夾讀取策略
    strategies = []
    for task_id, output_dir in successful_tasks:
        strategy_file = os.path.join(output_dir, 'best_strategy_v2.json')
        if os.path.exists(strategy_file):
            try:
                with open(strategy_file, 'r') as f:
                    data = json.load(f)
                
                # 獲取測試指標
                test_metrics = data.get('test_metrics', {})
                single = test_metrics.get('single', {})
                
                # 獲取 Top-K 公式
                top_k = data.get('top_k_formulas', [])
                
                strategies.append({
                    'task_id': task_id,
                    'folder': f"task_{task_id}",
                    'folder_path': output_dir,
                    'filepath': strategy_file,
                    'formula': data.get('formula_readable', 'N/A'),
                    'tokens': data.get('formula_tokens', []),
                    'train_score': data.get('train_score', 0),
                    'test_sharpe': single.get('sharpe', 0),
                    'test_return': single.get('total_ret', 0),
                    'test_maxdd': single.get('max_dd', 0),
                    'top_k_formulas': top_k,
                    'test_period': data.get('test_period', ''),
                    'benchmark_return': data.get('benchmark_return', 0),
                    'norm_params': data.get('norm_params', {}),
                })
                print(f"   ✓ 任務 {task_id}: 策略已讀取")
            except Exception as e:
                print(f"   ⚠️ 任務 {task_id}: 無法讀取策略 - {e}")
        else:
            print(f"   ⚠️ 任務 {task_id}: 策略文件不存在")
    
    if not strategies:
        print("   ❌ 沒有找到有效的策略文件")
        return merged_folder
    
    # 按測試 Sharpe 排序
    strategies.sort(key=lambda x: x.get('test_sharpe', 0), reverse=True)
    print(f"\n   找到 {len(strategies)} 個有效策略")
    
    # ==================== 打印排名 ====================
    print("\n" + "-" * 90)
    print(f"🏆 Top {min(10, len(strategies))} 策略 (按測試 Sharpe 排序):")
    print("-" * 90)
    print(f"  {'#':<3} {'Task':>6} {'Sharpe':>8} {'Return':>10} {'MaxDD':>8} {'Train':>8}")
    print("-" * 90)
    
    for i, s in enumerate(strategies[:10], 1):
        sharpe = s.get('test_sharpe', 0)
        ret = s.get('test_return', 0)
        maxdd = s.get('test_maxdd', 0)
        train = s.get('train_score', 0)
        tid = s['task_id']
        
        print(f"  {i:<3} {tid:>6} {sharpe:>8.2f} {ret:>9.1%} {maxdd:>7.1%} {train:>8.2f}")
    
    print("-" * 90)
    
    # ==================== 收集所有 Top-K 公式 ====================
    all_formulas = []
    for s in strategies:
        for f in s.get('top_k_formulas', []):
            all_formulas.append({
                'formula': f.get('formula', ''),
                'tokens': f.get('tokens', []),
                'score': f.get('score', 0),
                'source': f"task_{s['task_id']}",
            })
    
    # 按 score 排序，取 Top 20
    all_formulas.sort(key=lambda x: x.get('score', 0), reverse=True)
    top_20_formulas = all_formulas[:20]
    
    # ==================== 收集所有 TASK 的 TOP3 公式（用於對比圖）====================
    all_top3_formulas = []
    for s in strategies:
        task_id = s['task_id']
        top_k = s.get('top_k_formulas', [])
        # 獲取該任務的閾值和滯後值
        strategy_file = s.get('filepath', '')
        threshold = 0.1
        hysteresis = 0.05
        try:
            if strategy_file and os.path.exists(strategy_file):
                with open(strategy_file, 'r') as f:
                    strategy_data = json.load(f)
                threshold = strategy_data.get('signal_threshold', 0.1)
                hysteresis = strategy_data.get('hysteresis', 0.05)
        except:
            pass
        
        # 取該任務的 TOP3
        for rank, f in enumerate(top_k[:3], 1):
            all_top3_formulas.append({
                'task_id': task_id,
                'rank': rank,
                'label': f"T{task_id}-Top{rank}",
                'formula': f.get('formula', ''),
                'tokens': f.get('tokens', []),
                'score': f.get('score', 0),
                'threshold': threshold,
                'hysteresis': hysteresis,
                'is_ensemble': False,
            })
        
        # 檢查該任務是否有 ensemble 策略
        task_folder = s.get('folder_path', '')
        ensemble_folder = os.path.join(task_folder, f'ensemble_top{len(top_k)}') if task_folder else ''
        if not os.path.exists(ensemble_folder):
            # 嘗試其他可能的 ensemble 文件夾名稱
            for k in [10, 5, 3]:
                ensemble_folder = os.path.join(task_folder, f'ensemble_top{k}')
                if os.path.exists(ensemble_folder):
                    break
        
        ensemble_file = os.path.join(ensemble_folder, 'best_strategy_v2.json') if ensemble_folder else ''
        if ensemble_file and os.path.exists(ensemble_file):
            try:
                with open(ensemble_file, 'r') as f:
                    ensemble_data = json.load(f)
                if ensemble_data.get('type') == 'ensemble':
                    all_top3_formulas.append({
                        'task_id': task_id,
                        'rank': 'E',  # E for Ensemble
                        'label': f"T{task_id}-Ensemble",
                        'formula': f"Ensemble({ensemble_data.get('ensemble_count', '?')})",
                        'tokens': None,  # ensemble 沒有單一 tokens
                        'tokens_list': ensemble_data.get('formula_tokens_list', []),
                        'weights': [c.get('weight', 0.1) for c in ensemble_data.get('component_formulas', [])],
                        'score': ensemble_data.get('train_score', 0),
                        'threshold': ensemble_data.get('signal_threshold', 0.1),
                        'hysteresis': ensemble_data.get('hysteresis', 0.05),
                        'is_ensemble': True,
                        'norm_params': ensemble_data.get('norm_params', {}),
                    })
            except Exception as e:
                pass
    
    # 統計
    num_ensembles = sum(1 for f in all_top3_formulas if f.get('is_ensemble', False))
    print(f"\n   📋 收集了 {len(all_top3_formulas)} 個公式（含 {num_ensembles} 個 Ensemble）來自 {len(strategies)} 個任務")
    
    # ==================== 保存整合的策略 JSON ====================
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    merged_strategy = {
        'version': '2.0-merged',
        'symbol': symbol,
        'num_tasks': len(successful_tasks),
        'merged_time': timestamp,
        'total_strategies': len(strategies),
        'best_strategy': strategies[0] if strategies else None,
        'top_20_formulas': top_20_formulas,
        'task_ids': [s['task_id'] for s in strategies],
    }
    
    merged_file = os.path.join(merged_folder, 'merged_strategies.json')
    with open(merged_file, 'w') as f:
        json.dump(merged_strategy, f, indent=2, default=str)
    print(f"\n💾 整合策略已保存: {merged_file}")
    
    # ==================== 複製最佳策略 ====================
    if strategies:
        best = strategies[0]
        # 複製最佳策略文件
        best_src = best['filepath']
        best_dst = os.path.join(merged_folder, 'best_strategy_v2.json')
        shutil.copy(best_src, best_dst)
        print(f"📋 最佳策略已複製: best_strategy_v2.json (來自任務 {best['task_id']})")
    
    # ==================== 繪製綜合對比圖表 ====================
    print("\n📊 繪製綜合對比圖表...")
    plot_merged_comparison(strategies, symbol, merged_folder, all_top3_formulas)
    
    # ==================== 生成報告 ====================
    report_file = os.path.join(merged_folder, 'merged_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"🏆 AlphaGPT V2 多次任務訓練整合報告 - {symbol}\n")
        f.write("=" * 80 + "\n")
        f.write(f"整合時間: {timestamp}\n")
        f.write(f"成功任務: {len(successful_tasks)}\n")
        f.write(f"有效策略: {len(strategies)}\n")
        f.write("-" * 80 + "\n\n")
        
        f.write("🏆 Top 10 策略 (按測試 Sharpe 排序):\n")
        f.write("-" * 80 + "\n")
        for i, s in enumerate(strategies[:10], 1):
            f.write(f"{i}. 任務 {s['task_id']}: Sharpe={s['test_sharpe']:.2f} | "
                    f"Return={s['test_return']:.1%} | MaxDD={s['test_maxdd']:.1%}\n")
            formula_str = s['formula'][:70] if len(s['formula']) > 70 else s['formula']
            f.write(f"   Formula: {formula_str}...\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("\n📜 Top 20 公式 (跨所有任務):\n")
        f.write("-" * 80 + "\n")
        for i, f_info in enumerate(top_20_formulas, 1):
            formula_str = f_info['formula'][:60] if len(f_info['formula']) > 60 else f_info['formula']
            f.write(f"{i}. Score={f_info['score']:.2f} | {formula_str}...\n")
    
    print(f"📝 報告已保存: {report_file}")
    
    print(f"\n✅ 整合完成！輸出文件夾: {merged_folder}")
    # ==================== 繪製分析圖表 ====================
    print("\n📊 繪製策略分析圖表...")
    plot_strategy_analysis(strategies, symbol, merged_folder, all_top3_formulas)
    
    print(f"""
📁 文件夾結構:
   {os.path.basename(merged_folder)}/
   ├── merged_strategies.json   # 整合策略（含 Top 20 公式）
   ├── best_strategy_v2.json    # 最佳策略 (任務 {strategies[0]['task_id'] if strategies else '?'})
   ├── merged_comparison.png    # 綜合對比圖表（含各任務 Ensemble）
   ├── strategy_analysis.png    # 策略分析（Sharpe/複雜度/過擬合）
   ├── merged_report.txt        # 文字報告
   └── task_1/ ~ task_N/        # 各任務原始輸出
""")
    
    return merged_folder


def plot_merged_comparison(strategies, symbol, output_folder, all_top3_formulas=None):
    """
    繪製多次任務訓練結果的綜合對比圖表
    
    新版本：收集所有 TASK 的 TOP3 公式，進行實際回測並繪製收益曲線對比
    
    Args:
        strategies: 各任務策略列表
        symbol: 股票代碼
        output_folder: 輸出文件夾
        all_top3_formulas: 所有 TASK 的 TOP3 公式列表（用於回測和繪圖）
    """
    plt = get_plt()
    
    if len(strategies) < 1:
        print("   ⚠️ 策略數量不足，無法繪圖")
        return
    
    plt.style.use('bmh')
    
    # 顏色配置（擴展更多顏色以支持多公式）
    colors = ['#2E86AB', '#28A745', '#F39C12', '#E74C3C', '#9B59B6', '#1ABC9C',
              '#3498DB', '#2ECC71', '#F1C40F', '#E67E22', '#9B59B6', '#1ABC9C']
    
    # ==================== 如果有 all_top3_formulas，進行實際回測對比 ====================
    if all_top3_formulas and len(all_top3_formulas) > 0:
        print(f"   📊 正在回測 {len(all_top3_formulas)} 個公式...")
        
        try:
            # 導入必要的模組
            from times_us_v2 import USDataEngineV2, discretize_position, COST_RATE, FEATURES, OPS_CONFIG, OP_FUNC_MAP, OP_ARITY_MAP
            import torch
            
            # 獲取數據引擎參數（從第一個策略）
            first_strategy = strategies[0]
            test_period = first_strategy.get('test_period', '')
            
            # 解析測試期間
            if '~' in test_period:
                parts = test_period.split('~')
                test_start = parts[0].strip()
                test_end = parts[1].strip()
            else:
                test_start = '2024-01-01'
                test_end = '2026-01-27'
            
            # 從策略獲取訓練期間
            train_start = '2015-01-01'  # 默認值
            train_end = test_start  # 訓練結束 = 測試開始
            
            # 加載數據引擎
            print(f"   📦 加載數據: {symbol} ({train_start} ~ {test_end})")
            engine = USDataEngineV2(symbol, train_start, train_end, test_end)
            engine.load()
            
            # 定義公式解析函數（與 times_us_v2 一致）
            def solve_formula(tokens, feat_data):
                """解析並執行因子公式"""
                import torch
                stack = []
                try:
                    for t in reversed(tokens):
                        if t < len(FEATURES):
                            stack.append(feat_data[t].unsqueeze(0))
                        else:
                            arity = OP_ARITY_MAP[t]
                            if len(stack) < arity: 
                                raise ValueError("Stack underflow")
                            args = [stack.pop() for _ in range(arity)]
                            func = OP_FUNC_MAP[t]
                            if arity == 2: 
                                res = func(args[0], args[1])
                            else: 
                                res = func(args[0])
                            if torch.isnan(res).any(): 
                                res = torch.nan_to_num(res)
                            stack.append(res)
                    
                    if len(stack) >= 1:
                        final = stack[-1]
                        if final.dim() == 2 and final.shape[0] == 1:
                            final = final.squeeze(0)
                        return final
                except Exception as e:
                    return None
                return None
            
            # 回測函數
            def backtest_formula(tokens, threshold=0.1, hysteresis=0.05):
                """回測單個公式"""
                factor = solve_formula(tokens, engine.feat_data)
                if factor is None:
                    return None
                
                split = engine.split_idx
                test_ctc_ret = engine.benchmark_ret[split:].cpu().numpy()
                test_otc_ret = engine.otc_ret[split:].cpu().numpy()
                test_cto_ret = engine.cto_ret[split:].cpu().numpy()
                test_dates = engine.dates[split:]
                
                # 計算信號
                f_all = factor.cpu().numpy()
                signal_all = np.tanh(f_all)
                pos_all = discretize_position(signal_all, threshold, hysteresis)
                
                pos_signal = pos_all[split:]
                signal = signal_all[split:]
                
                # 設置 effective_pos
                effective_pos = np.zeros(len(pos_signal), dtype=np.int32)
                effective_pos[0] = pos_all[split - 1] if split > 0 else 0
                effective_pos[1:] = pos_signal[:-1]
                
                # 計算收益
                daily_ret = np.zeros(len(pos_signal), dtype=np.float64)
                
                if effective_pos[0] != 0:
                    daily_ret[0] = effective_pos[0] * test_otc_ret[0] - COST_RATE
                
                for i in range(1, len(pos_signal)):
                    today_pos = effective_pos[i]
                    yesterday_pos = effective_pos[i - 1]
                    
                    if today_pos == 0:
                        if yesterday_pos != 0:
                            daily_ret[i] = yesterday_pos * test_cto_ret[i] - COST_RATE
                    else:
                        if yesterday_pos == 0:
                            daily_ret[i] = today_pos * test_otc_ret[i] - COST_RATE
                        elif yesterday_pos == today_pos:
                            daily_ret[i] = today_pos * test_ctc_ret[i]
                        else:
                            daily_ret[i] = yesterday_pos * test_cto_ret[i] + today_pos * test_otc_ret[i] - 2 * COST_RATE
                
                equity = (1 + daily_ret).cumprod()
                
                # 計算指標
                total_ret = equity[-1] - 1
                ann_ret = equity[-1] ** (252/len(equity)) - 1 if len(equity) > 0 else 0
                vol = np.std(daily_ret) * np.sqrt(252)
                sharpe = (ann_ret - 0.02) / (vol + 1e-6)
                dd = 1 - equity / np.maximum.accumulate(equity)
                max_dd = np.max(dd)
                
                return {
                    'equity': equity,
                    'position': effective_pos,
                    'signal': signal,
                    'dates': test_dates,
                    'metrics': {
                        'total_ret': total_ret,
                        'sharpe': sharpe,
                        'max_dd': max_dd,
                    }
                }
            
            # Ensemble 回測函數
            def backtest_ensemble(tokens_list, weights, threshold=0.1, hysteresis=0.05):
                """回測 Ensemble 策略（多公式加權平均）"""
                # 計算每個組件的因子值
                factor_list = []
                for tokens in tokens_list:
                    factor = solve_formula(tokens, engine.feat_data)
                    if factor is not None:
                        factor_list.append(factor)
                
                if not factor_list:
                    return None
                
                # 加權平均
                stacked = torch.stack(factor_list, dim=0)
                if weights and len(weights) == len(factor_list):
                    w = torch.tensor(weights, dtype=torch.float32, device=stacked.device).unsqueeze(1)
                    w = w / w.sum()
                    avg_factor = (stacked * w).sum(dim=0)
                else:
                    avg_factor = stacked.mean(dim=0)
                
                # 後續與單公式回測相同
                split = engine.split_idx
                test_ctc_ret = engine.benchmark_ret[split:].cpu().numpy()
                test_otc_ret = engine.otc_ret[split:].cpu().numpy()
                test_cto_ret = engine.cto_ret[split:].cpu().numpy()
                test_dates = engine.dates[split:]
                
                f_all = avg_factor.cpu().numpy()
                signal_all = np.tanh(f_all)
                pos_all = discretize_position(signal_all, threshold, hysteresis)
                
                pos_signal = pos_all[split:]
                signal = signal_all[split:]
                
                effective_pos = np.zeros(len(pos_signal), dtype=np.int32)
                effective_pos[0] = pos_all[split - 1] if split > 0 else 0
                effective_pos[1:] = pos_signal[:-1]
                
                daily_ret = np.zeros(len(pos_signal), dtype=np.float64)
                
                if effective_pos[0] != 0:
                    daily_ret[0] = effective_pos[0] * test_otc_ret[0] - COST_RATE
                
                for i in range(1, len(pos_signal)):
                    today_pos = effective_pos[i]
                    yesterday_pos = effective_pos[i - 1]
                    
                    if today_pos == 0:
                        if yesterday_pos != 0:
                            daily_ret[i] = yesterday_pos * test_cto_ret[i] - COST_RATE
                    else:
                        if yesterday_pos == 0:
                            daily_ret[i] = today_pos * test_otc_ret[i] - COST_RATE
                        elif yesterday_pos == today_pos:
                            daily_ret[i] = today_pos * test_ctc_ret[i]
                        else:
                            daily_ret[i] = yesterday_pos * test_cto_ret[i] + today_pos * test_otc_ret[i] - 2 * COST_RATE
                
                equity = (1 + daily_ret).cumprod()
                
                total_ret = equity[-1] - 1
                ann_ret = equity[-1] ** (252/len(equity)) - 1 if len(equity) > 0 else 0
                vol = np.std(daily_ret) * np.sqrt(252)
                sharpe = (ann_ret - 0.02) / (vol + 1e-6)
                dd = 1 - equity / np.maximum.accumulate(equity)
                max_dd = np.max(dd)
                
                return {
                    'equity': equity,
                    'position': effective_pos,
                    'signal': signal,
                    'dates': test_dates,
                    'metrics': {
                        'total_ret': total_ret,
                        'sharpe': sharpe,
                        'max_dd': max_dd,
                    }
                }
            
            # 回測所有公式（包括 Ensemble）
            backtest_results = []
            for f_info in all_top3_formulas:
                threshold = f_info.get('threshold', 0.1)
                hysteresis = f_info.get('hysteresis', 0.05)
                
                if f_info.get('is_ensemble', False):
                    # Ensemble 策略
                    tokens_list = f_info.get('tokens_list', [])
                    weights = f_info.get('weights', [])
                    result = backtest_ensemble(tokens_list, weights, threshold, hysteresis)
                else:
                    # 單公式策略
                    result = backtest_formula(f_info['tokens'], threshold, hysteresis)
                
                if result:
                    result['label'] = f_info['label']
                    result['formula'] = f_info['formula']
                    result['task_id'] = f_info['task_id']
                    result['rank'] = f_info['rank']
                    result['is_ensemble'] = f_info.get('is_ensemble', False)
                    result['score'] = f_info.get('score', 0)
                    backtest_results.append(result)
            
            if not backtest_results:
                print("   ⚠️ 所有公式回測失敗")
                return
            
            # Buy & Hold
            split = engine.split_idx
            test_ctc_ret = engine.benchmark_ret[split:].cpu().numpy()
            test_otc_ret = engine.otc_ret[split:].cpu().numpy()
            test_dates = engine.dates[split:]
            
            bh_ret = np.zeros(len(test_ctc_ret), dtype=np.float64)
            if len(bh_ret) > 0:
                bh_ret[0] = test_otc_ret[0]
                if len(bh_ret) > 1:
                    bh_ret[1:] = test_ctc_ret[1:]
            bench_equity = (1 + bh_ret).cumprod()
            bench_total_ret = bench_equity[-1] - 1
            
            # ==================== 繪製圖表 ====================
            # 按 Sharpe 排序，顯示 Top 公式
            backtest_results.sort(key=lambda x: x['metrics']['sharpe'], reverse=True)
            top_results = backtest_results[:min(9, len(backtest_results))]  # 最多顯示 9 個
            
            # 計算需要的子圖數量
            num_formulas = len(top_results)
            # 第1圖：淨值曲線 | 第2-N圖：每個公式的 Position+Signal | 最後1圖：回撤
            num_pos_plots = min(6, num_formulas)  # 最多顯示 6 個 Position 圖
            total_plots = 1 + num_pos_plots + 1
            height_ratios = [3] + [1.2] * num_pos_plots + [2]
            
            fig, axes = plt.subplots(total_plots, 1, figsize=(14, 4 + num_pos_plots * 2.5), 
                                     gridspec_kw={'height_ratios': height_ratios})
            
            # 第1圖：淨值曲線 - 所有公式對比
            ax1 = axes[0]
            
            for i, r in enumerate(top_results):
                m = r['metrics']
                label = f"T{r['task_id']}-{r['rank']}: {r['formula'][:20]}... | Sharpe {m['sharpe']:.2f}"
                ax1.plot(r['dates'], r['equity'], label=label, linewidth=2 if i < 3 else 1.2, 
                         color=colors[i % len(colors)], alpha=1.0 if i < 3 else 0.6)
            
            ax1.plot(test_dates, bench_equity, label=f'{symbol} Buy & Hold ({bench_total_ret:.1%})', 
                     alpha=0.5, linewidth=1.5, color='#A23B72', linestyle=':')
            
            ax1.set_title(f'{symbol} Multi-Task TOP3 Formulas Comparison | Test: {test_dates.iloc[0].date()} ~ {test_dates.iloc[-1].date()}', fontsize=14)
            ax1.set_ylabel('Cumulative Return')
            ax1.legend(loc='upper left', fontsize=7, ncol=2)
            ax1.grid(True, alpha=0.3)
            
            # 第2-N圖：各公式的 Position + Signal
            for idx, r in enumerate(top_results[:num_pos_plots]):
                ax = axes[1 + idx]
                
                # 繪製 Signal 曲線（實際公式結果）
                ax.plot(r['dates'], r['signal'], color=colors[idx % len(colors)], 
                        alpha=0.7, linewidth=1, label='Signal')
                
                # 繪製 Position 區域
                ax.fill_between(r['dates'], r['position'], step='mid', 
                                alpha=0.3, color=colors[idx % len(colors)], label='Position')
                
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                ax.axhline(y=1, color='green', linestyle=':', linewidth=0.5, alpha=0.5)
                ax.axhline(y=-1, color='red', linestyle=':', linewidth=0.5, alpha=0.5)
                
                # 標記閾值線
                threshold = 0.1
                ax.axhline(y=threshold, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
                ax.axhline(y=-threshold, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
                
                long_cnt = (r['position'] == 1).sum()
                short_cnt = (r['position'] == -1).sum()
                hold_cnt = (r['position'] == 0).sum()
                hold_pct = hold_cnt / len(r['position']) * 100
                
                formula_short = r['formula'][:35] + '...' if len(r['formula']) > 35 else r['formula']
                ax.set_title(f"T{r['task_id']}-Top{r['rank']}: {formula_short} | L:{long_cnt} S:{short_cnt} H:{hold_cnt}({hold_pct:.0f}%)", fontsize=9)
                ax.set_ylabel('Signal/Pos')
                ax.set_ylim(-1.5, 1.5)
                ax.grid(True, alpha=0.3)
            
            # 最後1圖：回撤對比
            ax_dd = axes[-1]
            for i, r in enumerate(top_results[:6]):
                dd = 1 - r['equity'] / np.maximum.accumulate(r['equity'])
                ax_dd.fill_between(r['dates'], -dd * 100, alpha=0.3, color=colors[i % len(colors)], 
                                   label=f"T{r['task_id']}-{r['rank']} MaxDD: {r['metrics']['max_dd']:.1%}")
            ax_dd.set_title('Drawdown Comparison (%)')
            ax_dd.set_ylabel('Drawdown %')
            ax_dd.set_xlabel('Date')
            ax_dd.legend(loc='lower left', fontsize=7, ncol=2)
            ax_dd.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存圖表
            output_file = os.path.join(output_folder, 'merged_comparison.png')
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"   📈 綜合對比圖表已保存: {output_file}")
            plt.close()
            
            # ==================== 保存回測結果排名 ====================
            print(f"\n   🏆 回測結果排名 (按 Sharpe):")
            print(f"   {'-'*80}")
            print(f"   {'#':<3} {'Task':>5} {'Rank':>5} {'Sharpe':>8} {'Return':>10} {'MaxDD':>8} {'Formula':<35}")
            print(f"   {'-'*80}")
            for i, r in enumerate(backtest_results[:10], 1):
                m = r['metrics']
                formula_short = r['formula'][:33] + '..' if len(r['formula']) > 35 else r['formula']
                print(f"   {i:<3} {r['task_id']:>5} {r['rank']:>5} {m['sharpe']:>8.2f} {m['total_ret']:>9.1%} {m['max_dd']:>7.1%} {formula_short}")
            print(f"   {'-'*80}")
            
            return
            
        except Exception as e:
            print(f"   ⚠️ 回測公式時發生錯誤: {e}")
            import traceback
            traceback.print_exc()
    
    # ==================== 回退：使用舊的統計圖（如果回測失敗）====================
    print("   📊 使用統計對比圖（回退模式）...")
    
    # 限制顯示數量
    top_strategies = strategies[:min(6, len(strategies))]
    
    # 創建圖表：4行子圖
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), gridspec_kw={'height_ratios': [3, 2, 2, 2]})
    
    # ==================== 第1圖：策略排名對比（橫向條形圖）====================
    ax1 = axes[0]
    
    # 準備數據
    labels = [f"#{i+1}: Task {s['task_id']}" for i, s in enumerate(top_strategies)]
    sharpes = [s['test_sharpe'] for s in top_strategies]
    returns = [s['test_return'] * 100 for s in top_strategies]  # 轉為百分比
    maxdds = [abs(s['test_maxdd']) * 100 for s in top_strategies]  # 轉為正數百分比
    
    x = np.arange(len(labels))
    width = 0.25
    
    bars1 = ax1.barh(x - width, sharpes, width, label='Sharpe Ratio', color=colors[0], alpha=0.8)
    bars2 = ax1.barh(x, returns, width, label='Return (%)', color=colors[1], alpha=0.8)
    bars3 = ax1.barh(x + width, [-d for d in maxdds], width, label='MaxDD (%)', color=colors[3], alpha=0.8)
    
    ax1.set_yticks(x)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Value')
    ax1.set_title(f'{symbol} Multi-Task Training Results - Task Comparison', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # 添加數值標籤
    for i, (sh, ret, dd) in enumerate(zip(sharpes, returns, maxdds)):
        ax1.text(sh + 0.05, i - width, f'{sh:.2f}', va='center', fontsize=8)
        ax1.text(ret + 0.5, i, f'{ret:.1f}%', va='center', fontsize=8)
        ax1.text(-dd - 0.5, i + width, f'-{dd:.1f}%', va='center', fontsize=8, ha='right')
    
    # ==================== 第2圖：公式複雜度 vs 性能 散點圖 ====================
    ax2 = axes[1]
    
    for i, s in enumerate(top_strategies):
        formula_len = len(s.get('tokens', []))
        sharpe = s['test_sharpe']
        ret = s['test_return'] * 100
        
        ax2.scatter(formula_len, sharpe, s=150, c=colors[i % len(colors)], 
                   alpha=0.8, edgecolors='black', linewidth=1,
                   label=f"Task {s['task_id']}")
        ax2.annotate(f"T{s['task_id']}\nRet:{ret:.1f}%", (formula_len, sharpe), 
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    
    ax2.set_xlabel('Formula Length (tokens)')
    ax2.set_ylabel('Test Sharpe Ratio')
    ax2.set_title('Formula Complexity vs Performance')
    ax2.legend(loc='upper right', ncol=3)
    ax2.grid(True, alpha=0.3)
    
    # ==================== 第3圖：訓練分數 vs 測試 Sharpe（過擬合檢測）====================
    ax3 = axes[2]
    
    train_scores = [s['train_score'] for s in top_strategies]
    test_sharpes = [s['test_sharpe'] for s in top_strategies]
    
    for i, (tr, te) in enumerate(zip(train_scores, test_sharpes)):
        ax3.scatter(tr, te, s=150, c=colors[i % len(colors)], 
                   alpha=0.8, edgecolors='black', linewidth=1,
                   label=f"Task {top_strategies[i]['task_id']}")
    
    # 添加對角線（理想情況：訓練=測試）
    if train_scores and test_sharpes:
        max_val = max(max(train_scores), max(test_sharpes))
        min_val = min(min(train_scores), min(test_sharpes))
        ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='Ideal (no overfit)')
    
    ax3.set_xlabel('Train Score')
    ax3.set_ylabel('Test Sharpe')
    ax3.set_title('Overfitting Detection: Train vs Test Performance')
    ax3.legend(loc='upper left', ncol=3)
    ax3.grid(True, alpha=0.3)
    
    # ==================== 第4圖：Top 公式詳情表格 ====================
    ax4 = axes[3]
    ax4.axis('off')
    
    # 創建表格數據
    table_data = []
    headers = ['Task', 'Sharpe', 'Return', 'MaxDD', 'Train', 'Formula']
    
    for i, s in enumerate(top_strategies, 1):
        formula = s['formula']
        if len(formula) > 40:
            formula = formula[:37] + "..."
        table_data.append([
            f"Task {s['task_id']}",
            f"{s['test_sharpe']:.2f}",
            f"{s['test_return']:.1%}",
            f"{s['test_maxdd']:.1%}",
            f"{s['train_score']:.2f}",
            formula
        ])
    
    if table_data:
        table = ax4.table(cellText=table_data, colLabels=headers,
                          loc='center', cellLoc='center',
                          colColours=['#E8E8E8'] * len(headers))
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # 設置表頭樣式
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4A90D9')
            table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        # 設置排名列顏色
        for i in range(1, len(table_data) + 1):
            table[(i, 0)].set_facecolor(colors[(i-1) % len(colors)])
            table[(i, 0)].set_text_props(color='white', fontweight='bold')
    
    ax4.set_title(f'Top {len(top_strategies)} Tasks Summary', fontsize=12, pad=20)
    
    plt.tight_layout()
    
    # 保存圖表
    output_file = os.path.join(output_folder, 'merged_comparison.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   📈 綜合對比圖表已保存: {output_file}")
    plt.close()


def plot_strategy_analysis(strategies, symbol, output_folder, all_top3_formulas=None):
    """
    繪製策略分析圖表
    
    包含：
    1. Sharpe 對比（各任務及 Ensemble）
    2. 複雜度散點圖（Score vs 公式長度）
    3. 過擬合檢測（Train Score vs Test Sharpe）
    4. 收益風險分佈（Return vs MaxDD）
    """
    plt = get_plt()
    
    if len(strategies) < 1:
        print("   ⚠️ 策略數量不足，無法繪製分析圖")
        return
    
    plt.style.use('bmh')
    
    colors = ['#2E86AB', '#28A745', '#F39C12', '#E74C3C', '#9B59B6', '#1ABC9C',
              '#3498DB', '#2ECC71', '#F1C40F', '#E67E22']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # ==================== 第1圖：Sharpe 對比條形圖 ====================
    ax1 = axes[0, 0]
    
    # 收集各任務的 Sharpe（包括 Ensemble）
    task_sharpes = []
    for s in strategies[:10]:
        task_id = s['task_id']
        task_sharpes.append({
            'label': f"T{task_id}",
            'sharpe': s.get('test_sharpe', 0),
            'type': 'single',
            'task_id': task_id,
        })
    
    # 從 all_top3_formulas 中提取 Ensemble 的 score（作為近似 Sharpe）
    if all_top3_formulas:
        for f in all_top3_formulas:
            if f.get('is_ensemble', False):
                task_id = f['task_id']
                # 嘗試找到對應任務的 ensemble test metrics
                # 這裡用 train_score 作為參考（實際應該用回測結果）
                task_sharpes.append({
                    'label': f"T{task_id}E",
                    'sharpe': f.get('score', 0),
                    'type': 'ensemble',
                    'task_id': task_id,
                })
    
    if task_sharpes:
        labels = [t['label'] for t in task_sharpes]
        sharpes = [t['sharpe'] for t in task_sharpes]
        bar_colors = ['#2E86AB' if t['type'] == 'single' else '#28A745' for t in task_sharpes]
        
        bars = ax1.bar(labels, sharpes, color=bar_colors, alpha=0.8, edgecolor='white')
        ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Sharpe=1.0')
        ax1.axhline(y=1.5, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Sharpe=1.5')
        
        # 在條形上標註數值
        for bar, sharpe in zip(bars, sharpes):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    f'{sharpe:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax1.set_title(f'{symbol} Sharpe Comparison (Blue=Single, Green=Ensemble)', fontsize=12)
        ax1.set_ylabel('Sharpe Ratio')
        ax1.set_xlabel('Task')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3, axis='y')
    
    # ==================== 第2圖：複雜度散點圖 ====================
    ax2 = axes[0, 1]
    
    if all_top3_formulas:
        scores = []
        complexities = []
        labels_scatter = []
        colors_scatter = []
        
        for f in all_top3_formulas:
            if not f.get('is_ensemble', False):
                formula = f.get('formula', '')
                # 計算複雜度（公式長度 / 運算符數量）
                complexity = len(formula) if formula else 0
                score = f.get('score', 0)
                
                scores.append(score)
                complexities.append(complexity)
                labels_scatter.append(f['label'])
                
                # 根據任務 ID 分配顏色
                task_idx = (f['task_id'] - 1) % len(colors)
                colors_scatter.append(colors[task_idx])
        
        if scores and complexities:
            scatter = ax2.scatter(complexities, scores, c=colors_scatter, s=80, alpha=0.7, edgecolors='white')
            
            # 添加趨勢線
            if len(scores) > 2:
                z = np.polyfit(complexities, scores, 1)
                p = np.poly1d(z)
                x_line = np.linspace(min(complexities), max(complexities), 100)
                ax2.plot(x_line, p(x_line), 'r--', alpha=0.5, label=f'Trend (slope={z[0]:.3f})')
            
            ax2.set_title('Complexity vs Score (Occam\'s Razor Check)', fontsize=12)
            ax2.set_xlabel('Formula Complexity (length)')
            ax2.set_ylabel('Training Score')
            ax2.legend(loc='upper right', fontsize=8)
            ax2.grid(True, alpha=0.3)
    
    # ==================== 第3圖：過擬合檢測 ====================
    ax3 = axes[1, 0]
    
    train_scores = []
    test_sharpes = []
    task_labels = []
    
    for s in strategies[:15]:
        train_scores.append(s.get('train_score', 0))
        test_sharpes.append(s.get('test_sharpe', 0))
        task_labels.append(f"T{s['task_id']}")
    
    if train_scores and test_sharpes:
        scatter = ax3.scatter(train_scores, test_sharpes, 
                             c=[colors[i % len(colors)] for i in range(len(train_scores))],
                             s=100, alpha=0.7, edgecolors='white')
        
        # 對角線（理想情況：train_score ≈ test_sharpe）
        max_val = max(max(train_scores), max(test_sharpes))
        min_val = min(min(train_scores), min(test_sharpes))
        ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='Ideal (no overfit)')
        
        # 標註點
        for i, label in enumerate(task_labels):
            ax3.annotate(label, (train_scores[i], test_sharpes[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 計算過擬合程度
        if len(train_scores) > 1:
            correlation = np.corrcoef(train_scores, test_sharpes)[0, 1]
            ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                    transform=ax3.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax3.set_title('Overfit Detection (Train Score vs Test Sharpe)', fontsize=12)
        ax3.set_xlabel('Training Score')
        ax3.set_ylabel('Test Sharpe')
        ax3.legend(loc='lower right', fontsize=8)
        ax3.grid(True, alpha=0.3)
    
    # ==================== 第4圖：收益風險分佈 ====================
    ax4 = axes[1, 1]
    
    returns = []
    maxdds = []
    task_labels_risk = []
    
    for s in strategies[:15]:
        returns.append(s.get('test_return', 0) * 100)  # 轉為百分比
        maxdds.append(abs(s.get('test_maxdd', 0)) * 100)  # 轉為正數百分比
        task_labels_risk.append(f"T{s['task_id']}")
    
    if returns and maxdds:
        scatter = ax4.scatter(maxdds, returns, 
                             c=[colors[i % len(colors)] for i in range(len(returns))],
                             s=100, alpha=0.7, edgecolors='white')
        
        # 標註點
        for i, label in enumerate(task_labels_risk):
            ax4.annotate(label, (maxdds[i], returns[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 計算 Calmar Ratio 等高線
        calmar_levels = [0.5, 1.0, 1.5, 2.0]
        max_dd_range = np.linspace(1, max(maxdds) * 1.1, 100)
        for calmar in calmar_levels:
            ret_line = calmar * max_dd_range
            ax4.plot(max_dd_range, ret_line, '--', alpha=0.3, 
                    label=f'Calmar={calmar}' if calmar == 1.0 else None)
        
        ax4.set_title('Risk-Return Profile (Return vs MaxDD)', fontsize=12)
        ax4.set_xlabel('Max Drawdown (%)')
        ax4.set_ylabel('Total Return (%)')
        ax4.legend(loc='upper left', fontsize=8)
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'{symbol} Strategy Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 保存圖表
    output_file = os.path.join(output_folder, 'strategy_analysis.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   📈 策略分析圖表已保存: {output_file}")
    plt.close()


def collect_best_strategies(symbol):
    """
    收集並比較多次任務訓練產生的最佳策略（舊版兼容函數）
    """
    strategies = collect_recent_strategies(symbol, minutes=60)
    
    if not strategies:
        print("   沒有找到策略文件")
        return
    
    print("\n" + "-" * 90)
    print(f"🏆 Top {min(5, len(strategies))} 策略 (按測試 Sharpe 排序):")
    print("-" * 90)
    print(f"  {'#':<3} {'Sharpe':>8} {'Return':>10} {'MaxDD':>8} {'Train':>8} {'Folder':<40}")
    print("-" * 90)
    
    for i, s in enumerate(strategies[:5], 1):
        sharpe = s.get('test_sharpe', 0)
        ret = s.get('test_return', 0)
        maxdd = s.get('test_maxdd', 0)
        train = s.get('train_score', 0)
        folder = s['folder']
        if len(folder) > 38:
            folder = "..." + folder[-35:]
        
        print(f"  {i:<3} {sharpe:>8.2f} {ret:>9.1%} {maxdd:>7.1%} {train:>8.2f} {folder}")
    
    print("-" * 90)
    
    if strategies:
        best = strategies[0]
        print(f"\n🥇 最佳策略:")
        print(f"   文件: {best['filepath']}")
        print(f"   公式: {best['formula'][:60]}..." if len(best['formula']) > 60 else f"   公式: {best['formula']}")


def mode_backtest():
    """回測模式"""
    print("\n" + "=" * 70)
    print("📊 回測策略")
    print("=" * 70)
    
    # 選擇策略
    strategy_path = select_strategy()
    if not strategy_path:
        input("\n按 Enter 鍵繼續...")
        return
    
    # 讀取策略的原始閾值和版本
    try:
        with open(strategy_path, 'r') as f:
            strategy_data = json.load(f)
        strategy_threshold = strategy_data.get('signal_threshold', DEFAULT_SIGNAL_THRESHOLD)
        strategy_version = strategy_data.get('version', '1.0')
        is_v2 = strategy_version.startswith('2')
    except:
        strategy_threshold = DEFAULT_SIGNAL_THRESHOLD
        is_v2 = False
    
    print(f"\n📋 策略版本: {'V2' if is_v2 else 'V1'}")
    
    # 獲取參數
    tickers = get_input("請輸入要測試的股票代碼 (多個用逗號分隔, 例如: SPY,QQQ,AAPL)", default="SPY")
    
    # 時間段選擇
    print("\n📅 時間段設置:")
    print("   1. 使用預設週期 (如: 6mo, 1y, 2y)")
    print("   2. 自定義時間段 (指定開始和結束日期)")
    use_custom_dates = get_input("請選擇 (1/2)", default="1")
    
    period = None
    start_date = None
    end_date = None
    
    if use_custom_dates == "2":
        # 自定義時間段
        start_date = get_input("請輸入開始日期 (YYYY-MM-DD)", default="2024-01-01")
        end_date = get_input("請輸入結束日期 (YYYY-MM-DD)", default=datetime.now().strftime('%Y-%m-%d'))
        period_display = f"{start_date} ~ {end_date}"
    else:
        # 預設週期
        period = get_input("請輸入回測週期 (例如: 6mo, 1y, 2y, ytd)", default="1y")
        period_display = period
    
    capital = get_input("請輸入初始資金 (USD)", default="100000")
    
    # 信號閾值（允許覆蓋）
    print(f"\n📊 策略原始閾值: ±{strategy_threshold}")
    print("   信號閾值設置 (|signal| < threshold → 觀望/不交易)")
    threshold_input = get_input(f"請輸入信號閾值 (留空使用策略原始值)", default="", required=False)
    override_threshold = float(threshold_input) if threshold_input else None
    
    # 滯後閾值（允許覆蓋）
    strategy_hysteresis = strategy_data.get('hysteresis', DEFAULT_HYSTERESIS)
    print(f"\n📊 策略原始滯後值: {strategy_hysteresis}")
    print("   滯後閾值設置 (平倉閾值 = threshold - hysteresis)")
    hysteresis_input = get_input(f"請輸入滯後值 (留空使用策略原始值)", default="", required=False)
    override_hysteresis = float(hysteresis_input) if hysteresis_input else None
    
    export_logs = get_yes_no("是否導出交易記錄 (CSV)?", 'n')
    generate_charts = get_yes_no("是否生成圖表?", 'y')
    
    # 確認
    backtest_threshold = override_threshold if override_threshold is not None else strategy_threshold
    backtest_hysteresis = override_hysteresis if override_hysteresis is not None else strategy_hysteresis
    print("\n" + "-" * 70)
    print("📝 回測配置確認 (V2 回測器):")
    print(f"   策略文件    : {os.path.basename(strategy_path)}")
    print(f"   策略版本    : {'V2' if is_v2 else 'V1'}")
    print(f"   信號閾值    : ±{backtest_threshold}" + (" (覆蓋)" if override_threshold is not None else ""))
    print(f"   滯後值      : {backtest_hysteresis}" + (" (覆蓋)" if override_hysteresis is not None else ""))
    print(f"   測試標的    : {tickers}")
    print(f"   回測時間段  : {period_display}")
    print(f"   初始資金    : ${float(capital):,.2f}")
    print(f"   導出記錄    : {'是' if export_logs else '否'}")
    print(f"   生成圖表    : {'是' if generate_charts else '否'}")
    print(f"   回測邏輯    : 持有直到信號改變 (不強制每天換倉)")
    print("-" * 70)
    
    if not get_yes_no("確認開始回測?", 'y'):
        print("已取消")
        return
    
    # 執行回測 (使用 V2 回測器 - 正確的持倉邏輯)
    print("\n📊 開始回測 (V2 - 持有直到信號改變)...")
    cmd = [
        sys.executable, 'backtest_strategy_v2.py',
        '--strategy', strategy_path,
        '--tickers', tickers.upper(),
        '--capital', capital,
    ]
    
    # 時間段參數
    if start_date and end_date:
        cmd.extend(['--start', start_date, '--end', end_date])
    else:
        cmd.extend(['--period', period])
    
    if override_threshold is not None:
        cmd.extend(['--threshold', str(override_threshold)])
    
    if override_hysteresis is not None:
        cmd.extend(['--hysteresis', str(override_hysteresis)])
    
    if export_logs:
        cmd.append('--export')
    if generate_charts:
        cmd.append('--plot')
    
    subprocess.run(cmd)
    
    input("\n按 Enter 鍵繼續...")


def mode_signal():
    """信號生成模式"""
    print("\n" + "=" * 70)
    print("📡 生成交易信號")
    print("=" * 70)
    
    # 選擇策略
    strategy_path = select_strategy()
    if not strategy_path:
        input("\n按 Enter 鍵繼續...")
        return
    
    # 讀取策略的原始閾值
    try:
        with open(strategy_path, 'r') as f:
            strategy_data = json.load(f)
        strategy_threshold = strategy_data.get('signal_threshold', DEFAULT_SIGNAL_THRESHOLD)
    except:
        strategy_threshold = DEFAULT_SIGNAL_THRESHOLD
    
    # 獲取參數
    symbols = get_input(
        "請輸入要掃描的股票代碼 (多個用逗號分隔)", 
        default="SPY,QQQ,AAPL,MSFT,NVDA,TSLA,AMD,GOOGL,META"
    )
    
    # 信號閾值（允許覆蓋）
    print(f"\n📊 策略原始閾值: ±{strategy_threshold}")
    print("   信號閾值設置 (|signal| < threshold → HOLD)")
    threshold_input = get_input(f"請輸入信號閾值 (留空使用策略原始值)", default="", required=False)
    override_threshold = float(threshold_input) if threshold_input else None
    
    monitor_enabled = get_yes_no("是否啟用監控模式 (持續更新)?", 'n')
    interval = 60
    if monitor_enabled:
        interval = get_input("請輸入更新間隔 (秒)", default="60")
    
    # 執行
    signal_threshold = override_threshold if override_threshold is not None else strategy_threshold
    print(f"\n📡 生成信號... (閾值: ±{signal_threshold})")
    cmd = [
        sys.executable, 'signal_generator.py',
        '--strategy', strategy_path,
        '--symbols', symbols.upper(),
    ]
    
    if override_threshold is not None:
        cmd.extend(['--threshold', str(override_threshold)])
    
    if monitor_enabled:
        cmd.extend(['--monitor', '--interval', str(interval)])
    
    subprocess.run(cmd)
    
    if not monitor_enabled:
        input("\n按 Enter 鍵繼續...")


def mode_list():
    """列出策略和數據"""
    print("\n" + "=" * 70)
    print("📋 已有策略和數據")
    print("=" * 70)
    
    strategies = list_strategies()
    
    if strategies:
        print("\n📜 策略文件 (按時間排序，最新在前):")
        print("-" * 100)
        print(f"{'路徑':<40} {'Ver':<4} {'代碼':<8} {'閾值':<7} {'Score':<8} {'Sharpe':<8} {'MaxDD':<8}")
        print("-" * 100)
        
        for s in strategies:
            score = s.get('train_score', 'N/A')
            if isinstance(score, float):
                score = f"{score:.2f}"
            
            sharpe = s.get('sharpe', 'N/A')
            if isinstance(sharpe, float):
                sharpe = f"{sharpe:.2f}"
            
            max_dd = s.get('max_dd', 'N/A')
            if isinstance(max_dd, float):
                max_dd = f"{max_dd:.1%}"
            
            thresh = s.get('signal_threshold', DEFAULT_SIGNAL_THRESHOLD)
            thresh_str = f"±{thresh}"
            # 截斷過長的路徑
            filepath = s['file']
            if len(filepath) > 38:
                filepath = "..." + filepath[-35:]
            
            print(f"{filepath:<40} {s['version']:<4} {s['symbol']:<8} {thresh_str:<7} {score:<8} {sharpe:<8} {max_dd:<8}")
        
        print("-" * 100)
        
        # 顯示因子詳情
        print("\n📜 公式詳情:")
        for s in strategies:
            formula = s.get('formula', 'N/A')
            thresh = s.get('signal_threshold', DEFAULT_SIGNAL_THRESHOLD)
            version = s['version']
            features = s.get('features', [])
            feature_count = len(features)
            if len(formula) > 60:
                formula = formula[:57] + "..."
            print(f"  [{version}] {s['symbol']} (±{thresh}, {feature_count}F): {formula}")
    else:
        print("\n  ❌ 沒有找到任何策略文件")
        print("     請先使用選項 1 或 2 訓練一個策略")
    
    # 統計子目錄
    print("\n📁 輸出子目錄:")
    print("-" * 60)
    subdirs = []
    if os.path.exists(OUTPUT_DIR):
        for item in os.listdir(OUTPUT_DIR):
            item_path = os.path.join(OUTPUT_DIR, item)
            if os.path.isdir(item_path):
                # 計算子目錄中的文件數
                file_count = len([f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))])
                # 判斷是否為 V2
                is_v2 = '_v2' in item.lower() or '_WF' in item
                subdirs.append((item, file_count, is_v2))
    
    if subdirs:
        for name, count, is_v2 in sorted(subdirs, reverse=True)[:10]:  # 最多顯示10個
            version_tag = "[V2]" if is_v2 else "[V1]"
            print(f"  📂 {version_tag} {name:<45} ({count} 文件)")
        if len(subdirs) > 10:
            print(f"  ... 還有 {len(subdirs) - 10} 個子目錄")
    else:
        print("  (無子目錄)")
    
    # 數據緩存
    print("\n📁 數據緩存:")
    print("-" * 60)
    caches = []
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            if f.startswith('data_cache_') and f.endswith('.parquet'):
                filepath = os.path.join(OUTPUT_DIR, f)
                size = os.path.getsize(filepath)
                print(f"  {f:<45} {size/1024:.1f} KB")
                caches.append(f)
    
    if not caches:
        print("  (無)")
    
    # 統計
    if os.path.exists(OUTPUT_DIR):
        total_files = 0
        total_dirs = 0
        png_count = 0
        csv_count = 0
        json_count = 0
        
        for root, dirs, files in os.walk(OUTPUT_DIR):
            total_dirs += len(dirs)
            total_files += len(files)
            png_count += len([f for f in files if f.endswith('.png')])
            csv_count += len([f for f in files if f.endswith('.csv')])
            json_count += len([f for f in files if f.endswith('.json')])
        
        v1_count = sum(1 for s in strategies if s['version'] == 'V1')
        v2_count = sum(1 for s in strategies if s['version'] == 'V2')
        
        print(f"\n📊 輸出統計:")
        print(f"  策略文件: {len(strategies)} 個 (V1: {v1_count}, V2: {v2_count})")
        print(f"  圖表文件: {png_count} 個")
        print(f"  日誌文件: {csv_count} 個")
        print(f"  JSON文件: {json_count} 個")
        print(f"  子目錄  : {total_dirs} 個")
        print(f"  總計    : {total_files} 個文件在 {OUTPUT_DIR}/")
    
    input("\n按 Enter 鍵繼續...")


def mode_clean():
    """清理輸出文件"""
    import shutil
    
    print("\n" + "=" * 70)
    print("🧹 清理輸出文件")
    print("=" * 70)
    
    if not os.path.exists(OUTPUT_DIR):
        print("\n  輸出目錄不存在")
        input("\n按 Enter 鍵繼續...")
        return
    
    # 統計文件和目錄
    root_files = []
    subdirs = []
    cache_files = []
    
    for item in os.listdir(OUTPUT_DIR):
        item_path = os.path.join(OUTPUT_DIR, item)
        if os.path.isdir(item_path):
            subdirs.append(item)
        elif os.path.isfile(item_path):
            root_files.append(item)
            if item.endswith('.parquet'):
                cache_files.append(item)
    
    if not root_files and not subdirs:
        print("\n  輸出目錄已經是空的")
        input("\n按 Enter 鍵繼續...")
        return
    
    # 分類
    strategy_files = [f for f in root_files if f.endswith('_best_strategy.json')]
    chart_files = [f for f in root_files if f.endswith('.png')]
    log_files = [f for f in root_files if f.endswith('.csv')]
    
    print(f"\n📁 當前輸出統計:")
    print(f"  子目錄    : {len(subdirs)} 個 (包含訓練/回測結果)")
    print(f"  根目錄文件: {len(root_files)} 個")
    print(f"    - 策略  : {len(strategy_files)} 個")
    print(f"    - 緩存  : {len(cache_files)} 個")
    print(f"    - 圖表  : {len(chart_files)} 個")
    print(f"    - 日誌  : {len(log_files)} 個")
    
    print("\n請選擇要清理的內容:")
    print("  1. 清理根目錄的圖表和日誌 (保留策略、緩存和子目錄)")
    print("  2. 清理數據緩存文件 (.parquet)")
    print("  3. 清理所有子目錄 ⚠️ (刪除所有訓練/回測結果)")
    print("  4. 清理根目錄所有文件 (保留子目錄)")
    print("  5. 清理全部 ⚠️ (刪除所有文件和子目錄)")
    print("  0. 取消")
    
    choice = input("\n請選擇 (0-5): ").strip()
    
    files_to_delete = []
    dirs_to_delete = []
    
    if choice == '1':
        files_to_delete = chart_files + log_files
    elif choice == '2':
        files_to_delete = cache_files
    elif choice == '3':
        dirs_to_delete = subdirs
    elif choice == '4':
        files_to_delete = root_files
    elif choice == '5':
        files_to_delete = root_files
        dirs_to_delete = subdirs
    else:
        print("已取消")
        input("\n按 Enter 鍵繼續...")
        return
    
    if not files_to_delete and not dirs_to_delete:
        print("沒有要刪除的項目")
        input("\n按 Enter 鍵繼續...")
        return
    
    print(f"\n將刪除:")
    if files_to_delete:
        print(f"  - {len(files_to_delete)} 個文件")
    if dirs_to_delete:
        print(f"  - {len(dirs_to_delete)} 個子目錄 (及其所有內容)")
    
    if not get_yes_no("確認刪除?", 'n'):
        print("已取消")
        input("\n按 Enter 鍵繼續...")
        return
    
    deleted_files = 0
    deleted_dirs = 0
    
    for f in files_to_delete:
        try:
            os.remove(os.path.join(OUTPUT_DIR, f))
            deleted_files += 1
        except Exception as e:
            print(f"  ⚠️ 無法刪除文件 {f}: {e}")
    
    for d in dirs_to_delete:
        try:
            shutil.rmtree(os.path.join(OUTPUT_DIR, d))
            deleted_dirs += 1
        except Exception as e:
            print(f"  ⚠️ 無法刪除目錄 {d}: {e}")
    
    print(f"\n✅ 已刪除 {deleted_files} 個文件, {deleted_dirs} 個目錄")
    input("\n按 Enter 鍵繼續...")


def mode_help():
    """顯示幫助"""
    print("\n" + "=" * 70)
    print("❓ 幫助說明")
    print("=" * 70)
    
    print("""
📖 功能說明:

  1. 🚀 訓練新策略 (防過擬合版 + 多任務)
     - 使用 times_us_v2.py
     - 11 個因子: 含 ATR, RSI, CLV, RS (相對強度), MOM, VIX
     - Walk-Forward 滾動驗證 - 確保因子穩健
     - 多目標獎勵: Sortino + Sharpe + Return + Alpha - MaxDD - Complexity
     - Ensemble 策略 - Top-K 公式信號平均
     - 複雜度懲罰 - 奧卡姆剃刀原則
     - 多次任務訓練 - 不同種子探索更廣

  2. 📊 回測策略
     - 正確的持倉邏輯（持有直到信號改變）
     - 輸出詳細的交易記錄

  3. 📡 生成交易信號
     - 使用策略對當前市場生成買/賣信號
     - 支持監控模式

  4. 📋 查看已有策略
     - 列出所有策略文件

  5. 🧹 清理輸出文件
     - 清理圖表、日誌、緩存等

  6. ❓ 幫助說明
     - 查看本說明

📊 因子說明 (11 個因子):

  基礎因子 (5個):
    RET     - 日收益率
    RET5    - 5日收益率
    VOL_CHG - 成交量變化
    V_RET   - 量價收益
    TREND   - 趨勢（相對 MA60）

  新增因子 (6個):
    ATR   - Average True Range (平均真實波幅)
            衡量市場波動率，高 ATR = 高波動

    RSI   - Relative Strength Index (相對強弱指數)
            衡量超買/超賣，RSI > 70 超買，RSI < 30 超賣

    CLV   - Close Location Value (收盤位置值)
            收盤價在當日範圍的位置，[-1, 1]

    RS    - Relative Strength (相對強度)
            相對於基準 (SPY) 的超額收益

    MOM   - Momentum (動量)
            20日價格變化率

    VIX   - VIX 恐慌指數 🆕
            市場情緒指標，VIX > 30 恐慌，VIX < 15 樂觀

📊 Walk-Forward 滾動驗證:

  傳統方式:
    Train: 2015-2023 → Test: 2024-2025
    問題: 可能只是在特定時期有效

  Walk-Forward 方式:
    Window 1: Train 2018-2021 → Val 2021-2022 ✓
    Window 2: Train 2019-2022 → Val 2022-2023 ✓
    Window 3: Train 2020-2023 → Val 2023-2024 ✓
    只有所有窗口都表現穩定的因子才被保留

📊 多目標獎勵機制:

  reward = 0.25 × Sortino      (風險調整收益)
         + 0.20 × Sharpe       (夏普比率)
         + 0.10 × Return       (絕對收益)
         + 0.10 × Alpha_SPY    (超越大盤 SPY)
         + 0.05 × Alpha_Target (超越目標 Buy & Hold)
         - 0.20 × MaxDD        (回撤懲罰) ⬆️
         - 0.10 × Complexity   (複雜度懲罰) ⬇️
  
  Alpha_SPY    = 策略收益 - SPY 收益 (超越大盤)
  Alpha_Target = 策略收益 - 目標標的 Buy & Hold 收益 (真正的 Alpha!)

🔄 多次任務訓練 (V2 新功能):

  問題: 單次訓練可能陷入局部最優解
        不同隨機種子會探索到不同的公式空間
  
  解決方案: 串行執行多次訓練任務，每次用不同種子
  
  效果:
    ┌────────────────────────────────────────────────────────────┐
    │ 單次任務 (Batch=512):  ~5s/it × 300 = 25 分鐘              │
    │ 4 次任務 (Batch=512):  ~5s/it × 300 × 4 = 100 分鐘        │
    │                        探索了 4× 的公式空間！               │
    │                        GPU 100% 利用（串行執行）           │
    └────────────────────────────────────────────────────────────┘
  
  優點:
    1. GPU 100% - 串行執行，無資源競爭
    2. 探索更廣 - 每次任務用不同隨機種子
    3. 自動整合 - 訓練後自動選出最佳策略
    4. 商業做法 - Ensemble over runs
  
  建議任務數:
    - 2~6 次（視時間預算而定）
    - 更多任務 = 更廣探索 = 更好結果

💡 使用建議:

  1. 標的: 建議用 SPY/QQQ 等大盤 ETF 尋找 Alpha
          個股 (NVDA/TSLA) 更難跑贏 Buy & Hold
  2. 回撤: 如果策略回報略低於 Buy & Hold，但回撤更小，
          這其實是成功的策略 (可以上槓桿)
  3. 任務數: 建議啟用多次任務訓練（2~6 次）
  4. 整合: 多次訓練後會自動整合結果到 MERGED 文件夾

📁 輸出目錄結構:

  output/
  ├── SPY_v2_MERGED_20260127_120000/  # 多次任務整合輸出
  │   ├── merged_strategies.json      # 整合策略（Top 20 公式）
  │   ├── best_strategy_v2.json       # 最佳策略
  │   ├── merged_comparison.png       # 綜合對比圖表（含各任務 Ensemble）
  │   ├── strategy_analysis.png       # 策略分析（Sharpe/複雜度/過擬合）
  │   └── task_1/ ~ task_N/           # 各任務原始輸出
  └── backtest_*/                     # 回測輸出
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
            mode_train_v2()
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
