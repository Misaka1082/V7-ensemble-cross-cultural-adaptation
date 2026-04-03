#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4.1.9.final 完整运行脚本
=========================
跨文化适应预测模型 - 完整流程

运行顺序：
  Step 1: 生成保留交互效应的10万样本数据（如果尚未生成）
  Step 2: 训练V7 Ultimate模型（最佳模型，R²=0.728）
  Step 3: 过拟合检验（嵌套CV验证Stacking）
  Step 4: 解释性分析（SHAP + 聚类 + 心理学报告）
  Step 5: 生成103个样本的个性化报告

用法：
  python run_all.py                    # 运行全部步骤
  python run_all.py --step 2           # 只运行第2步
  python run_all.py --step 4 5         # 只运行第4、5步
  python run_all.py --skip-data        # 跳过数据生成（数据已存在）
  python run_all.py --quick            # 快速模式（跳过耗时步骤）

各步骤说明：
  Step 1 (generate_data):     约 5-10 分钟
  Step 2 (train_v7):          约 60-120 分钟（含贝叶斯优化）
  Step 3 (check_overfitting): 约 30-60 分钟（嵌套CV）
  Step 4 (interpretability):  约 5-10 分钟
  Step 5 (individual_reports):约 10-20 分钟（生成103页PDF）
"""

import subprocess
import sys
import os
import time
import argparse
from pathlib import Path

# 设置工作目录为脚本所在目录
SCRIPT_DIR = Path(__file__).parent
os.chdir(SCRIPT_DIR)

def run_step(name, script, description, skip=False):
    """运行单个步骤"""
    if skip:
        print(f"\n{'='*70}")
        print(f"[跳过] {name}: {description}")
        print(f"{'='*70}")
        return True

    print(f"\n{'='*70}")
    print(f"[开始] {name}: {description}")
    print(f"{'='*70}")
    start = time.time()

    result = subprocess.run(
        [sys.executable, script],
        cwd=str(SCRIPT_DIR),
        capture_output=False,
    )

    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"\n[完成] {name} - 用时 {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
        return True
    else:
        print(f"\n[失败] {name} - 返回码: {result.returncode}")
        return False


def check_data_exists():
    """检查必要数据文件是否存在"""
    required = [
        'data/processed/real_data_103.xlsx',
        'data/processed/improved_100k.csv',
        'data/processed/interaction_preserved_100k.csv',
    ]
    missing = []
    for f in required:
        if not (SCRIPT_DIR / f).exists():
            missing.append(f)
    return missing


def main():
    parser = argparse.ArgumentParser(description='4.1.9.final 完整运行脚本')
    parser.add_argument('--step', nargs='+', type=int,
                       help='只运行指定步骤（1-5）')
    parser.add_argument('--skip-data', action='store_true',
                       help='跳过数据生成步骤（数据已存在时使用）')
    parser.add_argument('--quick', action='store_true',
                       help='快速模式：跳过耗时的训练和过拟合检验')
    args = parser.parse_args()

    print("=" * 70)
    print("4.1.9.final 跨文化适应预测模型 - 完整运行流程")
    print("=" * 70)
    print(f"工作目录: {SCRIPT_DIR}")
    print()

    # 检查数据
    missing_data = check_data_exists()
    if missing_data:
        print("[警告] 以下数据文件缺失：")
        for f in missing_data:
            print(f"   - {f}")
        if 'data/processed/real_data_103.xlsx' in missing_data:
            print("\n[错误] 真实数据文件 real_data_103.xlsx 缺失，无法继续！")
            print("   请将真实数据文件放到 data/processed/ 目录下。")
            sys.exit(1)
        print()

    # 定义步骤
    steps = [
        (1, 'generate_data',      'generate_interaction_preserved_dataset.py',
         '生成保留交互效应的10万样本数据'),
        (2, 'train_v7',           'train_v7_ultimate.py',
         'V7 Ultimate训练（贝叶斯优化+XGB+LGB+CatBoost+DeepFM NAS+Stacking）'),
        (3, 'check_overfitting',  'check_v7_overfitting.py',
         'V7 Stacking过拟合检验（嵌套CV验证）'),
        (4, 'interpretability',   'interpretability_analysis.py',
         '解释性分析（SHAP+聚类+心理学报告）'),
        (5, 'individual_reports', 'generate_individual_reports.py',
         '生成103个样本个性化心理学报告（PDF+Excel+Markdown）'),
    ]

    # 确定要运行的步骤
    if args.step:
        run_steps = set(args.step)
    elif args.quick:
        run_steps = {4, 5}  # 快速模式只运行解释性分析
        print("[快速模式] 只运行解释性分析和个案报告生成")
    else:
        run_steps = {1, 2, 3, 4, 5}

    # 检查数据生成是否需要跳过
    skip_data_gen = args.skip_data or (
        (SCRIPT_DIR / 'data/processed/improved_100k.csv').exists() and
        (SCRIPT_DIR / 'data/processed/interaction_preserved_100k.csv').exists()
    )

    if skip_data_gen and 1 in run_steps:
        print("[信息] 数据文件已存在，跳过数据生成步骤（Step 1）")
        print("   如需重新生成，请删除 data/processed/improved_100k.csv 后重试")

    # 运行步骤
    total_start = time.time()
    results = {}

    for step_num, step_name, script, description in steps:
        if step_num not in run_steps:
            continue

        skip = (step_num == 1 and skip_data_gen)
        success = run_step(step_name, script, description, skip=skip)
        results[step_num] = success

        if not success and step_num in {1, 2}:
            print(f"\n[失败] 关键步骤 {step_num} 失败，后续步骤可能无法正常运行")
            if step_num == 2:
                print("   注意：步骤4和5不依赖步骤2的结果，可以单独运行")

    # 汇总
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print("运行汇总")
    print("=" * 70)
    for step_num, step_name, script, description in steps:
        if step_num in results:
            status = "[成功]" if results[step_num] else "[失败]"
            print(f"  Step {step_num} ({step_name}): {status}")
        elif step_num in run_steps:
            print(f"  Step {step_num} ({step_name}): [跳过]")

    print(f"\n总用时: {total_elapsed:.1f}秒 ({total_elapsed/60:.1f}分钟)")

    # 输出文件位置
    print("\n" + "=" * 70)
    print("主要输出文件")
    print("=" * 70)
    output_files = [
        ('results/v7_ultimate/v7_results.json',
         'V7模型完整结果（R²=0.728）'),
        ('results/overfitting_check_v7/overfitting_check.json',
         '过拟合检验结果（嵌套CV R²=0.701）'),
        ('results/interpretability/psychology_interpretation_report.md',
         '心理学解释性报告'),
        ('results/interpretability/shap_summary.png',
         'SHAP特征重要性图'),
        ('results/interpretability/cluster_analysis.png',
         '103样本聚类分析图'),
        ('results/individual_reports/individual_reports_all_103.pdf',
         '103个样本个性化PDF报告'),
        ('results/individual_reports/all_samples_summary.xlsx',
         '103个样本汇总Excel'),
        ('results/individual_reports/individual_cases_detailed.md',
         '103个样本详细Markdown报告'),
        ('results/final_robust/real_data_predictions.csv',
         '103样本预测结果CSV'),
        ('FINAL_OPTIMIZATION_REPORT.md',
         '最终优化报告（含完整模型演进历程）'),
    ]

    for filepath, desc in output_files:
        full_path = SCRIPT_DIR / filepath
        exists = "[OK]" if full_path.exists() else "[--]"
        print(f"  {exists} {filepath}")
        print(f"     {desc}")

    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
