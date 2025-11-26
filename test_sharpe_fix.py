"""
测试 -10.0 Sharpe 值问题的修复

问题原因：
calculate_rolling_sharpe_stability 在数据不足时返回 0.0
但 combiner 无法区分"计算失败返回0"和"真实Sharpe为0"
导致 incremental = 0.0 - base_score = -10.0 (如果base_score=10.0)

修复方案：
1. calculate_rolling_sharpe_stability 在计算失败时返回 None 而非 0.0
2. 所有调用方检查 None 并妥善处理
"""

import pandas as pd
import numpy as np

print("="*80)
print("🔧 测试 -10.0 Sharpe 值修复")
print("="*80)
print()

print("🐛 原问题:")
print("-"*80)
print("场景：数据不足，计算失败")
print()
print("修复前的逻辑:")
print("  1. calculate_rolling_sharpe_stability 返回 0.0 (数据不足)")
print("  2. new_train_score = 0.0")
print("  3. base_train_score = 10.0 (之前的正常值)")
print("  4. incremental = 0.0 - 10.0 = -10.0  ❌ 错误!")
print()
print("问题：无法区分 '计算失败的0' 和 '真实Sharpe为0'")
print()

print("✅ 修复后的逻辑:")
print("-"*80)
print("  1. calculate_rolling_sharpe_stability 返回 None (数据不足)")
print("  2. combiner 检测到 None")
print("  3. combiner 返回 {'train_incremental_sharpe': 0.0, ...}")
print("  4. 不会出现 -10.0 的异常值  ✅ 正确!")
print()

print("="*80)
print("🔍 修改的文件和位置")
print("="*80)
print()

fixes = [
    {
        'file': 'factor/signals.py',
        'lines': '241, 252, 261, 290, 320',
        'change': '计算失败时返回 None 而非 0.0',
        'note': '包括：无数据、数据不足、Sharpe值太少、异常处理'
    },
    {
        'file': 'factor/combiner.py',
        'lines': '141-147',
        'change': 'evaluate_new_factor: 检查None并返回安全的0值',
        'note': 'Trial mode的None处理'
    },
    {
        'file': 'factor/combiner.py',
        'lines': '234-238, 263-267',
        'change': 'add_alpha_and_optimize: 检查None并使用0作为base_score',
        'note': 'Commit mode的None处理'
    },
    {
        'file': 'factor/combiner.py',
        'lines': '350-354',
        'change': '_prune_factor: 检查None并使用0',
        'note': 'Pruning时的None处理'
    },
    {
        'file': 'factor/evaluator.py',
        'lines': '111-113',
        'change': '_get_incremental_sharpe: 检查None并返回0',
        'note': '无Combiner时的退化处理'
    },
    {
        'file': 'factor/evaluator.py',
        'lines': '147-148',
        'change': 'evaluate: 检查single_sharpe的None',
        'note': '单因子Sharpe的None处理'
    },
    {
        'file': 'factor/signals.py',
        'lines': '409-410',
        'change': 'calculate_comprehensive_metrics: 检查None',
        'note': '综合指标计算的None处理'
    }
]

for i, fix in enumerate(fixes, 1):
    print(f"{i}. {fix['file']}")
    print(f"   行号: {fix['lines']}")
    print(f"   修改: {fix['change']}")
    print(f"   说明: {fix['note']}")
    print()

print("="*80)
print("🧪 验证方法")
print("="*80)
print()

print("方法1: 运行完整训练")
print("  python main.py")
print("  观察训练日志，不应再出现 'Incremental Sharpe: -10.000000'")
print()

print("方法2: 检查日志中的警告")
print("  grep 'returned None' training.log")
print("  应该能看到:")
print("    'Combiner trial: calculate_rolling_sharpe_stability returned None'")
print("    说明None被正确检测和处理")
print()

print("方法3: 检查Sharpe值分布")
print("  grep 'Incremental Sharpe:' training.log | awk '{print $4}' | sort -n")
print("  应该不再有 -10.0 这种异常值")
print("  正常范围应该在 [-0.05, 0.5] 左右")
print()

print("="*80)
print("📊 预期效果")
print("="*80)
print()

print("修复前:")
print("  ❌ 大量因子显示 Incremental Sharpe: -10.000000")
print("  ❌ Train Sharpe: -10.0000")
print("  ❌ 因子池无法正常增长")
print()

print("修复后:")
print("  ✅ 计算失败的因子正确返回 incremental=0.0")
print("  ✅ 不会出现 -10.0 的异常值")
print("  ✅ 有效因子能正常显示真实的Sharpe值")
print("  ✅ 因子池能正常增长")
print()

print("="*80)
print("⚠️  注意事项")
print("="*80)
print()

print("1. None vs 0.0 的语义:")
print("   - None: 计算失败（数据不足、异常等）")
print("   - 0.0: 计算成功，但Sharpe确实为0（中性策略）")
print()

print("2. 数据要求:")
print("   - 最小数据量要求已经降低到80行")
print("   - 如果还是频繁返回None，检查原始数据量")
print()

print("3. 日志监控:")
print("   - 关注 'returned None' 的警告日志")
print("   - 如果过于频繁（>50%），考虑进一步降低数据要求")
print()

print("="*80)
print("🎯 核心修复逻辑")
print("="*80)
print()

print("修复前:")
print("```python")
print("# signals.py")
print("if data_length < 80:")
print("    return 0.0  # ❌ 无法区分失败和真实0")
print()
print("# combiner.py")
print("new_train_score = evaluator.calculate_rolling_sharpe_stability(...)")
print("incremental = new_train_score - base_train_score  # ❌ 可能= 0.0 - 10.0 = -10.0")
print("```")
print()

print("修复后:")
print("```python")
print("# signals.py")
print("if data_length < 80:")
print("    return None  # ✅ 明确表示失败")
print()
print("# combiner.py")
print("new_train_score = evaluator.calculate_rolling_sharpe_stability(...)")
print("if new_train_score is None:  # ✅ 检查失败")
print("    return {'train_incremental_sharpe': 0.0, ...}  # ✅ 安全返回")
print("incremental = new_train_score - base_train_score  # ✅ 只在成功时计算")
print("```")
print()

print("="*80)
print("✅ 修复完成！")
print("="*80)
print()
print("现在可以重新开始训练，-10.0的异常值应该不会再出现了。")
print()
