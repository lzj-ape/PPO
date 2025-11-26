"""
模拟实际挖掘过程，测试PPO生成的表达式
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

current_dir = Path.cwd()
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / 'config'))
sys.path.append(str(current_dir / 'factor'))
sys.path.append(str(current_dir / 'PPO'))
sys.path.append(str(current_dir / 'utils'))

from config import TrainingConfig
from miner_core import FactorMinerCore
from data_generator import generate_realistic_market_data

# 使用与main.ipynb相同的参数
data = generate_realistic_market_data(n_rows=50000, start_date='2023-01-01', seed=42)
config = TrainingConfig()

miner = FactorMinerCore(data, config=config, max_factors=100, max_expr_len=50)

print("="*60)
print("测试: 生成1个batch的表达式并评估")
print("="*60)

# 生成1个batch
batch_size = 16
batch_results = miner.expr_generator.generate_expression_batch(batch_size)

print(f"\n生成了 {len(batch_results)} 个表达式\n")

valid_count = 0
invalid_count = 0
reasons = {}

for idx, (tokens, state_ids, trajectory) in enumerate(batch_results):
    expr_str = ' '.join(tokens[:20])  # 只显示前20个token
    print(f"{idx+1}. {expr_str}...")

    # 评估
    eval_result = miner.factor_evaluator.evaluate_expression(tokens, trial_only=True)

    if eval_result['valid']:
        print(f"   ✅ 有效 - 奖励: {eval_result['reward']:.4f}")
        valid_count += 1
    else:
        reason = eval_result.get('reason', 'unknown')
        print(f"   ❌ 无效 - 原因: {reason}")
        invalid_count += 1
        reasons[reason] = reasons.get(reason, 0) + 1

print("\n" + "="*60)
print(f"总结: {valid_count} 有效, {invalid_count} 无效")
if reasons:
    print("\n失败原因分布:")
    for reason, count in sorted(reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"  {reason}: {count}/{batch_size} ({count/batch_size*100:.1f}%)")
print("="*60)

# 分析表达式长度
lengths = [len(tokens) for tokens, _, _ in batch_results]
print(f"\n表达式长度统计:")
print(f"  平均: {np.mean(lengths):.1f}")
print(f"  最小: {min(lengths)}, 最大: {max(lengths)}")
print(f"  中位数: {np.median(lengths):.1f}")
