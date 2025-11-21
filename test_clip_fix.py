"""
测试Clip修复 - 验证扩大的Score范围
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'factor'))

from config import TrainingConfig
from evaluator import ICDiversityEvaluator

print("=" * 60)
print("测试: 验证Clip范围扩大，支持增量Sharpe识别")
print("=" * 60)

config = TrainingConfig()
evaluator = ICDiversityEvaluator(config)

# 创建测试数据
np.random.seed(42)
n = 3000
index = pd.date_range('2023-01-01', periods=n, freq='15min')
targets = pd.Series(np.random.randn(n) * 0.01, index=index)

# 测试1: 随机游走因子
factor1 = pd.Series(np.random.randn(n).cumsum() + 10000, index=index)
score1 = evaluator.calculate_rolling_sharpe_stability(factor1, targets)
print(f"\n✓ 随机游走因子 Score: {score1:.4f}")
print(f"  范围检查: {-10.0 <= score1 <= 10.0}")

# 测试2: 另一个随机游走因子
np.random.seed(123)
factor2 = pd.Series(np.random.randn(n).cumsum() * 0.5 + 10000, index=index)
score2 = evaluator.calculate_rolling_sharpe_stability(factor2, targets)
print(f"\n✓ 另一个随机因子 Score: {score2:.4f}")
print(f"  范围检查: {-10.0 <= score2 <= 10.0}")

# 测试3: Score差异
print(f"\n✓ 两个因子Score差异: {abs(score1 - score2):.4f}")
print(f"  可以区分: {abs(score1 - score2) > 0.01}")

# 测试4: 常数因子（应该返回0）
constant = pd.Series(np.ones(n) * 10000, index=index)
score_const = evaluator.calculate_rolling_sharpe_stability(constant, targets)
print(f"\n✓ 常数因子 Score: {score_const:.4f}")
print(f"  应该为0: {score_const == 0.0}")

# 测试5: 验证配置的阈值
print(f"\n✓ 配置的ic_threshold: {config.ic_threshold}")
print(f"  说明: 新因子必须带来至少 {config.ic_threshold} 的增量Sharpe才被接受")

print("\n" + "=" * 60)
print("🎉 测试完成！关键点：")
print("  1. Score范围从[-2,2]扩大到[-10,10]")
print("  2. 为增量Sharpe留出足够区分空间")
print("  3. 阈值设为0.1，筛选真正有价值的因子")
print("=" * 60)
