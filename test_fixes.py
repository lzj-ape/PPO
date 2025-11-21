"""
快速测试脚本 - 验证修复效果
"""
import sys
import os
import numpy as np
import pandas as pd

# 添加路径
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'factor'))

from config import TrainingConfig

# 测试1: 验证学习率配置
print("=" * 60)
print("测试 1: 验证配置修复")
print("=" * 60)
config = TrainingConfig()
print(f"✓ lr_actor: {config.lr_actor} (应该是 3e-4 = {3e-4})")
print(f"✓ lr_critic: {config.lr_critic} (应该是 3e-4 = {3e-4})")
print(f"✓ entropy_coeff: {config.entropy_coeff} (应该是 0.05)")
assert config.lr_actor == 3e-4, "学习率未正确修复"
assert config.entropy_coeff == 0.05, "熵系数未正确修复"
print("✅ 配置修复成功\n")

# 测试2: 验证evaluator的修复
print("=" * 60)
print("测试 2: 验证 rolling_sharpe_stability 修复")
print("=" * 60)
from evaluator import ICDiversityEvaluator

evaluator = ICDiversityEvaluator(config)

# 创建测试数据
np.random.seed(42)
n = 3000
predictions = pd.Series(np.random.randn(n).cumsum() + 10000,
                       index=pd.date_range('2023-01-01', periods=n, freq='15min'))
targets = pd.Series(np.random.randn(n) * 0.01,
                   index=predictions.index)

# 测试稳定性计算
score = evaluator.calculate_rolling_sharpe_stability(predictions, targets)
print(f"✓ Stability Score: {score:.4f}")
print(f"✓ Score 在合理范围内: {-2.0 <= score <= 2.0}")
assert -2.0 <= score <= 2.0, f"Score {score} 超出合理范围 [-2.0, 2.0]"
print("✅ rolling_sharpe_stability 修复成功\n")

# 测试3: 验证简单因子不会返回异常高分
print("=" * 60)
print("测试 3: 验证简单因子（常数）不会返回异常高分")
print("=" * 60)

# 创建一个常数因子
constant_pred = pd.Series(np.ones(n) * 10000, index=predictions.index)
score_constant = evaluator.calculate_rolling_sharpe_stability(constant_pred, targets)
print(f"✓ 常数因子 Score: {score_constant:.4f}")
assert score_constant == 0.0, f"常数因子应该返回0，实际返回 {score_constant}"
print("✅ 常数因子正确返回0\n")

# 测试4: 验证净收益计算
print("=" * 60)
print("测试 4: 验证 _get_net_returns 修复")
print("=" * 60)

net_returns = evaluator._get_net_returns(predictions, targets)
print(f"✓ Net Returns 数量: {len(net_returns)}")
print(f"✓ Net Returns 统计:")
print(f"    均值: {net_returns.mean():.6f}")
print(f"    标准差: {net_returns.std():.6f}")
print(f"    最小值: {net_returns.min():.6f}")
print(f"    最大值: {net_returns.max():.6f}")

# 常数因子的净收益应该为空
net_returns_const = evaluator._get_net_returns(constant_pred, targets)
print(f"✓ 常数因子的 Net Returns 数量: {len(net_returns_const)} (应该是0)")
assert len(net_returns_const) == 0, "常数因子应该返回空的净收益"
print("✅ _get_net_returns 修复成功\n")

print("=" * 60)
print("🎉 所有测试通过！修复验证成功")
print("=" * 60)
print("\n建议：")
print("1. 重新运行训练，使用修复后的配置")
print("2. 观察是否生成更复杂的因子表达式")
print("3. 检查奖励值是否在 [-2, 2] 的合理范围内")
print("4. 确认 PPO 策略是否正常更新（policy_loss 应该有变化）")
