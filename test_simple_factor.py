"""
简单测试:给一个与target正相关的因子,看能否得到正Sharpe
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path.cwd() / 'config'))
sys.path.append(str(Path.cwd() / 'factor'))

from config import TrainingConfig
from evaluator import ICDiversityEvaluator

# 生成简单数据
np.random.seed(42)
n = 600
target = pd.Series(np.random.randn(n) * 0.01)

# 测试1: 完美预测 (predictions = target)
print("="*60)
print("测试1: 完美预测 (predictions = target)")
print("="*60)
predictions = target.copy()
ic = predictions.corr(target)
print(f"IC (相关系数): {ic:.4f}")

config = TrainingConfig()
evaluator = ICDiversityEvaluator(config)
sharpe = evaluator.calculate_rolling_sharpe_stability(predictions, target)
print(f"Rolling Sharpe Stability: {sharpe:.4f}\n")

# 测试2: 高度正相关 (predictions = target + noise)
print("="*60)
print("测试2: 高度正相关 (predictions = target * 0.8 + noise)")
print("="*60)
predictions = target * 0.8 + np.random.randn(n) * 0.005
ic = predictions.corr(target)
print(f"IC (相关系数): {ic:.4f}")

sharpe = evaluator.calculate_rolling_sharpe_stability(predictions, target)
print(f"Rolling Sharpe Stability: {sharpe:.4f}\n")

# 测试3: 负相关 (predictions = -target)
print("="*60)
print("测试3: 负相关 (predictions = -target)")
print("="*60)
predictions = -target
ic = predictions.corr(target)
print(f"IC (相关系数): {ic:.4f}")

sharpe = evaluator.calculate_rolling_sharpe_stability(predictions, target)
print(f"Rolling Sharpe Stability: {sharpe:.4f}\n")

# 测试4: 无相关 (随机)
print("="*60)
print("测试4: 无相关 (随机)")
print("="*60)
predictions = pd.Series(np.random.randn(n) * 0.01)
ic = predictions.corr(target)
print(f"IC (相关系数): {ic:.4f}")

sharpe = evaluator.calculate_rolling_sharpe_stability(predictions, target)
print(f"Rolling Sharpe Stability: {sharpe:.4f}\n")

# 测试5: 检查一个简单因子 close.pct_change()
print("="*60)
print("测试5: 真实市场数据 - close收益率")
print("="*60)

sys.path.append(str(Path.cwd() / 'utils'))
from data_generator import generate_realistic_market_data

data = generate_realistic_market_data(n_rows=1000, seed=42)
data['future_return'] = data['close'].pct_change(10).shift(-10).fillna(0)
train_size = 600
train_data = data.iloc[:train_size]

# 因子: 过去收益率
factor = train_data['close'].pct_change().fillna(0)
target = train_data['future_return']

ic = factor.corr(target)
print(f"IC (相关系数): {ic:.4f}")

sharpe = evaluator.calculate_rolling_sharpe_stability(factor, target)
print(f"Rolling Sharpe Stability: {sharpe:.4f}")
