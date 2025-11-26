"""
直接测试Rolling Sharpe计算是否正常
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / 'factor'))
sys.path.append(str(current_dir / 'config'))
sys.path.append(str(current_dir / 'utils'))

from config import TrainingConfig
from evaluator import ICDiversityEvaluator
from data_generator import generate_realistic_market_data

# 生成测试数据
print("生成测试数据...")
data = generate_realistic_market_data(n_rows=1000, start_date='2024-01-01', seed=42)
config = TrainingConfig()

# 计算target
horizon = config.prediction_horizon
data['future_return'] = data['close'].pct_change(horizon).shift(-horizon).fillna(0)

# 分割数据
train_size = int(len(data) * config.train_ratio)
train_data = data.iloc[:train_size].copy()
train_target = train_data['future_return']

print(f"训练数据: {len(train_data)} 行")
print(f"目标均值: {train_target.mean():.6f}, 标准差: {train_target.std():.6f}")

# 创建评估器
evaluator = ICDiversityEvaluator(config)

# 测试1: 使用收盘价作为预测
print("\n" + "="*60)
print("测试1: 使用收盘价作为预测")
print("="*60)
predictions = train_data['close'].copy()
print(f"预测均值: {predictions.mean():.2f}, 标准差: {predictions.std():.2f}")

sharpe = evaluator.calculate_rolling_sharpe_stability(predictions, train_target)
print(f"Rolling Sharpe Stability: {sharpe:.6f}")

# 测试2: 使用收益率作为预测
print("\n" + "="*60)
print("测试2: 使用收益率作为预测")
print("="*60)
predictions = train_data['close'].pct_change().fillna(0)
print(f"预测均值: {predictions.mean():.6f}, 标准差: {predictions.std():.6f}")

sharpe = evaluator.calculate_rolling_sharpe_stability(predictions, train_target)
print(f"Rolling Sharpe Stability: {sharpe:.6f}")

# 测试3: 使用随机信号
print("\n" + "="*60)
print("测试3: 使用随机信号")
print("="*60)
np.random.seed(42)
predictions = pd.Series(np.random.randn(len(train_data)), index=train_data.index)
print(f"预测均值: {predictions.mean():.6f}, 标准差: {predictions.std():.6f}")

sharpe = evaluator.calculate_rolling_sharpe_stability(predictions, train_target)
print(f"Rolling Sharpe Stability: {sharpe:.6f}")

# 测试4: 使用完美的预测（等于target）
print("\n" + "="*60)
print("测试4: 使用完美预测（=target）")
print("="*60)
predictions = train_target.copy()
print(f"预测均值: {predictions.mean():.6f}, 标准差: {predictions.std():.6f}")

sharpe = evaluator.calculate_rolling_sharpe_stability(predictions, train_target)
print(f"Rolling Sharpe Stability: {sharpe:.6f}")

# 测试5: 检查_get_net_returns
print("\n" + "="*60)
print("测试5: 检查_get_net_returns方法")
print("="*60)
predictions = train_data['close'].pct_change().fillna(0) * 100  # 放大信号
print(f"预测均值: {predictions.mean():.6f}, 标准差: {predictions.std():.6f}")

net_returns = evaluator._get_net_returns(predictions, train_target)
print(f"净收益长度: {len(net_returns)}")
if len(net_returns) > 0:
    print(f"净收益均值: {net_returns.mean():.6f}, 标准差: {net_returns.std():.6f}")
    sharpe = evaluator.calculate_rolling_sharpe_stability(predictions, train_target)
    print(f"Rolling Sharpe Stability: {sharpe:.6f}")
else:
    print("⚠️  _get_net_returns返回空序列！")

print("\n" + "="*60)
print("诊断完成")
print("="*60)
