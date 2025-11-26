"""
调试Rolling Sharpe计算的详细步骤
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
data = generate_realistic_market_data(n_rows=1000, start_date='2024-01-01', seed=42)
config = TrainingConfig()

horizon = config.prediction_horizon
data['future_return'] = data['close'].pct_change(horizon).shift(-horizon).fillna(0)

train_size = int(len(data) * config.train_ratio)
train_data = data.iloc[:train_size].copy()
train_target = train_data['future_return']

print(f"训练数据: {len(train_data)} 行")

# 创建评估器
evaluator = ICDiversityEvaluator(config)

# 使用完美预测测试
predictions = train_target.copy()
print(f"\n预测数据:")
print(f"  长度: {len(predictions)}")
print(f"  均值: {predictions.mean():.6f}")
print(f"  标准差: {predictions.std():.6f}")
print(f"  非零值数量: {(predictions != 0).sum()}")

# 手动执行_get_net_returns的逻辑
print("\n="*60)
print("Step 1: _get_net_returns")
print("="*60)

valid_idx = predictions.index.intersection(train_target.index)
print(f"Valid index length: {len(valid_idx)}")

pred_val = predictions.loc[valid_idx]
target_val = train_target.loc[valid_idx]

print(f"pred_val std: {pred_val.std():.10f}")
print(f"pred_val unique values: {len(pred_val.unique())}")

lookback = max(int(evaluator.sharpe_signal_lookback), 20)
min_periods = min(lookback // 2, 20)
print(f"lookback: {lookback}, min_periods: {min_periods}")

# 计算z-score
roll = pred_val.rolling(window=lookback, min_periods=min_periods)
mu = roll.mean()
sigma = roll.std()

print(f"\nmu 统计:")
print(f"  均值: {mu.mean():.6f}")
print(f"  NaN数: {mu.isna().sum()}")

print(f"\nsigma 统计:")
print(f"  均值: {sigma.mean():.6f}")
print(f"  最小值: {sigma.min():.6f}")
print(f"  =0的数量: {(sigma == 0).sum()}")
print(f"  NaN数: {sigma.isna().sum()}")

sigma_clean = sigma.replace(0, np.nan)
z_scores = (pred_val - mu) / (sigma_clean + 1e-9)

print(f"\nz_scores 统计:")
print(f"  均值: {z_scores.mean():.6f}")
print(f"  最小值: {z_scores.min():.6f}")
print(f"  最大值: {z_scores.max():.6f}")
print(f"  >1的数量: {(z_scores > 1.0).sum()}")
print(f"  <-1的数量: {(z_scores < -1.0).sum()}")
print(f"  NaN数: {z_scores.isna().sum()}")

# 生成信号
signals = pd.Series(0.0, index=pred_val.index)
signals[z_scores > 1.0] = evaluator.max_position
signals[z_scores < -1.0] = -evaluator.max_position

print(f"\nsignals 统计:")
print(f"  唯一值: {signals.unique()}")
print(f"  >0的数量: {(signals > 0).sum()}")
print(f"  <0的数量: {(signals < 0).sum()}")
print(f"  =0的数量: {(signals == 0).sum()}")
print(f"  标准差: {signals.std():.6f}")

unique_signals = signals.unique()
print(f"  唯一信号数量: {len(unique_signals)}")

if len(unique_signals) <= 1:
    print("  ❌ 信号完全不变，会返回空序列!")
else:
    print("  ✅ 信号有变化")

# 计算收益
gross_returns = signals * target_val
cost = signals.diff().abs().fillna(0.0) * evaluator.transaction_cost
net_returns = (gross_returns - cost).dropna()

print(f"\nnet_returns 统计:")
print(f"  长度: {len(net_returns)}")
if len(net_returns) > 0:
    print(f"  均值: {net_returns.mean():.6f}")
    print(f"  标准差: {net_returns.std():.6f}")
else:
    print("  ❌ 空序列!")

# 测试calculate_rolling_sharpe_stability
print("\n="*60)
print("Step 2: calculate_rolling_sharpe_stability")
print("="*60)

sharpe = evaluator.calculate_rolling_sharpe_stability(predictions, train_target)
print(f"最终 Rolling Sharpe Stability: {sharpe:.6f}")
