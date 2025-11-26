import sys
sys.path.append('/Users/duanjin/Desktop/强化学习/PPO')
from factor.operators import TimeSeriesOperators
import pandas as pd
import numpy as np

print("测试开始")

ts_ops = TimeSeriesOperators()
data = pd.Series(np.random.randn(100))

# 测试sma
result = ts_ops.sma(data, 5)
print(f"sma5: valid={(~result.isna()).sum()}/{len(result)}")

# 测试std
result2 = ts_ops.std(data, 20)
print(f"std20: valid={(~result2.isna()).sum()}/{len(result2)}")

print("测试完成")
