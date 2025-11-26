"""
手动复现rolling sharpe计算逻辑
"""
import pandas as pd
import numpy as np

# 模拟net_returns
np.random.seed(42)
net_returns = pd.Series(np.random.randn(600) * 0.01 + 0.001)

print(f"net_returns:")
print(f"  长度: {len(net_returns)}")
print(f"  均值: {net_returns.mean():.6f}")
print(f"  标准差: {net_returns.std():.6f}")

# 参数设置
bar_minutes = 15
window_days = 3
bars_per_day = 24 * 60 / bar_minutes  # 96
bars_per_year = 365 * 24 * 60 / bar_minutes  # 35040

ideal_window_bars = int(window_days * bars_per_day)  # 288
print(f"\nideal_window_bars: {ideal_window_bars}")

# 🔥 动态调整窗口
data_length = len(net_returns)
print(f"data_length: {data_length}")

if data_length < 150:
    print("❌ 数据太少,返回0")
else:
    window_bars = max(30, min(ideal_window_bars, data_length // 5))
    min_required_bars = window_bars * 2

    print(f"window_bars: {window_bars}")
    print(f"min_required_bars: {min_required_bars}")

    if data_length < min_required_bars:
        print(f"❌ data_length({data_length}) < min_required_bars({min_required_bars}), 返回0")
    else:
        print("✅ 数据足够,开始计算滚动Sharpe")

        # 计算滚动Sharpe
        rolling_mean = net_returns.rolling(window=window_bars, min_periods=window_bars//2).mean()
        rolling_std = net_returns.rolling(window=window_bars, min_periods=window_bars//2).std()

        print(f"\nrolling_mean:")
        print(f"  均值: {rolling_mean.mean():.6f}")
        print(f"  NaN数: {rolling_mean.isna().sum()}")

        print(f"\nrolling_std:")
        print(f"  均值: {rolling_std.mean():.6f}")
        print(f"  最小值: {rolling_std.min():.6f}")
        print(f"  =0的数量: {(rolling_std == 0).sum()}")
        print(f"  NaN数: {rolling_std.isna().sum()}")

        # 替换0
        rolling_std = rolling_std.replace(0, np.nan)

        # 计算滚动Sharpe
        rolling_sharpe = (rolling_mean / (rolling_std + 1e-9)) * np.sqrt(bars_per_year)
        rolling_sharpe = rolling_sharpe.dropna()

        print(f"\nrolling_sharpe (after dropna):")
        print(f"  长度: {len(rolling_sharpe)}")
        if len(rolling_sharpe) > 0:
            print(f"  均值: {rolling_sharpe.mean():.6f}")
            print(f"  标准差: {rolling_sharpe.std():.6f}")
            print(f"  最小值: {rolling_sharpe.min():.6f}")
            print(f"  最大值: {rolling_sharpe.max():.6f}")
        else:
            print("  ❌ dropna后变成空序列!")

        # Clip
        rolling_sharpe = rolling_sharpe.clip(-5, 5)

        if len(rolling_sharpe) < 10:
            print(f"❌ rolling_sharpe长度({len(rolling_sharpe)}) < 10, 返回0")
        else:
            print("✅ rolling_sharpe长度足够")

            mean_s = rolling_sharpe.mean()
            std_s = rolling_sharpe.std()

            print(f"\nmean_s: {mean_s:.6f}")
            print(f"std_s: {std_s:.6f}")

            if std_s < 1e-6:
                print(f"❌ std_s({std_s:.10f}) < 1e-6, 返回0")
            else:
                stability_penalty = 1.5
                stability_score = mean_s - stability_penalty * std_s
                stability_score = np.clip(stability_score, -10.0, 10.0)

                print(f"✅ stability_score: {stability_score:.6f}")
