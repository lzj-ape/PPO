"""
详细诊断第二个因子入池的计算过程
"""
import sys
import os
import pandas as pd
import numpy as np
import logging

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'factor'))
sys.path.insert(0, os.path.join(current_dir, 'config'))

from config import TrainingConfig
from signals import SignalGenerator, PerformanceEvaluator
from evaluator import ICDiversityEvaluator
from combiner import ImprovedCombinationModel

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def generate_data():
    np.random.seed(42)
    n_bars = 3000
    returns = np.random.randn(n_bars) * 0.02 + 0.0001
    prices = 10000 * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        'close': prices,
        'volume': np.random.rand(n_bars) * 1000 + 500,
    })
    return data

def main():
    print("="*100)
    print("详细诊断：第二个因子入池时的计算逻辑")
    print("="*100)

    # 准备数据
    data = generate_data()
    config = TrainingConfig()

    train_size = int(len(data) * 0.6)
    train_data = data[:train_size].copy()
    val_data = data[train_size:].copy()

    train_data['target'] = train_data['close'].pct_change(10).shift(-10)
    val_data['target'] = val_data['close'].pct_change(10).shift(-10)

    train_data = train_data.dropna()
    val_data = val_data.dropna()

    # 创建两个因子
    # Factor 1: 5期动量
    train_factor1 = train_data['close'].pct_change(5)
    train_factor1 = ((train_factor1 - train_factor1.rolling(100, min_periods=20).mean()) /
                     (train_factor1.rolling(100, min_periods=20).std() + 1e-8)).fillna(0).clip(-3, 3)

    val_factor1 = val_data['close'].pct_change(5)
    val_factor1 = ((val_factor1 - val_factor1.rolling(100, min_periods=20).mean()) /
                   (val_factor1.rolling(100, min_periods=20).std() + 1e-8)).fillna(0).clip(-3, 3)

    # Factor 2: 20期动量
    train_factor2 = train_data['close'].pct_change(20)
    train_factor2 = ((train_factor2 - train_factor2.rolling(100, min_periods=20).mean()) /
                     (train_factor2.rolling(100, min_periods=20).std() + 1e-8)).fillna(0).clip(-3, 3)

    val_factor2 = val_data['close'].pct_change(20)
    val_factor2 = ((val_factor2 - val_factor2.rolling(100, min_periods=20).mean()) /
                   (val_factor2.rolling(100, min_periods=20).std() + 1e-8)).fillna(0).clip(-3, 3)

    # 创建评估器
    evaluator = ICDiversityEvaluator(config)
    combiner = ImprovedCombinationModel(config, max_alpha_count=15)
    combiner.set_targets(train_data['target'], val_data['target'])
    combiner.set_evaluator(evaluator)
    evaluator.set_combiner(combiner)

    print(f"\n📊 初始状态:")
    print(f"  池子大小: {len(combiner.alpha_pool)}")
    print(f"  基准分数: {combiner.base_train_score:.6f}")

    # ==================== 第一个因子 ====================
    print("\n" + "="*100)
    print("第一个因子: momentum_5 (5期动量)")
    print("="*100)

    print("\n🔍 步骤 1: 试算 (Trial Mode)")
    print("-" * 100)

    result1_trial = combiner.evaluate_new_factor(
        {'name': 'momentum_5'}, train_factor1, val_factor1
    )

    inc1 = result1_trial['train_incremental_sharpe']
    new_score1 = result1_trial['train_stats']['sharpe']

    print(f"  当前池子大小: 0")
    print(f"  当前基准分数: 0.000000 (空池子)")
    print(f"  ↓")
    print(f"  使用 Ridge 拟合单因子 [momentum_5]")
    print(f"  计算组合的 Mean(Rolling Sharpe)")
    print(f"  ↓")
    print(f"  新组合分数: {new_score1:.6f}")
    print(f"  增量 Sharpe = {new_score1:.6f} - 0.000000 = {inc1:.6f}")
    print(f"  ↓")
    print(f"  阈值判断: {inc1:.6f} > -0.03? {'✅ 是' if inc1 > -0.03 else '❌ 否'}")

    if inc1 > -0.03:
        print("\n🔄 步骤 2: 提交 (Commit Mode)")
        print("-" * 100)

        commit1 = combiner.add_alpha_and_optimize(
            {'name': 'momentum_5'}, train_factor1, val_factor1
        )

        print(f"  ✅ 因子添加到池子")
        print(f"  池子大小: 0 → {commit1['pool_size']}")
        print(f"  基准分数: 0.000000 → {commit1['current_train_score']:.6f}")
        print(f"  权重: [{combiner.current_weights[0]:.6f}]")

    # ==================== 第二个因子 ====================
    print("\n" + "="*100)
    print("第二个因子: momentum_20 (20期动量)")
    print("="*100)

    print("\n🔍 步骤 1: 试算 (Trial Mode)")
    print("-" * 100)

    base_before = combiner.base_train_score

    result2_trial = combiner.evaluate_new_factor(
        {'name': 'momentum_20'}, train_factor2, val_factor2
    )

    inc2 = result2_trial['train_incremental_sharpe']
    new_score2 = result2_trial['train_stats']['sharpe']

    print(f"  当前池子大小: 1")
    print(f"  当前基准分数: {base_before:.6f} (momentum_5 单因子)")
    print(f"  当前池子矩阵: [momentum_5]")
    print(f"  ↓")
    print(f"  构造临时矩阵: [momentum_5, momentum_20]")
    print(f"  使用 Ridge 拟合两因子组合")
    print(f"    - Ridge 会学习最优权重 w1, w2")
    print(f"    - 组合预测 = w1 × momentum_5 + w2 × momentum_20")
    print(f"  ↓")
    print(f"  计算新组合的 Mean(Rolling Sharpe)")
    print(f"  新组合分数: {new_score2:.6f}")
    print(f"  ↓")
    print(f"  增量 Sharpe = {new_score2:.6f} - {base_before:.6f} = {inc2:.6f}")
    print(f"  ↓")
    print(f"  阈值判断: {inc2:.6f} > -0.03? {'✅ 是' if inc2 > -0.03 else '❌ 否'}")
    print(f"  说明: 前3个因子使用阈值 -0.03，因为样本少，允许轻微负增量")

    if inc2 > -0.03:
        print("\n🔄 步骤 2: 提交 (Commit Mode)")
        print("-" * 100)

        commit2 = combiner.add_alpha_and_optimize(
            {'name': 'momentum_20'}, train_factor2, val_factor2
        )

        print(f"  ✅ 因子添加到池子")
        print(f"  池子大小: 1 → {commit2['pool_size']}")
        print(f"  基准分数: {base_before:.6f} → {commit2['current_train_score']:.6f}")
        print(f"  实际增量: {commit2['incremental_contribution']:.6f}")
        print(f"  权重: {combiner.current_weights}")
        print(f"    - [0] momentum_5:  {combiner.current_weights[0]:.6f}")
        print(f"    - [1] momentum_20: {combiner.current_weights[1]:.6f}")

    # ==================== 关键点总结 ====================
    print("\n" + "="*100)
    print("🎯 关键点总结")
    print("="*100)

    print(f"""
1. **第一个因子 (池子为空)**:
   - 基准分数 = 0.0 (空池子)
   - 新分数 = Mean(Rolling Sharpe of 单因子)
   - 增量 = 新分数 - 0.0 = 新分数本身
   - 阈值 = -0.03

2. **第二个因子 (池子有1个因子)**:
   - 基准分数 = {base_before:.6f} (第一个因子的分数)
   - 构造临时矩阵 = [因子1, 因子2]
   - 使用 Ridge 拟合，学习最优权重 [w1, w2]
   - 组合预测 = w1 × 因子1 + w2 × 因子2
   - 新分数 = Mean(Rolling Sharpe of 组合预测)
   - 增量 = 新分数 - 基准分数
   - 阈值 = -0.03 (因为池子 < 3)

3. **Ridge 回归的作用**:
   - Ridge 会自动找到最优的线性组合权重
   - 权重可以是负数，意味着反向使用该因子
   - Alpha=1.0 是 L2 正则化系数，防止过拟合
   - fit_intercept=False 表示不学习截距项

4. **评价标准统一**:
   - 无论池子里有几个因子，都使用 Mean(Rolling Sharpe) 评价
   - 增量 = 新组合的 Sharpe - 旧组合的 Sharpe
   - 这是真正的"协同效应"：看新因子能否提升组合表现

5. **阈值自适应**:
   - 前3个: -0.03 (容忍负增量，快速启动)
   - 4-5个: 0.001 (要求微小正增量)
   - 6-10个: 0.003 (要求明显正增量)
   - 10+个: 0.006 (要求显著正增量，池子已经很好了)
""")

    print("="*100)

if __name__ == "__main__":
    main()
