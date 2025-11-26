"""
测试第二个、第三个因子入池时的计算逻辑
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
sys.path.insert(0, os.path.join(current_dir, 'utils'))

from config import TrainingConfig
from signals import SignalGenerator, PerformanceEvaluator
from evaluator import ICDiversityEvaluator
from combiner import ImprovedCombinationModel

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_synthetic_data():
    """生成模拟数据"""
    logger.info("生成模拟数据...")
    np.random.seed(42)
    n_bars = 3000

    # 生成价格序列（模拟BTC）
    returns = np.random.randn(n_bars) * 0.02 + 0.0001
    prices = 10000 * np.exp(np.cumsum(returns))

    # 生成OHLCV数据
    data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(n_bars) * 0.005),
        'high': prices * (1 + np.abs(np.random.randn(n_bars)) * 0.01),
        'low': prices * (1 - np.abs(np.random.randn(n_bars)) * 0.01),
        'close': prices,
        'volume': np.random.rand(n_bars) * 1000 + 500,
    })

    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)

    return data

def create_factor(data: pd.DataFrame, factor_type: str) -> pd.Series:
    """
    创建不同类型的因子

    Args:
        data: 价格数据
        factor_type: 因子类型 ('momentum_5', 'momentum_20', 'volatility', 'volume')
    """
    if factor_type == 'momentum_5':
        # 5期动量
        returns = data['close'].pct_change(5)
        mean = returns.rolling(100, min_periods=20).mean()
        std = returns.rolling(100, min_periods=20).std()
        factor = ((returns - mean) / (std + 1e-8)).fillna(0).clip(-3, 3)

    elif factor_type == 'momentum_20':
        # 20期动量（更长周期）
        returns = data['close'].pct_change(20)
        mean = returns.rolling(100, min_periods=20).mean()
        std = returns.rolling(100, min_periods=20).std()
        factor = ((returns - mean) / (std + 1e-8)).fillna(0).clip(-3, 3)

    elif factor_type == 'volatility':
        # 波动率因子
        returns = data['close'].pct_change()
        volatility = returns.rolling(20, min_periods=10).std()
        mean = volatility.rolling(100, min_periods=20).mean()
        std = volatility.rolling(100, min_periods=20).std()
        factor = ((volatility - mean) / (std + 1e-8)).fillna(0).clip(-3, 3)

    elif factor_type == 'volume':
        # 成交量因子
        volume_ma = data['volume'].rolling(20, min_periods=10).mean()
        volume_ratio = data['volume'] / (volume_ma + 1e-8)
        mean = volume_ratio.rolling(100, min_periods=20).mean()
        std = volume_ratio.rolling(100, min_periods=20).std()
        factor = ((volume_ratio - mean) / (std + 1e-8)).fillna(0).clip(-3, 3)

    else:
        raise ValueError(f"Unknown factor type: {factor_type}")

    return factor

def test_incremental_addition():
    """测试因子逐个添加的过程"""
    logger.info("="*80)
    logger.info("测试因子逐个添加的增量计算逻辑")
    logger.info("="*80)

    # 1. 准备数据
    data = generate_synthetic_data()
    config = TrainingConfig()

    train_size = int(len(data) * 0.6)
    train_data = data[:train_size].copy()
    val_data = data[train_size:].copy()

    # 计算目标值
    train_data['target'] = train_data['close'].pct_change(config.prediction_horizon).shift(-config.prediction_horizon)
    val_data['target'] = val_data['close'].pct_change(config.prediction_horizon).shift(-config.prediction_horizon)

    train_data = train_data.dropna()
    val_data = val_data.dropna()

    train_target = train_data['target']
    val_target = val_data['target']

    logger.info(f"训练集大小: {len(train_data)}, 验证集大小: {len(val_data)}")

    # 2. 创建评估器和组合器
    evaluator = ICDiversityEvaluator(config)
    combiner = ImprovedCombinationModel(config, max_alpha_count=15)
    combiner.set_targets(train_target, val_target)
    combiner.set_evaluator(evaluator)
    evaluator.set_combiner(combiner)

    # 3. 准备多个因子
    factor_types = ['momentum_5', 'momentum_20', 'volatility', 'volume']
    factors = {}

    for ftype in factor_types:
        train_factor = create_factor(train_data, ftype)
        val_factor = create_factor(val_data, ftype)
        factors[ftype] = {
            'train': train_factor,
            'val': val_factor
        }
        logger.info(f"创建因子 {ftype}: train_mean={train_factor.mean():.4f}, train_std={train_factor.std():.4f}")

    # 4. 逐个添加因子
    logger.info("\n" + "="*80)
    logger.info("开始逐个添加因子")
    logger.info("="*80)

    results = []

    for i, ftype in enumerate(factor_types, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"第 {i} 个因子: {ftype}")
        logger.info(f"当前池子大小: {len(combiner.alpha_pool)}")
        logger.info(f"当前基准分数: {combiner.base_train_score:.6f}")
        logger.info(f"{'='*80}")

        train_factor = factors[ftype]['train']
        val_factor = factors[ftype]['val']

        # 4.1 试算模式：计算增量
        trial_result = combiner.evaluate_new_factor(
            alpha_info={'name': ftype},
            train_factor=train_factor,
            val_factor=val_factor
        )

        incremental_sharpe = trial_result.get('train_incremental_sharpe', 0.0)
        new_train_score = trial_result['train_stats'].get('sharpe', 0.0)

        # 4.2 判断阈值
        pool_size = len(combiner.alpha_pool)
        base_threshold = config.ic_threshold

        if pool_size < 3:
            threshold = -0.03
            threshold_desc = "前3个因子，阈值=-0.03"
        elif pool_size < 5:
            threshold = 0.001
            threshold_desc = "第4-5个因子，阈值=0.001"
        elif pool_size < 10:
            threshold = base_threshold * 0.3
            threshold_desc = f"第6-10个因子，阈值={threshold:.4f}"
        else:
            threshold = base_threshold * 0.6
            threshold_desc = f"10个以上因子，阈值={threshold:.4f}"

        qualifies = incremental_sharpe > threshold

        logger.info(f"\n📊 试算结果:")
        logger.info(f"  基准分数 (旧):      {combiner.base_train_score:.6f}")
        logger.info(f"  新组合分数:         {new_train_score:.6f}")
        logger.info(f"  增量 Sharpe:        {incremental_sharpe:.6f}")
        logger.info(f"  入池阈值:           {threshold:.6f} ({threshold_desc})")
        logger.info(f"  是否满足条件:       {'✅ 是' if qualifies else '❌ 否'}")

        # 4.3 如果满足条件，真正添加
        if qualifies:
            logger.info(f"\n💚 因子 {ftype} 满足条件，添加到池子...")

            commit_result = combiner.add_alpha_and_optimize(
                alpha_info={'name': ftype, 'type': ftype},
                train_factor=train_factor,
                val_factor=val_factor
            )

            new_pool_size = commit_result.get('pool_size', 0)
            new_base_score = commit_result.get('current_train_score', 0.0)
            actual_increment = commit_result.get('incremental_contribution', 0.0)

            logger.info(f"  ✅ 添加成功!")
            logger.info(f"  新池子大小:         {new_pool_size}")
            logger.info(f"  新基准分数:         {new_base_score:.6f}")
            logger.info(f"  实际增量:           {actual_increment:.6f}")

            # 显示权重
            if combiner.current_weights is not None:
                weights = combiner.current_weights
                logger.info(f"  当前权重:")
                for j, w in enumerate(weights):
                    factor_name = combiner.alpha_pool[j].get('name', f'alpha_{j}')
                    logger.info(f"    [{j}] {factor_name}: {w:.6f}")

            result_status = "✅ 已添加"
        else:
            logger.info(f"\n💔 因子 {ftype} 不满足条件，拒绝添加")
            logger.info(f"  原因: 增量 {incremental_sharpe:.6f} <= 阈值 {threshold:.6f}")
            result_status = "❌ 被拒绝"

        # 记录结果
        results.append({
            'order': i,
            'factor': ftype,
            'pool_size_before': pool_size,
            'base_score_before': combiner.base_train_score if not qualifies else commit_result.get('current_train_score', 0.0) - actual_increment if qualifies else combiner.base_train_score,
            'incremental_sharpe': incremental_sharpe,
            'threshold': threshold,
            'qualifies': qualifies,
            'status': result_status,
            'pool_size_after': len(combiner.alpha_pool),
            'base_score_after': combiner.base_train_score
        })

    # 5. 总结
    logger.info("\n" + "="*80)
    logger.info("📋 总结")
    logger.info("="*80)

    results_df = pd.DataFrame(results)

    logger.info(f"\n最终池子大小: {len(combiner.alpha_pool)}")
    logger.info(f"最终基准分数: {combiner.base_train_score:.6f}")

    logger.info(f"\n各因子尝试结果:")
    for _, row in results_df.iterrows():
        logger.info(f"  [{row['order']}] {row['factor']:15s} | "
                   f"增量={row['incremental_sharpe']:7.4f} | "
                   f"阈值={row['threshold']:7.4f} | "
                   f"{row['status']}")

    logger.info(f"\n入池因子:")
    accepted = results_df[results_df['qualifies']]
    if len(accepted) > 0:
        for _, row in accepted.iterrows():
            logger.info(f"  [{row['order']}] {row['factor']}: 增量={row['incremental_sharpe']:.6f}")
    else:
        logger.info("  无因子入池")

    logger.info(f"\n被拒绝因子:")
    rejected = results_df[~results_df['qualifies']]
    if len(rejected) > 0:
        for _, row in rejected.iterrows():
            logger.info(f"  [{row['order']}] {row['factor']}: 增量={row['incremental_sharpe']:.6f} < 阈值={row['threshold']:.6f}")
    else:
        logger.info("  所有因子都入池了")

    return results_df

if __name__ == "__main__":
    try:
        results = test_incremental_addition()

        print("\n" + "="*80)
        print("✅ 测试完成")
        print("="*80)

    except Exception as e:
        logger.error(f"测试过程出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
