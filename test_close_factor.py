"""
测试 close 因子的得分计算
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data():
    """加载数据"""
    # 尝试多个可能的数据文件路径
    possible_paths = [
        '/Users/duanjin/Desktop/强化学习/PPO/data/btc_15min.csv',
        '/Users/duanjin/Desktop/强化学习/PPO/data/btc_data.csv',
        '/Users/duanjin/Desktop/强化学习/PPO/data/data.csv',
    ]

    data = None
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"找到数据文件: {path}")
            data = pd.read_csv(path)
            break

    if data is None:
        # 如果没有找到数据文件，生成模拟数据
        logger.warning("未找到数据文件，生成模拟数据...")
        return generate_synthetic_data()

    return data

def generate_synthetic_data():
    """生成模拟数据"""
    logger.info("生成模拟数据...")
    np.random.seed(42)
    n_bars = 3000

    # 生成价格序列（模拟BTC）
    returns = np.random.randn(n_bars) * 0.02 + 0.0001  # 加入正向漂移
    prices = 10000 * np.exp(np.cumsum(returns))

    # 生成OHLCV数据
    data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(n_bars) * 0.005),
        'high': prices * (1 + np.abs(np.random.randn(n_bars)) * 0.01),
        'low': prices * (1 - np.abs(np.random.randn(n_bars)) * 0.01),
        'close': prices,
        'volume': np.random.rand(n_bars) * 1000 + 500,
    })

    # 调整high和low以确保合理性
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)

    logger.info(f"生成了 {len(data)} 条模拟数据")
    return data

def calculate_close_factor_score():
    """计算 close 因子的得分"""
    logger.info("="*80)
    logger.info("开始计算 close 因子得分...")
    logger.info("="*80)

    # 1. 加载数据
    data = load_data()
    logger.info(f"数据形状: {data.shape}")
    logger.info(f"数据列: {data.columns.tolist()}")

    # 2. 创建配置
    config = TrainingConfig()

    # 3. 准备训练数据
    train_size = int(len(data) * 0.6)
    train_data = data[:train_size].copy()
    val_data = data[train_size:].copy()

    logger.info(f"训练集大小: {len(train_data)}")
    logger.info(f"验证集大小: {len(val_data)}")

    # 4. 计算目标值（未来收益）
    train_data['target'] = train_data['close'].pct_change(config.prediction_horizon).shift(-config.prediction_horizon)
    val_data['target'] = val_data['close'].pct_change(config.prediction_horizon).shift(-config.prediction_horizon)

    # 去掉NaN
    train_data = train_data.dropna()
    val_data = val_data.dropna()

    train_target = train_data['target']
    val_target = val_data['target']

    logger.info(f"训练集有效数据: {len(train_data)}")
    logger.info(f"验证集有效数据: {len(val_data)}")

    # 5. 准备因子（测试多种 close 相关因子）
    # 方案1: close 的收益率
    train_returns = train_data['close'].pct_change(5)  # 5期动量
    val_returns = val_data['close'].pct_change(5)

    # 滚动标准化
    train_mean = train_returns.rolling(100, min_periods=20).mean()
    train_std = train_returns.rolling(100, min_periods=20).std()

    train_factor = ((train_returns - train_mean) / (train_std + 1e-8)).fillna(0).clip(-3, 3)

    # 验证集使用相同的统计量（扩展窗口）
    # 为了简化，这里使用验证集自己的滚动统计量（实际应该用训练集的最后统计量）
    val_mean = val_returns.rolling(100, min_periods=20).mean()
    val_std = val_returns.rolling(100, min_periods=20).std()
    val_factor = ((val_returns - val_mean) / (val_std + 1e-8)).fillna(0).clip(-3, 3)

    logger.info(f"\nclose 动量因子统计 (5期收益率):")
    logger.info(f"  训练集: mean={train_factor.mean():.4f}, std={train_factor.std():.4f}, valid={train_factor.notna().sum()}")
    logger.info(f"  验证集: mean={val_factor.mean():.4f}, std={val_factor.std():.4f}, valid={val_factor.notna().sum()}")

    # 方案2: 也测试简单的标准化 close
    train_close_norm = (train_data['close'] - train_data['close'].rolling(100, min_periods=20).mean()) / (train_data['close'].rolling(100, min_periods=20).std() + 1e-8)
    train_close_norm = train_close_norm.fillna(0).clip(-3, 3)

    logger.info(f"\n标准化 close 统计:")
    logger.info(f"  训练集: mean={train_close_norm.mean():.4f}, std={train_close_norm.std():.4f}")

    # 6. 创建评估器
    logger.info("\n创建评估器...")
    evaluator = ICDiversityEvaluator(config)

    # 创建 Combiner（模拟空池子）
    combiner = ImprovedCombinationModel(config, max_alpha_count=15)
    combiner.set_targets(train_target, val_target)
    combiner.set_evaluator(evaluator)

    # 注入 combiner 到 evaluator
    evaluator.set_combiner(combiner)

    logger.info("评估器创建成功")

    # 7. 计算得分
    logger.info("\n" + "="*80)
    logger.info("计算 close 因子得分...")
    logger.info("="*80)

    # 计算增量 Sharpe（试算模式）
    result = combiner.evaluate_new_factor(
        alpha_info={'name': 'close'},
        train_factor=train_factor,
        val_factor=val_factor
    )

    incremental_sharpe = result.get('train_incremental_sharpe', 0.0)
    train_stats = result.get('train_stats', {})

    logger.info("\n" + "="*80)
    logger.info("📊 计算结果")
    logger.info("="*80)
    logger.info(f"增量 Sharpe:        {incremental_sharpe:.6f}")
    logger.info(f"训练集 Sharpe:      {train_stats.get('sharpe', 0.0):.6f}")
    logger.info(f"基准 Sharpe:        {combiner.base_train_score:.6f} (空池子)")
    logger.info(f"")
    logger.info(f"入池阈值 (第1个):   -0.03")
    logger.info(f"是否满足入池条件:   {'✅ 是' if incremental_sharpe > -0.03 else '❌ 否'}")
    logger.info("="*80)

    # 7.5 调试：查看滚动 Sharpe 的计算过程
    logger.info("\n" + "="*80)
    logger.info("🔍 调试：分析滚动 Sharpe 计算过程")
    logger.info("="*80)

    # 手动计算一次，查看中间结果
    performance_eval = evaluator.performance_evaluator
    net_returns, gross_returns, signals = performance_eval.calculate_net_returns(
        train_factor, train_target
    )

    if len(net_returns) > 0:
        logger.info(f"净收益序列长度: {len(net_returns)}")
        logger.info(f"净收益统计: mean={net_returns.mean():.6f}, std={net_returns.std():.6f}")
        logger.info(f"净收益范围: [{net_returns.min():.4f}, {net_returns.max():.4f}]")

        # 计算滚动 Sharpe
        bars_per_day = 24 * 60 / config.bar_minutes
        window_bars = int(3 * bars_per_day)  # 3天窗口
        window_bars = max(30, min(window_bars, len(net_returns) // 5))

        logger.info(f"滚动窗口大小: {window_bars} bars")

        rolling_mean = net_returns.rolling(window=window_bars, min_periods=window_bars//2).mean()
        rolling_std = net_returns.rolling(window=window_bars, min_periods=window_bars//2).std()
        rolling_std = rolling_std.replace(0, np.nan)

        rolling_sharpe = (rolling_mean / (rolling_std + 1e-9)) * np.sqrt(performance_eval.bars_per_year)
        rolling_sharpe = rolling_sharpe.dropna().clip(-50, 50)

        logger.info(f"滚动 Sharpe 序列长度: {len(rolling_sharpe)}")
        if len(rolling_sharpe) > 0:
            logger.info(f"滚动 Sharpe 统计: mean={rolling_sharpe.mean():.4f}, std={rolling_sharpe.std():.4f}")
            logger.info(f"滚动 Sharpe 范围: [{rolling_sharpe.min():.4f}, {rolling_sharpe.max():.4f}]")
            logger.info(f"滚动 Sharpe 前10个值: {rolling_sharpe.head(10).values}")

            # 计算稳定性得分
            mean_s = rolling_sharpe.mean()
            std_s = rolling_sharpe.std()
            stability_score = mean_s - 1.5 * std_s
            logger.info(f"\n稳定性得分计算:")
            logger.info(f"  Mean(Rolling Sharpe) = {mean_s:.4f}")
            logger.info(f"  Std(Rolling Sharpe) = {std_s:.4f}")
            logger.info(f"  Stability = {mean_s:.4f} - 1.5 × {std_s:.4f} = {stability_score:.4f}")
        else:
            logger.warning("滚动 Sharpe 序列为空！")
    else:
        logger.warning("净收益序列为空！")

    logger.info("="*80)

    # 8. 计算详细指标
    logger.info("\n计算详细指标...")
    metrics = performance_eval.calculate_comprehensive_metrics(
        train_factor, train_target, window_days=3
    )

    if 'error' not in metrics:
        logger.info("\n" + "="*80)
        logger.info("📈 详细指标")
        logger.info("="*80)
        logger.info(f"IC:                 {metrics['ic']:.4f}")
        logger.info(f"Sharpe比率:         {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Sharpe稳定性:       {metrics['sharpe_stability']:.2f}")
        logger.info(f"总收益:             {metrics['total_return']*100:.2f}%")
        logger.info(f"年化收益:           {metrics['annual_return']*100:.2f}%")
        logger.info(f"波动率(年化):       {metrics['volatility']*100:.2f}%")
        logger.info(f"最大回撤:           {metrics['max_drawdown']*100:.2f}%")
        logger.info(f"Calmar比率:         {metrics['calmar_ratio']:.2f}")
        logger.info(f"胜率:               {metrics['win_rate']*100:.1f}%")
        logger.info(f"换手率:             {metrics['turnover']:.4f}")
        logger.info(f"平均持仓:           {metrics['avg_position']:.4f}")
        logger.info(f"交易周期数:         {metrics['num_periods']}")
        logger.info("="*80)

    return {
        'incremental_sharpe': incremental_sharpe,
        'train_sharpe': train_stats.get('sharpe', 0.0),
        'qualifies': incremental_sharpe > -0.03,
        'metrics': metrics if 'error' not in metrics else None
    }

if __name__ == "__main__":
    try:
        result = calculate_close_factor_score()

        print("\n" + "="*80)
        print("🎯 最终结论")
        print("="*80)
        if result['qualifies']:
            print(f"✅ close 因子满足入池条件！")
            print(f"   增量 Sharpe = {result['incremental_sharpe']:.6f} > -0.03")
        else:
            print(f"❌ close 因子不满足入池条件")
            print(f"   增量 Sharpe = {result['incremental_sharpe']:.6f} <= -0.03")
        print("="*80)

    except Exception as e:
        logger.error(f"计算过程出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
