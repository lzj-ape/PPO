"""
快速诊断 train_computation_failed 问题

11/16 的因子计算失败，这是严重的问题！
"""

import sys
from pathlib import Path

# 添加路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / 'factor') not in sys.path:
    sys.path.insert(0, str(project_root / 'factor'))
if str(project_root / 'config') not in sys.path:
    sys.path.insert(0, str(project_root / 'config'))

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_test_data(n=1000):
    """创建测试数据"""
    np.random.seed(42)
    data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(n) * 0.5),
        'volume': np.random.randint(1000, 10000, n),
        'high': 100 + np.cumsum(np.random.randn(n) * 0.5) + 1,
        'low': 100 + np.cumsum(np.random.randn(n) * 0.5) - 1,
        'open': 100 + np.cumsum(np.random.randn(n) * 0.5),
    })
    return data

def test_computation():
    """测试因子计算"""
    from operators import TimeSeriesOperators

    print("="*80)
    print("🔍 诊断 train_computation_failed 问题")
    print("="*80)
    print()

    # 创建测试数据
    data = create_test_data(1000)
    print(f"✅ 测试数据创建成功: {len(data)} 行")
    print(f"   特征: {list(data.columns)}")
    print(f"   数据范围: close [{data['close'].min():.2f}, {data['close'].max():.2f}]")
    print()

    # 初始化操作符
    ts_ops = TimeSeriesOperators()

    # 构建测试算子
    test_operators = {
        'sma5': lambda x: ts_ops.sma(x, 5),
        'sma10': lambda x: ts_ops.sma(x, 10),
        'ema5': lambda x: ts_ops.ema(x, 5),
        'std10': lambda x: ts_ops.std(x, 10),
        'delta1': lambda x: ts_ops.delta(x, 1),
        'add': ts_ops.add,
        'sub': ts_ops.sub,
        'mul': ts_ops.mul,
        'div': ts_ops.div,
    }

    print("🧪 测试各个算子的计算...")
    print("-"*80)

    failures = []
    successes = []

    # 测试一元算子
    for op_name, op_func in test_operators.items():
        if op_name in ['add', 'sub', 'mul', 'div']:
            continue  # 跳过二元算子

        try:
            result = op_func(data['close'])

            # 检查结果
            nan_ratio = result.isna().sum() / len(result)
            inf_ratio = np.isinf(result.replace([np.inf, -np.inf], np.nan).fillna(0)).sum() / len(result)

            if nan_ratio > 0.5:
                failures.append({
                    'op': op_name,
                    'reason': f'Too many NaN ({nan_ratio*100:.1f}%)',
                    'result': result
                })
                print(f"❌ {op_name}: NaN比例过高 ({nan_ratio*100:.1f}%)")
            elif inf_ratio > 0.1:
                failures.append({
                    'op': op_name,
                    'reason': f'Too many Inf ({inf_ratio*100:.1f}%)',
                    'result': result
                })
                print(f"❌ {op_name}: Inf比例过高 ({inf_ratio*100:.1f}%)")
            else:
                successes.append(op_name)
                print(f"✅ {op_name}: OK (NaN={nan_ratio*100:.1f}%, mean={result.mean():.4f}, std={result.std():.4f})")

        except Exception as e:
            failures.append({
                'op': op_name,
                'reason': str(e),
                'result': None
            })
            print(f"❌ {op_name}: Exception - {e}")

    print()
    print("🧪 测试二元算子...")
    print("-"*80)

    # 测试二元算子
    try:
        x = data['close']
        y = data['volume']

        for op_name in ['add', 'sub', 'mul', 'div']:
            op_func = test_operators[op_name]
            try:
                result = op_func(x, y)
                nan_ratio = result.isna().sum() / len(result)

                if nan_ratio > 0.5:
                    failures.append({
                        'op': op_name,
                        'reason': f'Too many NaN ({nan_ratio*100:.1f}%)',
                        'result': result
                    })
                    print(f"❌ {op_name}: NaN比例过高 ({nan_ratio*100:.1f}%)")
                else:
                    successes.append(op_name)
                    print(f"✅ {op_name}: OK (NaN={nan_ratio*100:.1f}%)")
            except Exception as e:
                failures.append({
                    'op': op_name,
                    'reason': str(e),
                    'result': None
                })
                print(f"❌ {op_name}: Exception - {e}")
    except Exception as e:
        print(f"❌ 二元算子测试整体失败: {e}")

    print()
    print("="*80)
    print("📊 诊断结果")
    print("="*80)
    print(f"✅ 成功: {len(successes)}")
    print(f"❌ 失败: {len(failures)}")
    print()

    if failures:
        print("失败的算子详情:")
        for f in failures:
            print(f"  - {f['op']}: {f['reason']}")

    return failures

def test_expression_computation():
    """测试表达式计算"""
    print("\n" + "="*80)
    print("🧪 测试完整表达式计算")
    print("="*80)
    print()

    from factor_evaluator import FactorEvaluator
    from operators import TimeSeriesOperators
    from combiner import ImprovedCombinationModel
    from config import TrainingConfig

    # 创建测试数据
    data = create_test_data(1000)

    # 计算目标
    data['future_return'] = data['close'].pct_change(10).shift(-10).fillna(0)

    # 数据分割
    train_size = int(len(data) * 0.6)
    train_data = data.iloc[:train_size].copy()
    val_data = data.iloc[train_size:].copy()

    train_target = train_data['future_return']
    val_target = val_data['future_return']

    # 初始化组件
    config = TrainingConfig()
    ts_ops = TimeSeriesOperators()

    feature_names = ['close', 'volume', 'high', 'low', 'open']

    operators = {
        'sma5': {'arity': 1, 'func': lambda x: ts_ops.sma(x, 5)},
        'add': {'arity': 2, 'func': ts_ops.add},
        'sub': {'arity': 2, 'func': ts_ops.sub},
    }

    combination_model = ImprovedCombinationModel(config=config, max_alpha_count=15)
    combination_model.set_targets(train_target, val_target)

    evaluator = FactorEvaluator(
        operators=operators,
        feature_names=feature_names,
        combination_model=combination_model,
        train_data=train_data,
        val_data=val_data,
        train_target=train_target,
        val_target=val_target
    )

    # 测试表达式
    test_expressions = [
        ['<BEG>', 'close', '<SEP>'],  # 最简单：只有一个特征
        ['<BEG>', 'close', 'sma5', '<SEP>'],  # 简单表达式
        ['<BEG>', 'close', 'volume', 'add', '<SEP>'],  # 二元算子
        ['<BEG>', 'close', 'sma5', 'volume', 'sub', '<SEP>'],  # 组合
    ]

    print("测试表达式:")
    print("-"*80)

    for i, tokens in enumerate(test_expressions, 1):
        expr_str = ' '.join(tokens)
        print(f"\n测试 {i}: {expr_str}")

        result = evaluator.evaluate_expression(tokens, trial_only=True)

        if result['valid']:
            print(f"  ✅ 计算成功")
            print(f"     reward={result['reward']:.6f}")
            print(f"     incremental_sharpe={result.get('incremental_sharpe', 0):.6f}")
        else:
            print(f"  ❌ 计算失败")
            print(f"     原因: {result.get('reason', 'unknown')}")

    print()

def check_data_length():
    """检查数据长度是否足够"""
    print("\n" + "="*80)
    print("🔍 检查数据长度要求")
    print("="*80)
    print()

    print("关键发现：")
    print("-"*80)
    print()

    print("1️⃣ combiner.evaluate_new_factor() 要求:")
    print("   if len(X_train) < 100:")
    print("       return 失败")
    print("   → 训练数据必须 >= 100 行")
    print()

    print("2️⃣ signals.calculate_rolling_sharpe_stability() 要求:")
    print("   if data_length < 150:")
    print("       return 0.0")
    print("   → 数据必须 >= 150 行才能计算Sharpe")
    print()

    print("3️⃣ 滚动窗口算子消耗数据:")
    print("   - sma10: 前10行NaN")
    print("   - sma20: 前20行NaN")
    print("   - std20: 前20行NaN")
    print("   → 复杂表达式可能消耗更多行")
    print()

    print("🔥 关键问题：")
    print("   如果训练数据只有 600 行（0.6 * 1000）")
    print("   经过多个滚动算子后，有效数据可能不足100行！")
    print()

    print("💡 解决方案：")
    print("   1. 降低 combiner 的最小数据要求（100 → 50）")
    print("   2. 降低 Sharpe 的最小数据要求（150 → 80）")
    print("   3. 更积极地填充NaN（forward fill）")
    print()

def main():
    print("\n" + "="*80)
    print("🚨 train_computation_failed 诊断")
    print("="*80)
    print()
    print("现象: 11/16 的因子在训练集计算失败")
    print("影响: 无法积累因子，因子池无法增长")
    print()

    # 测试1: 算子计算
    failures = test_computation()

    # 测试2: 数据长度
    check_data_length()

    # 测试3: 表达式计算
    test_expression_computation()

    print("\n" + "="*80)
    print("🎯 诊断结论")
    print("="*80)
    print()
    print("最可能的3个原因:")
    print()
    print("1️⃣ 数据长度不足（最可能）")
    print("   - combiner要求至少100行")
    print("   - Sharpe计算要求至少150行")
    print("   - 滚动算子消耗大量前置行")
    print("   → 有效数据不足导致计算失败")
    print()
    print("2️⃣ NaN处理过于严格")
    print("   - compute_factor_values中检查 nan_ratio > 0.5")
    print("   - 中间步骤的NaN累积")
    print("   → 表达式被过早拒绝")
    print()
    print("3️⃣ 统计量计算失败")
    print("   - _clean_series 中的统计量计算可能失败")
    print("   - 标准化可能产生NaN")
    print("   → current_factor_stats = None")
    print()

    print("🔧 立即修复方向:")
    print("   1. 降低最小数据要求")
    print("   2. 放宽NaN容忍度")
    print("   3. 改进NaN填充策略")
    print()

if __name__ == "__main__":
    main()
