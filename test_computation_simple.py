"""
简化版计算失败诊断测试
专注于最核心的问题：为什么表达式计算会失败
"""

import numpy as np
import pandas as pd
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from factor.operators import TimeSeriesOperators

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_basic_operator_issues():
    """测试基础操作符是否有问题"""

    logger.info("="*80)
    logger.info("测试1: 基础操作符计算")
    logger.info("="*80)

    # 生成测试数据
    np.random.seed(42)
    n = 300
    test_data = pd.DataFrame({
        'close': np.random.randn(n).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, n),
    })

    ts_ops = TimeSeriesOperators()

    # 测试各种操作符
    operators_to_test = [
        ('sma5', lambda: ts_ops.sma(test_data['close'], 5)),
        ('sma10', lambda: ts_ops.sma(test_data['close'], 10)),
        ('sma20', lambda: ts_ops.sma(test_data['close'], 20)),
        ('std10', lambda: ts_ops.std(test_data['close'], 10)),
        ('std20', lambda: ts_ops.std(test_data['close'], 20)),
        ('delta1', lambda: ts_ops.delta(test_data['close'], 1)),
        ('rank', lambda: ts_ops.rank(test_data['close'])),
        ('zscore20', lambda: ts_ops.zscore(test_data['close'], 20)),
        ('rsi14', lambda: ts_ops.rsi(test_data['close'], 14)),
        ('abs', lambda: ts_ops.abs_op(test_data['close'])),
        ('add', lambda: ts_ops.add(test_data['close'], test_data['volume'])),
        ('div', lambda: ts_ops.div(test_data['close'], test_data['volume'])),
    ]

    success_count = 0
    fail_count = 0

    for op_name, op_func in operators_to_test:
        try:
            result = op_func()

            # 检查结果质量
            total_len = len(result)
            nan_count = result.isna().sum()
            inf_count = np.isinf(result).sum()
            valid_count = total_len - nan_count - inf_count

            valid_ratio = valid_count / total_len

            if valid_ratio >= 0.5:
                logger.info(f"✅ {op_name:15s}: valid={valid_ratio*100:5.1f}%, "
                          f"mean={result.mean():8.4f}, std={result.std():8.4f}")
                success_count += 1
            else:
                logger.warning(f"⚠️  {op_name:15s}: valid={valid_ratio*100:5.1f}% (TOO LOW)")
                fail_count += 1

        except Exception as e:
            logger.error(f"❌ {op_name:15s}: {e}")
            fail_count += 1

    logger.info(f"\n结果: 成功={success_count}, 失败={fail_count}")


def test_rpn_expression_computation():
    """测试RPN表达式的端到端计算"""

    logger.info("\n" + "="*80)
    logger.info("测试2: RPN表达式端到端计算")
    logger.info("="*80)

    # 生成测试数据
    np.random.seed(42)
    n = 300
    data = pd.DataFrame({
        'open': np.random.randn(n).cumsum() + 100,
        'high': np.random.randn(n).cumsum() + 102,
        'low': np.random.randn(n).cumsum() + 98,
        'close': np.random.randn(n).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, n),
    })

    ts_ops = TimeSeriesOperators()
    feature_names = ['open', 'high', 'low', 'close', 'volume']

    # 构建简化的operators字典
    operators = {
        'sma5': {'arity': 1, 'func': lambda x: ts_ops.sma(x, 5)},
        'sma10': {'arity': 1, 'func': lambda x: ts_ops.sma(x, 10)},
        'std20': {'arity': 1, 'func': lambda x: ts_ops.std(x, 20)},
        'delta1': {'arity': 1, 'func': lambda x: ts_ops.delta(x, 1)},
        'rank': {'arity': 1, 'func': ts_ops.rank},
        'abs': {'arity': 1, 'func': ts_ops.abs_op},
        'add': {'arity': 2, 'func': ts_ops.add},
        'sub': {'arity': 2, 'func': ts_ops.sub},
        'div': {'arity': 2, 'func': ts_ops.div},
    }

    # 测试表达式
    test_expressions = [
        (['<BEG>', 'close', 'sma5', '<SEP>'], "简单平滑"),
        (['<BEG>', 'close', 'sma5', 'close', 'sma10', 'sub', '<SEP>'], "双均线差"),
        (['<BEG>', 'close', 'delta1', 'abs', '<SEP>'], "绝对变化"),
        (['<BEG>', 'high', 'low', 'sub', 'close', 'div', '<SEP>'], "价格范围比率"),
        (['<BEG>', 'close', 'std20', 'rank', '<SEP>'], "波动率排名"),
    ]

    success_count = 0
    fail_count = 0

    for tokens, description in test_expressions:
        logger.info(f"\n测试: {description}")
        logger.info(f"  Tokens: {' '.join(tokens)}")

        try:
            # 计算因子值
            result = compute_factor_from_rpn(tokens, data, feature_names, operators)

            # 检查结果
            total_len = len(result)
            nan_count = result.isna().sum()
            inf_count = np.isinf(result).sum()
            valid_count = total_len - nan_count - inf_count
            valid_ratio = valid_count / total_len

            if valid_ratio >= 0.5:
                logger.info(f"  ✅ 成功: valid={valid_ratio*100:.1f}%, "
                          f"mean={result.mean():.4f}, std={result.std():.4f}")
                success_count += 1
            else:
                logger.warning(f"  ⚠️  低质量: valid={valid_ratio*100:.1f}%")
                fail_count += 1

        except Exception as e:
            logger.error(f"  ❌ 失败: {e}")
            fail_count += 1

    logger.info(f"\n结果: 成功={success_count}, 失败={fail_count}")


def compute_factor_from_rpn(tokens, data, feature_names, operators):
    """从RPN tokens计算因子值"""
    expr_tokens = tokens[1:-1]  # 去除<BEG>和<SEP>

    stack = []

    for token in expr_tokens:
        if token in feature_names:
            stack.append(data[token].copy())
        elif token in operators:
            op_info = operators[token]
            arity = op_info['arity']
            func = op_info['func']

            if len(stack) < arity:
                raise ValueError(f"Stack underflow for {token}")

            operands = [stack.pop() for _ in range(arity)]
            operands.reverse()

            result = func(*operands)
            stack.append(result)
        else:
            raise ValueError(f"Unknown token: {token}")

    if len(stack) != 1:
        raise ValueError(f"Final stack size {len(stack)} != 1")

    return stack[0]


def test_data_length_impact():
    """测试数据长度对计算成功率的影响"""

    logger.info("\n" + "="*80)
    logger.info("测试3: 数据长度影响")
    logger.info("="*80)

    ts_ops = TimeSeriesOperators()

    # 测试不同数据长度
    data_lengths = [50, 100, 200, 500]

    for n in data_lengths:
        np.random.seed(42)
        data = pd.DataFrame({
            'close': np.random.randn(n).cumsum() + 100,
        })

        logger.info(f"\n数据长度: {n}")

        # 测试需要不同窗口的操作符
        tests = [
            ('sma5', lambda: ts_ops.sma(data['close'], 5)),
            ('sma20', lambda: ts_ops.sma(data['close'], 20)),
            ('std10', lambda: ts_ops.std(data['close'], 10)),
            ('std20', lambda: ts_ops.std(data['close'], 20)),
        ]

        for op_name, op_func in tests:
            try:
                result = op_func()
                valid_ratio = (~result.isna()).sum() / len(result)
                logger.info(f"  {op_name:10s}: valid={valid_ratio*100:5.1f}%")
            except Exception as e:
                logger.error(f"  {op_name:10s}: {e}")


def test_edge_cases():
    """测试边界情况"""

    logger.info("\n" + "="*80)
    logger.info("测试4: 边界情况")
    logger.info("="*80)

    ts_ops = TimeSeriesOperators()

    # 边界情况1: 包含大量NaN的数据
    logger.info("\n边界1: 含NaN的数据")
    data_with_nan = pd.Series(np.random.randn(100))
    data_with_nan.iloc[20:40] = np.nan

    try:
        result = ts_ops.sma(data_with_nan, 10)
        valid_ratio = (~result.isna()).sum() / len(result)
        logger.info(f"  sma10 on data with NaN: valid={valid_ratio*100:.1f}%")
    except Exception as e:
        logger.error(f"  sma10 failed: {e}")

    # 边界情况2: 全0数据
    logger.info("\n边界2: 全0数据")
    zero_data = pd.Series(np.zeros(100))

    try:
        result = ts_ops.div(pd.Series(np.ones(100)), zero_data)
        inf_count = np.isinf(result).sum()
        logger.info(f"  div by zero: {inf_count} Inf values")
    except Exception as e:
        logger.error(f"  div by zero failed: {e}")

    # 边界情况3: 极小数据集
    logger.info("\n边界3: 极小数据集(n=10)")
    tiny_data = pd.Series(np.random.randn(10))

    for window in [5, 10, 20]:
        try:
            result = ts_ops.sma(tiny_data, window)
            valid_ratio = (~result.isna()).sum() / len(result)
            logger.info(f"  sma{window} on n=10: valid={valid_ratio*100:.1f}%")
        except Exception as e:
            logger.error(f"  sma{window} failed: {e}")


if __name__ == "__main__":
    logger.info("🔍 开始简化诊断测试\n")

    test_basic_operator_issues()
    test_rpn_expression_computation()
    test_data_length_impact()
    test_edge_cases()

    logger.info("\n" + "="*80)
    logger.info("✅ 所有测试完成")
    logger.info("="*80)
