"""
添加到你的training notebook中使用的诊断工具

使用方法:
--------
在notebook中:
```python
from diagnose_utils import diagnose_failed_expressions

# 在miner训练循环中,当发现计算失败时:
diagnose_failed_expressions(
    failed_tokens=tokens_list,  # 失败的表达式tokens
    miner=miner  # 你的FactorMinerCore实例
)
```
"""

import logging
from typing import List, Dict
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def check_rpn_balance(tokens: List[str], operators: Dict, feature_names: List[str]) -> dict:
    """
    检查RPN表达式的栈平衡

    Returns:
        dict: {
            'is_valid': bool,
            'final_stack_size': int,
            'error_message': str or None,
            'error_position': int or None
        }
    """
    if len(tokens) < 3:
        return {
            'is_valid': False,
            'final_stack_size': 0,
            'error_message': 'Expression too short',
            'error_position': 0
        }

    if tokens[0] != '<BEG>' or tokens[-1] != '<SEP>':
        return {
            'is_valid': False,
            'final_stack_size': 0,
            'error_message': 'Missing <BEG> or <SEP>',
            'error_position': 0
        }

    expr_tokens = tokens[1:-1]
    stack = 0

    for i, token in enumerate(expr_tokens):
        if token in feature_names:
            stack += 1
        elif token in operators:
            arity = operators[token]['arity']
            if stack < arity:
                return {
                    'is_valid': False,
                    'final_stack_size': stack,
                    'error_message': f'Stack underflow at token "{token}"',
                    'error_position': i + 1  # +1 for <BEG>
                }
            stack = stack - arity + 1
        else:
            return {
                'is_valid': False,
                'final_stack_size': stack,
                'error_message': f'Unknown token "{token}"',
                'error_position': i + 1
            }

    is_valid = (stack == 1)
    return {
        'is_valid': is_valid,
        'final_stack_size': stack,
        'error_message': None if is_valid else f'Final stack size is {stack}, expected 1',
        'error_position': None
    }


def try_compute_factor(tokens: List[str], data: pd.DataFrame,
                       feature_names: List[str], operators: Dict) -> dict:
    """
    尝试计算因子,返回详细的诊断信息

    Returns:
        dict: {
            'success': bool,
            'result': pd.Series or None,
            'valid_ratio': float,
            'nan_ratio': float,
            'inf_ratio': float,
            'error_message': str or None,
            'failed_at_token': str or None
        }
    """
    expr_tokens = tokens[1:-1]
    stack = []

    try:
        for token in expr_tokens:
            if token in feature_names:
                stack.append(data[token].copy())

            elif token in operators:
                op_info = operators[token]
                arity = op_info['arity']
                func = op_info['func']

                if len(stack) < arity:
                    return {
                        'success': False,
                        'result': None,
                        'valid_ratio': 0.0,
                        'nan_ratio': 1.0,
                        'inf_ratio': 0.0,
                        'error_message': f'Stack underflow at operator {token}',
                        'failed_at_token': token
                    }

                operands = [stack.pop() for _ in range(arity)]
                operands.reverse()

                result = func(*operands)

                if result is None:
                    return {
                        'success': False,
                        'result': None,
                        'valid_ratio': 0.0,
                        'nan_ratio': 1.0,
                        'inf_ratio': 0.0,
                        'error_message': f'Operator {token} returned None',
                        'failed_at_token': token
                    }

                stack.append(result)

        if len(stack) != 1:
            return {
                'success': False,
                'result': None,
                'valid_ratio': 0.0,
                'nan_ratio': 1.0,
                'inf_ratio': 0.0,
                'error_message': f'Final stack size {len(stack)} != 1',
                'failed_at_token': None
            }

        final_result = stack[0]
        total_len = len(final_result)
        nan_count = final_result.isna().sum()
        inf_count = np.isinf(final_result).sum()
        valid_count = total_len - nan_count - inf_count

        return {
            'success': True,
            'result': final_result,
            'valid_ratio': valid_count / total_len,
            'nan_ratio': nan_count / total_len,
            'inf_ratio': inf_count / total_len,
            'error_message': None,
            'failed_at_token': None
        }

    except Exception as e:
        return {
            'success': False,
            'result': None,
            'valid_ratio': 0.0,
            'nan_ratio': 1.0,
            'inf_ratio': 0.0,
            'error_message': str(e),
            'failed_at_token': token if 'token' in locals() else None
        }


def diagnose_failed_expressions(failed_tokens_list: List[List[str]], miner) -> None:
    """
    诊断失败的表达式列表

    Args:
        failed_tokens_list: 失败的表达式tokens列表
        miner: FactorMinerCore实例
    """
    logger.info("="*80)
    logger.info("🔍 开始诊断失败的表达式")
    logger.info("="*80)

    if not failed_tokens_list:
        logger.info("没有失败的表达式需要诊断")
        return

    logger.info(f"失败表达式数量: {len(failed_tokens_list)}")

    # 统计
    balance_failures = 0
    computation_failures = 0
    low_quality_results = 0

    for idx, tokens in enumerate(failed_tokens_list[:10]):  # 只诊断前10个
        logger.info(f"\n{'='*80}")
        logger.info(f"表达式 {idx+1}/{len(failed_tokens_list)}")
        logger.info(f"Tokens: {' '.join(tokens)}")

        # 1. 检查栈平衡
        balance_result = check_rpn_balance(
            tokens,
            miner.operators,
            miner.feature_names
        )

        if not balance_result['is_valid']:
            logger.error(f"❌ 栈平衡检查失败:")
            logger.error(f"   {balance_result['error_message']}")
            if balance_result['error_position'] is not None:
                logger.error(f"   错误位置: 第{balance_result['error_position']}个token")
            logger.error(f"   最终栈大小: {balance_result['final_stack_size']}")
            balance_failures += 1
            continue

        logger.info(f"✅ 栈平衡检查通过 (栈大小=1)")

        # 2. 尝试计算
        compute_result = try_compute_factor(
            tokens,
            miner.train_data,
            miner.feature_names,
            miner.operators
        )

        if not compute_result['success']:
            logger.error(f"❌ 计算失败:")
            logger.error(f"   {compute_result['error_message']}")
            if compute_result['failed_at_token']:
                logger.error(f"   失败于token: {compute_result['failed_at_token']}")
            computation_failures += 1
            continue

        # 3. 检查结果质量
        valid_ratio = compute_result['valid_ratio']
        nan_ratio = compute_result['nan_ratio']
        inf_ratio = compute_result['inf_ratio']

        logger.info(f"✅ 计算成功:")
        logger.info(f"   有效率: {valid_ratio*100:.1f}%")
        logger.info(f"   NaN率: {nan_ratio*100:.1f}%")
        logger.info(f"   Inf率: {inf_ratio*100:.1f}%")

        if valid_ratio < 0.5:
            logger.warning(f"⚠️  结果质量低 (有效率 < 50%)")
            low_quality_results += 1

        if compute_result['result'] is not None:
            result = compute_result['result']
            logger.info(f"   均值: {result.mean():.4f}")
            logger.info(f"   标准差: {result.std():.4f}")

    # 输出总结
    logger.info(f"\n{'='*80}")
    logger.info("📊 诊断总结")
    logger.info(f"{'='*80}")
    logger.info(f"栈平衡失败: {balance_failures}/{len(failed_tokens_list)}")
    logger.info(f"计算失败: {computation_failures}/{len(failed_tokens_list)}")
    logger.info(f"低质量结果: {low_quality_results}/{len(failed_tokens_list)}")
    logger.info(f"{'='*80}\n")


def diagnose_single_expression(tokens: List[str], miner, verbose: bool = True) -> dict:
    """
    诊断单个表达式

    Returns:
        dict: 包含所有诊断信息
    """
    result = {
        'tokens': tokens,
        'balance_check': None,
        'computation': None
    }

    # 检查栈平衡
    result['balance_check'] = check_rpn_balance(
        tokens,
        miner.operators,
        miner.feature_names
    )

    if not result['balance_check']['is_valid']:
        if verbose:
            logger.error(f"栈平衡失败: {result['balance_check']['error_message']}")
        return result

    # 尝试计算
    result['computation'] = try_compute_factor(
        tokens,
        miner.train_data,
        miner.feature_names,
        miner.operators
    )

    if verbose:
        if result['computation']['success']:
            logger.info(f"✅ 表达式有效: {' '.join(tokens)}")
            logger.info(f"   有效率: {result['computation']['valid_ratio']*100:.1f}%")
        else:
            logger.error(f"❌ 计算失败: {result['computation']['error_message']}")

    return result


# 使用示例
"""
# 在你的训练循环中:

# 1. 收集失败的表达式
failed_expressions = []

# 在batch处理中,当发现失败时:
for tokens, state_ids, trajectory in batch_results:
    try:
        factor_values = miner.factor_evaluator.compute_factor_train(tokens)
        if factor_values is None:
            failed_expressions.append(tokens)
    except:
        failed_expressions.append(tokens)

# 2. 批量诊断
if len(failed_expressions) > 0:
    from diagnose_utils import diagnose_failed_expressions
    diagnose_failed_expressions(failed_expressions, miner)

# 3. 或者诊断单个表达式
from diagnose_utils import diagnose_single_expression
result = diagnose_single_expression(['<BEG>', 'close', 'sma5', '<SEP>'], miner)
"""
