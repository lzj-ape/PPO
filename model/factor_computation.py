"""
因子计算工具模块 - 提供统一的因子值计算接口
用于避免循环导入问题，提供模块化的因子计算功能
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class FactorComputer:
    """因子计算器 - 从RPN表达式计算因子值"""

    def __init__(self, operators: Dict, feature_names: List[str] = None):
        """
        初始化因子计算器

        Args:
            operators: 操作符字典，格式为 {op_name: {'arity': int, 'func': callable}}
            feature_names: 特征名称列表（可选，用于验证）
        """
        self.operators = operators
        self.feature_names = feature_names or []

    def compute_factor_from_tokens(self, tokens: List[str], data: pd.DataFrame) -> Optional[pd.Series]:
        """
        从RPN格式的tokens计算因子值

        Args:
            tokens: RPN格式的token列表，如 ['<BEG>', 'close', 'sma10', '<SEP>']
            data: 数据DataFrame

        Returns:
            因子值Series，失败返回None
        """
        if not tokens or tokens[0] != '<BEG>' or tokens[-1] != '<SEP>':
            logger.debug(f"Invalid token format: {tokens[:3]}...{tokens[-3:]}")
            return None

        try:
            stack = []
            expr_tokens = tokens[1:-1]  # 去掉 <BEG> 和 <SEP>

            for token in expr_tokens:
                if token in data.columns:
                    # 特征
                    stack.append(data[token].copy())

                elif token in self.operators:
                    # 操作符
                    op_info = self.operators[token]
                    arity = op_info['arity']

                    if len(stack) < arity:
                        logger.debug(f"Stack underflow for operator {token}: need {arity}, have {len(stack)}")
                        return None

                    # 弹出参数
                    args = [stack.pop() for _ in range(arity)]
                    args.reverse()

                    # 执行操作
                    try:
                        result = op_info['func'](*args)
                    except Exception as e:
                        logger.debug(f"Operator {token} execution failed: {e}")
                        return None

                    # 清理结果
                    result = result.replace([np.inf, -np.inf], np.nan)
                    result = result.ffill().fillna(0)

                    stack.append(result)

                else:
                    logger.debug(f"Unknown token: {token}")
                    return None

            if len(stack) != 1:
                logger.debug(f"Invalid expression: stack size = {len(stack)}")
                return None

            return stack[0]

        except Exception as e:
            logger.debug(f"Factor computation error: {e}")
            return None

    def compute_factor_matrix(self, alpha_pool: List[Dict], data: pd.DataFrame) -> pd.DataFrame:
        """
        计算因子矩阵

        Args:
            alpha_pool: 因子池，每个元素包含 'tokens' 键
            data: 数据DataFrame

        Returns:
            因子矩阵DataFrame，列名为 factor_0, factor_1, ...
        """
        factor_dict = {}

        for i, alpha_info in enumerate(alpha_pool):
            tokens = alpha_info.get('tokens', [])
            factor = self.compute_factor_from_tokens(tokens, data)

            if factor is None:
                logger.warning(f"Factor {i} computation failed, using zeros")
                factor_dict[f'factor_{i}'] = pd.Series(0.0, index=data.index)
            else:
                factor_dict[f'factor_{i}'] = factor

        return pd.DataFrame(factor_dict)

    def tokens_to_expression(self, tokens: List[str]) -> str:
        """
        将RPN格式的tokens转换为可读表达式

        Args:
            tokens: RPN格式的token列表

        Returns:
            可读的表达式字符串
        """
        if not tokens or tokens[0] != '<BEG>' or tokens[-1] != '<SEP>':
            return 'INVALID_EXPRESSION'

        stack: List[str] = []
        expr_tokens = tokens[1:-1]

        try:
            for token in expr_tokens:
                if token in self.operators:
                    op_info = self.operators[token]
                    arity = op_info['arity']
                    if len(stack) < arity:
                        return 'INVALID_EXPRESSION'

                    args = [stack.pop() for _ in range(arity)]
                    args.reverse()
                    arg_str = ', '.join(args)
                    expr = f"{token}({arg_str})"
                    stack.append(expr)
                else:
                    # 特征名
                    stack.append(token)

            if len(stack) != 1:
                return 'INVALID_EXPRESSION'

            return stack[0]

        except Exception as e:
            logger.debug(f"Expression stringify error: {e}")
            return 'INVALID_EXPRESSION'


# 便捷函数，提供与旧接口兼容的调用方式
def compute_factor_from_tokens(tokens: List[str], data: pd.DataFrame, operators: Dict) -> Optional[pd.Series]:
    """
    便捷函数：从RPN格式的tokens计算因子值

    Args:
        tokens: RPN格式的token列表
        data: 数据DataFrame
        operators: 操作符字典

    Returns:
        因子值Series，失败返回None
    """
    computer = FactorComputer(operators)
    return computer.compute_factor_from_tokens(tokens, data)
