"""
表达式生成器模块
负责生成和验证因子表达式
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional
import logging

try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)


class ExpressionGenerator:
    """表达式生成器 - 使用PPO策略网络生成因子表达式"""

    def __init__(self,
                 actor_critic,
                 vocab: List[str],
                 token_to_id: Dict[str, int],
                 id_to_token: Dict[int, str],
                 operators: Dict,
                 feature_names: List[str],
                 feature_scales: Dict[str, float],
                 max_expr_len: int = 20,
                 device: torch.device = None,
                 use_amp: bool = False):
        """
        初始化表达式生成器

        Args:
            actor_critic: PPO的Actor-Critic网络
            vocab: 词汇表
            token_to_id: token到id的映射
            id_to_token: id到token的映射
            operators: 操作符字典
            feature_names: 特征名称列表
            feature_scales: 特征数量级字典
            max_expr_len: 最大表达式长度
            device: 计算设备
            use_amp: 是否使用混合精度
        """
        self.actor_critic = actor_critic
        self.vocab = vocab
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.operators = operators
        self.feature_names = feature_names
        self.feature_scales = feature_scales
        self.max_expr_len = max_expr_len
        self.device = device or torch.device('cpu')
        self.use_amp = use_amp

        self.pad_token_id = token_to_id['<PAD>']

    def generate_expression_batch(self, batch_size: int = 8) -> List[Tuple[List[str], List[int], Dict]]:
        """
        批量生成表达式

        Args:
            batch_size: 批大小

        Returns:
            List[(tokens, state_ids, trajectory)]
        """
        batch_states = [[self.token_to_id['<BEG>']] for _ in range(batch_size)]
        batch_tokens = [['<BEG>'] for _ in range(batch_size)]
        batch_finished = [False] * batch_size
        batch_trajectories = [
            {
                'states': [],
                'actions': [],
                'types': [],
                'type_log_probs': [],
                'action_log_probs': [],
                'values': []
            }
            for _ in range(batch_size)
        ]

        for step in range(self.max_expr_len - 1):
            active_indices = [i for i in range(batch_size) if not batch_finished[i]]

            if not active_indices:
                break

            active_states = [batch_states[i] for i in active_indices]
            active_valid_info = [self._get_valid_actions(state) for state in active_states]

            with torch.no_grad():
                if self.use_amp:
                    with autocast(device_type='cuda'):
                        type_logits_batch, action_logits_batch, values_batch = \
                            self.actor_critic.forward_batch(active_states)
                else:
                    type_logits_batch, action_logits_batch, values_batch = \
                        self.actor_critic.forward_batch(active_states)

            for idx, i in enumerate(active_indices):
                valid_types, valid_actions_by_type = active_valid_info[idx]

                if len(valid_types) == 1 and valid_types[0] == 2:
                    batch_tokens[i].append('<SEP>')
                    batch_finished[i] = True
                    continue

                value = values_batch[idx]

                # Step 1: 采样动作类型
                type_logits = type_logits_batch[idx]
                type_mask = torch.full((3,), float('-inf'), device=self.device)
                type_mask[valid_types] = 0.0
                masked_type_logits = type_logits + type_mask

                type_probs = F.softmax(masked_type_logits, dim=-1)
                type_dist = Categorical(type_probs)
                action_type_tensor = type_dist.sample()
                action_type = action_type_tensor.item()
                type_log_prob = type_dist.log_prob(action_type_tensor).item()

                # Step 2: 采样具体动作
                valid_actions_for_type = valid_actions_by_type[action_type]
                action_logits = action_logits_batch[idx]

                action_mask = torch.full((len(self.vocab),), float('-inf'), device=self.device)
                action_mask[valid_actions_for_type] = 0.0
                masked_action_logits = action_logits + action_mask

                action_probs = F.softmax(masked_action_logits, dim=-1)
                action_dist = Categorical(action_probs)
                action_tensor = action_dist.sample()
                action = action_tensor.item()
                action_log_prob = action_dist.log_prob(action_tensor).item()

                # 保存trajectory
                batch_trajectories[i]['states'].append(batch_states[i].copy())
                batch_trajectories[i]['types'].append(action_type)
                batch_trajectories[i]['actions'].append(action)
                batch_trajectories[i]['type_log_probs'].append(type_log_prob)
                batch_trajectories[i]['action_log_probs'].append(action_log_prob)
                batch_trajectories[i]['values'].append(value.item())

                token = self.id_to_token[action]
                batch_tokens[i].append(token)
                batch_states[i].append(action)

                if token == '<SEP>':
                    batch_finished[i] = True

        results = []
        for i in range(batch_size):
            results.append((batch_tokens[i], batch_states[i], batch_trajectories[i]))

        return results

    def _get_valid_actions(self, state: List[int]) -> Tuple[List[int], Dict[int, List[int]]]:
        """
        层次化动作选择 - 增加数量级检查

        Returns:
            (valid_types, valid_actions_by_type)
        """
        if len(state) >= self.max_expr_len:
            return [2], {2: [self.token_to_id['<SEP>']]}

        stack_size = self._calculate_stack_size(state)
        scale_stack = self._get_scale_stack(state)

        valid_types = []
        valid_actions_by_type = {}

        if stack_size >= 0:
            # Type 0: Features
            feature_actions = [self.token_to_id[f] for f in self.feature_names]
            if feature_actions:
                valid_types.append(0)
                valid_actions_by_type[0] = feature_actions

            # Type 1: Operators - 增加数量级检查
            operator_actions = []
            for op_name, op_info in self.operators.items():
                # 检查栈大小
                if stack_size >= op_info['arity']:
                    # 检查数量级兼容性
                    if self._is_operator_scale_compatible(op_name, scale_stack):
                        operator_actions.append(self.token_to_id[op_name])

            if operator_actions:
                valid_types.append(1)
                valid_actions_by_type[1] = operator_actions

        # Type 2: End
        if stack_size == 1:
            valid_types.append(2)
            valid_actions_by_type[2] = [self.token_to_id['<SEP>']]

        if not valid_types:
            return [2], {2: [self.token_to_id['<SEP>']]}

        return valid_types, valid_actions_by_type

    def _calculate_stack_size(self, state: List[int]) -> int:
        """计算表达式栈的大小"""
        if len(state) <= 1:
            return 0

        stack = 0
        for token_id in state[1:]:
            token = self.id_to_token[token_id]

            if token in self.feature_names:
                stack += 1
            elif token in self.operators:
                arity = self.operators[token]['arity']
                stack = stack - arity + 1
                if stack < 0:
                    return -1
            elif token == '<SEP>':
                break

        return stack

    def _get_scale_stack(self, state: List[int]) -> List[float]:
        """获取当前表达式的数量级栈"""
        scale_stack = []

        for token_id in state[1:]:  # 跳过<BEG>
            token = self.id_to_token[token_id]

            if token in self.feature_names:
                scale = self.feature_scales.get(token, 1.0)
                scale_stack.append(scale)

            elif token in self.operators:
                op_info = self.operators[token]
                arity = op_info['arity']

                if len(scale_stack) < arity:
                    return []

                operand_scales = []
                for _ in range(arity):
                    operand_scales.append(scale_stack.pop())
                operand_scales.reverse()

                # 计算结果数量级
                result_scale = self._compute_result_scale(token, operand_scales)
                scale_stack.append(result_scale)

            elif token == '<SEP>':
                break

        return scale_stack

    def _compute_result_scale(self, op_name: str, operand_scales: List[float]) -> float:
        """计算操作结果的数量级"""
        arity = len(operand_scales)

        if arity == 1:
            # 归一化到[0,1]的一元算子
            if op_name in ['sigmoid', 'rank', 'ts_rank10']:
                return 1.0
            # 归一化到[-1,1]的一元算子
            elif op_name in ['tanh', 'sign']:
                return 1.0
            # RSI等固定范围算子
            elif op_name in ['rsi14']:
                return 50.0
            # 保持数量级的算子
            elif op_name in ['abs', 'delay1', 'delay3']:
                return operand_scales[0]
            # 改变数量级为无量纲的算子
            elif op_name in ['log', 'exp']:
                return 1.0
            # 相对变化类算子
            elif op_name in ['delta1', 'momentum5', 'roc10']:
                return operand_scales[0] * 0.1
            # 平滑类算子
            elif op_name in ['sma5', 'sma10', 'sma20', 'ema5', 'ema10',
                            'wma10', 'dema10', 'tema10']:
                return operand_scales[0]
            # 波动率类算子
            elif op_name in ['std10', 'std20', 'variance20', 'mad20']:
                return operand_scales[0]
            # 分位数
            elif op_name in ['quantile20', 'ts_min10', 'ts_max10']:
                return operand_scales[0]
            # 幂、平方根
            elif op_name in ['pow', 'sqrt']:
                return operand_scales[0]
            # 标准化类算子
            elif op_name in ['zscore20', 'scale']:
                return 1.0
            # 技术指标
            elif op_name in ['macd', 'bb_upper', 'bb_lower']:
                return operand_scales[0]
            # 加权类算子
            elif op_name in ['decay10']:
                return operand_scales[0]
            # 连乘算子
            elif op_name in ['ts_prod10']:
                return 1.0
            else:
                return operand_scales[0]

        elif arity == 2:
            if op_name in ['add', 'sub', 'max', 'min']:
                return np.mean(operand_scales)
            elif op_name == 'mul':
                return operand_scales[0] * operand_scales[1]
            elif op_name == 'div':
                if operand_scales[1] > 1e-10:
                    return operand_scales[0] / operand_scales[1]
                else:
                    return operand_scales[0]
            elif op_name in ['corr20', 'covar']:
                return 1.0
            else:
                return np.mean(operand_scales)

        elif arity == 3:
            if op_name == 'condition':
                return np.mean([operand_scales[1], operand_scales[2]])
            else:
                return np.mean(operand_scales)

        else:
            return np.mean(operand_scales) if operand_scales else 1.0

    def _is_operator_scale_compatible(self, op_name: str, scale_stack: List[float]) -> bool:
        """检查操作符是否与当前数量级栈兼容"""
        if op_name not in self.operators:
            return False

        op_info = self.operators[op_name]
        arity = op_info['arity']
        scale_rule = op_info.get('scale_rule', 'any')
        scale_threshold = op_info.get('scale_threshold', None)

        if len(scale_stack) < arity:
            return False

        if scale_rule == 'any':
            return True

        if scale_rule == 'similar_only':
            if arity == 2:
                scale1 = scale_stack[-2]
                scale2 = scale_stack[-1]

                if scale1 == 0 or scale2 == 0:
                    return scale1 == scale2

                ratio = max(scale1, scale2) / min(scale1, scale2)
                threshold = scale_threshold if scale_threshold is not None else 100.0
                return ratio <= threshold
            return True

        return True

    def tokens_to_expression(self, tokens: List[str]) -> str:
        """将RPN格式的tokens转换为可读表达式"""
        if not tokens or tokens[0] != '<BEG>' or tokens[-1] != '<SEP>':
            return 'INVALID_EXPRESSION'

        stack: List[str] = []
        expr_tokens = tokens[1:-1]

        try:
            for token in expr_tokens:
                if token in self.feature_names:
                    stack.append(token)
                elif token in self.operators:
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
                    return 'INVALID_EXPRESSION'

            if len(stack) != 1:
                return 'INVALID_EXPRESSION'

            return stack[0]
        except Exception as exc:
            logger.debug(f"Expression stringify error: {exc}")
            return 'INVALID_EXPRESSION'
