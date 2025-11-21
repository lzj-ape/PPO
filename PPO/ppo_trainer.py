"""
PPO训练器模块
负责PPO算法的训练和更新
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import logging
from typing import Dict, List, Tuple

try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

logger = logging.getLogger(__name__)


class PPOTrainer:
    """PPO训练器 - 负责策略网络的训练和更新"""

    def __init__(self,
                 actor_critic,
                 ppo_buffer,
                 config,
                 vocab: List[str],
                 token_to_id: Dict[str, int],
                 id_to_token: Dict[int, str],
                 operators: Dict,
                 feature_names: List[str],
                 optimizer,
                 device: torch.device = None,
                 use_amp: bool = False):
        """
        初始化PPO训练器

        Args:
            actor_critic: Actor-Critic网络
            ppo_buffer: PPO缓冲区
            config: 训练配置
            vocab: 词汇表
            token_to_id: token到id的映射
            id_to_token: id到token的映射
            operators: 操作符字典
            feature_names: 特征名称列表
            optimizer: 优化器
            device: 计算设备
            use_amp: 是否使用混合精度
        """
        self.actor_critic = actor_critic
        self.ppo_buffer = ppo_buffer
        self.config = config
        self.vocab = vocab
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.operators = operators
        self.feature_names = feature_names
        self.optimizer = optimizer
        self.device = device or torch.device('cpu')
        self.use_amp = use_amp

        if self.use_amp:
            self.scaler = GradScaler()

    def train_ppo_step(self, get_valid_actions_fn) -> Dict[str, float]:
        """
        执行一次PPO训练步骤

        Args:
            get_valid_actions_fn: 获取有效动作的函数

        Returns:
            训练统计信息
        """
        if len(self.ppo_buffer) < self.config.batch_size:
            return {}

        batch = self.ppo_buffer.get_batch()

        if not batch['states']:
            return {}

        try:
            advantages, returns = self._compute_gae_advantages(
                rewards=batch['rewards'],
                values=batch['values'],
                dones=batch['dones']
            )

            returns = torch.FloatTensor(returns).to(self.device)
            advantages = torch.FloatTensor(advantages).to(self.device)
            old_values = torch.FloatTensor(batch['values']).to(self.device)
            old_log_probs = torch.FloatTensor(batch['log_probs']).to(self.device)
            actions = torch.LongTensor(batch['actions']).to(self.device)

            policy_losses = []
            value_losses = []
            all_entropies = []

            indices = torch.randperm(len(batch['states']))
            batch_size = min(self.config.batch_size, len(indices))

            for epoch in range(self.config.ppo_epochs):
                for start in range(0, len(indices), batch_size):
                    end = min(start + batch_size, len(indices))
                    batch_indices = indices[start:end]

                    if len(batch_indices) == 0:
                        continue

                    batch_states = [batch['states'][i] for i in batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    batch_returns = returns[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_old_values = old_values[batch_indices]

                    # 在mini-batch级别归一化advantage
                    if batch_advantages.std() > 1e-6:
                        batch_advantages = (batch_advantages - batch_advantages.mean()) / \
                                         (batch_advantages.std() + 1e-8)

                    if self.use_amp:
                        with autocast(device_type='cuda'):
                            type_logits_batch, action_logits_batch, values_batch = \
                                self.actor_critic.forward_batch(batch_states)
                    else:
                        type_logits_batch, action_logits_batch, values_batch = \
                            self.actor_critic.forward_batch(batch_states)

                    if action_logits_batch.numel() == 0:
                        continue

                    new_log_probs = []
                    entropies = []

                    LOG_3 = np.log(3)

                    for i, (state, action) in enumerate(zip(batch_states, batch_actions)):
                        valid_types, valid_actions_by_type = get_valid_actions_fn(state)

                        # Type log_prob
                        type_logits = type_logits_batch[i]
                        type_mask = torch.full((3,), float('-inf'), device=self.device)
                        type_mask[valid_types] = 0.0
                        masked_type_logits = type_logits + type_mask

                        type_probs = F.softmax(masked_type_logits, dim=-1)
                        type_dist = Categorical(type_probs)

                        action_token = self.id_to_token[action.item()]
                        if action_token == '<SEP>':
                            action_type = 2
                        elif action_token in self.feature_names:
                            action_type = 0
                        else:
                            action_type = 1

                        type_log_prob = type_dist.log_prob(
                            torch.tensor(action_type, device=self.device)
                        )
                        type_entropy = type_dist.entropy()

                        # Action log_prob
                        valid_actions_for_type = valid_actions_by_type[action_type]
                        action_logits = action_logits_batch[i]

                        action_mask = torch.full((len(self.vocab),), float('-inf'),
                                                device=self.device)
                        action_mask[valid_actions_for_type] = 0.0
                        masked_action_logits = action_logits + action_mask

                        action_probs = F.softmax(masked_action_logits, dim=-1)
                        action_dist = Categorical(action_probs)
                        action_log_prob = action_dist.log_prob(action)
                        action_entropy = action_dist.entropy()

                        combined_log_prob = type_log_prob + action_log_prob

                        # 归一化熵
                        type_entropy_normalized = type_entropy / LOG_3
                        action_space_size = len(valid_actions_for_type)
                        if action_space_size > 1:
                            action_entropy_normalized = action_entropy / np.log(action_space_size)
                        else:
                            action_entropy_normalized = action_entropy
                        combined_entropy = type_entropy_normalized + action_entropy_normalized

                        new_log_probs.append(combined_log_prob)
                        entropies.append(combined_entropy)

                    if not new_log_probs:
                        continue

                    new_log_probs = torch.stack(new_log_probs).float()
                    entropies = torch.stack(entropies).float()

                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    clipped_ratio = torch.clamp(
                        ratio,
                        1 - self.config.clip_param,
                        1 + self.config.clip_param
                    )

                    policy_loss = -torch.min(
                        ratio * batch_advantages,
                        clipped_ratio * batch_advantages
                    ).mean()

                    value_loss_unclipped = F.mse_loss(values_batch, batch_returns)
                    values_clipped = batch_old_values + torch.clamp(
                        values_batch - batch_old_values,
                        -self.config.value_clip_param,
                        self.config.value_clip_param
                    )
                    value_loss_clipped = F.mse_loss(values_clipped, batch_returns)
                    value_loss = torch.max(value_loss_unclipped, value_loss_clipped)

                    entropy_loss = -entropies.mean()

                    total_loss = (
                        policy_loss +
                        self.config.value_coeff * value_loss +
                        self.config.entropy_coeff * entropy_loss
                    )

                    self.optimizer.zero_grad()

                    if self.use_amp:
                        self.scaler.scale(total_loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.actor_critic.parameters(),
                            self.config.max_grad_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.actor_critic.parameters(),
                            self.config.max_grad_norm
                        )
                        self.optimizer.step()

                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    all_entropies.extend([e.item() for e in entropies])

            self.ppo_buffer.clear()

            # 计算平均熵损失
            avg_entropy = np.mean(all_entropies) if all_entropies else 0.0

            return {
                'policy_loss': np.mean(policy_losses) if policy_losses else 0.0,
                'value_loss': np.mean(value_losses) if value_losses else 0.0,
                'entropy_loss': -avg_entropy * self.config.entropy_coeff,  # 注意符号
                'learning_rate': self.config.lr_actor,
                'advantage_mean': float(advantages.mean().item()),
                'advantage_std': float(advantages.std().item()),
                'value_mean': float(old_values.mean().item()),
                'value_std': float(old_values.std().item()),
            }

        except Exception as e:
            logger.warning(f"PPO training error: {e}")
            self.ppo_buffer.clear()
            return {}

    def _compute_gae_advantages(self,
                               rewards: List[float],
                               values: List[float],
                               dones: List[bool]) -> Tuple[List[float], List[float]]:
        """
        计算GAE优势函数

        Args:
            rewards: 奖励列表
            values: 价值估计列表
            dones: 完成标志列表

        Returns:
            (advantages, returns)
        """
        advantages = []
        returns = []

        gae = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[i + 1] if not dones[i] else 0.0

            next_non_terminal = 0.0 if dones[i] else 1.0

            delta = rewards[i] + self.config.gamma * next_value * next_non_terminal - values[i]
            gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])

        return advantages, returns
