

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """训练配置 - 实盘优化版 v2"""
    # 🔥 模型参数（恢复正常学习率，避免策略不更新）
    lr_actor: float = 3e-4       # 恢复到正常学习率
    lr_critic: float = 3e-4      # 恢复到正常学习率
    hidden_dim: int = 128
    lstm_layers: int = 2
    batch_size: int = 32
    ppo_epochs: int = 4
    clip_param: float = 0.2
    value_clip_param: float = 1.0  # 从0.2提升到1.0（配合奖励归一化）
    entropy_coeff: float = 0.05    # 提升到0.05以增加探索，避免只生成简单因子
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5
    gamma: float = 0.99            # 从0.95提升到0.99（更重视早期步骤）
    gae_lambda: float = 0.95
    dropout: float = 0.1
    buffer_size: int = 2048        # 从1024增加到2048
    # 因子组合器参数
    combiner_type: str = 'linear'  # 'linear' or 'lstm'
    combiner_lr: float = 1e-3
    combiner_hidden_dim: int = 64
    combiner_lstm_layers: int = 1
    combiner_train_interval: int = 10  # 每N次迭代训练一次组合器
    
    # 防过拟合参数（LSTM专用）
    combiner_patience: int = 15  # Early stopping patience
    combiner_weight_decay: float = 1e-3  # L2正则化
    combiner_dropout: float = 0.3  # Dropout比例
    
    # 数据参数
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    prediction_horizon: int = 10
    bar_minutes: int = 15
    
    # 交易参数
    transaction_cost: float = 0.0005
    max_position: float = 1.0
    rebalance_mode: str = 'non_overlapping'
    
    # 因子筛选阈值
    # 🔥 增量Sharpe阈值：只有带来实质性改进的因子才被接受
    #
    # 新策略（2025-01）：
    # - 使用自适应阈值：前期低阈值快速积累因子，后期高阈值精选因子
    # - 统一使用增量Sharpe（经过linear优化后的组合改进）作为评价标准
    # - 前3个因子：ic_threshold = -0.03（允许轻微负值）
    # - 第4-5个：ic_threshold = 0.001（0.1%增量）
    # - 第6-10个：ic_threshold = base * 0.3（0.3%增量）
    # - 10个以上：ic_threshold = base * 0.6（0.6%增量）
    ic_threshold: float = 0.01  # 基础阈值（用于后期）

    # 🔥 高级Reward配置（新增）
    reward_type: str = 'hybrid'  # 'incremental', 'penalized', 'stable', 'hybrid', 'full'

    # 方案一：增量Sharpe参数
    incremental_weight: float = 5.0  # 增量Sharpe权重

    # 方案二：惩罚项参数
    complexity_lambda: float = 0.3   # 复杂度惩罚系数
    turnover_gamma: float = 2.0      # 换手率惩罚系数
    max_expr_length: int = 30        # 最大表达式长度

    # 方案三：滚动稳定性参数
    rolling_window_ratio: float = 0.25      # 滚动窗口占比
    rolling_stability_weight: float = 2.0   # 稳定性权重

    # 通用惩罚参数
    overfitting_threshold: float = 1.5      # 过拟合阈值
    overfitting_penalty: float = 1.0        # 过拟合惩罚系数

