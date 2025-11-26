"""
ExpressionGenerator 单元测试
测试表达式生成器的各个功能模块
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Dict
from unittest.mock import Mock, MagicMock


class MockActorCritic(nn.Module):
    """模拟的Actor-Critic网络"""
    
    def __init__(self, vocab_size: int, device: torch.device):
        super().__init__()
        self.vocab_size = vocab_size
        self.device = device
        
    def forward_batch(self, states: List[List[int]]):
        """模拟批量前向传播"""
        batch_size = len(states)
        
        # 返回随机的logits和values
        type_logits = torch.randn(batch_size, 3, device=self.device)  # 3种类型
        action_logits = torch.randn(batch_size, self.vocab_size, device=self.device)
        values = torch.randn(batch_size, device=self.device)
        
        return type_logits, action_logits, values


class TestExpressionGenerator(unittest.TestCase):
    """ExpressionGenerator测试类"""
    
    def setUp(self):
        """初始化测试环境"""
        # 构建词汇表
        self.feature_names = ['close', 'open', 'high', 'low', 'volume']
        self.operator_names = ['add', 'sub', 'mul', 'div', 'sma5', 'rank', 'delay1']
        special_tokens = ['<PAD>', '<BEG>', '<SEP>']
        
        self.vocab = special_tokens + self.feature_names + self.operator_names
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
        # 定义操作符
        self.operators = {
            'add': {'arity': 2, 'scale_rule': 'any'},
            'sub': {'arity': 2, 'scale_rule': 'any'},
            'mul': {'arity': 2, 'scale_rule': 'similar_only', 'scale_threshold': 100.0},
            'div': {'arity': 2, 'scale_rule': 'similar_only', 'scale_threshold': 100.0},
            'sma5': {'arity': 1, 'scale_rule': 'any'},
            'rank': {'arity': 1, 'scale_rule': 'any'},
            'delay1': {'arity': 1, 'scale_rule': 'any'},
        }
        
        # 特征数量级
        self.feature_scales = {
            'close': 100.0,
            'open': 100.0,
            'high': 100.0,
            'low': 100.0,
            'volume': 1000000.0,
        }
        
        self.device = torch.device('cpu')
        self.max_expr_len = 10
        
        # 创建模拟的actor_critic
        self.mock_actor_critic = MockActorCritic(len(self.vocab), self.device)
        
        # 导入ExpressionGenerator (假设在同一目录)
        # 这里需要根据实际情况调整导入
        try:
            from expression_generator import ExpressionGenerator
            self.ExpressionGenerator = ExpressionGenerator
        except ImportError:
            # 如果导入失败，跳过测试
            self.skipTest("Cannot import ExpressionGenerator")
            
        # 创建生成器实例
        self.generator = self.ExpressionGenerator(
            actor_critic=self.mock_actor_critic,
            vocab=self.vocab,
            token_to_id=self.token_to_id,
            id_to_token=self.id_to_token,
            operators=self.operators,
            feature_names=self.feature_names,
            feature_scales=self.feature_scales,
            max_expr_len=self.max_expr_len,
            device=self.device,
            use_amp=False
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.generator.max_expr_len, self.max_expr_len)
        self.assertEqual(self.generator.pad_token_id, self.token_to_id['<PAD>'])
        self.assertEqual(len(self.generator.vocab), len(self.vocab))
        
    def test_calculate_stack_size(self):
        """测试栈大小计算"""
        # 测试空状态
        state = [self.token_to_id['<BEG>']]
        self.assertEqual(self.generator._calculate_stack_size(state), 0)
        
        # 测试单个特征
        state = [self.token_to_id['<BEG>'], self.token_to_id['close']]
        self.assertEqual(self.generator._calculate_stack_size(state), 1)
        
        # 测试两个特征
        state = [self.token_to_id['<BEG>'], self.token_to_id['close'], 
                self.token_to_id['open']]
        self.assertEqual(self.generator._calculate_stack_size(state), 2)
        
        # 测试一元操作符
        state = [self.token_to_id['<BEG>'], self.token_to_id['close'], 
                self.token_to_id['sma5']]
        self.assertEqual(self.generator._calculate_stack_size(state), 1)
        
        # 测试二元操作符
        state = [self.token_to_id['<BEG>'], self.token_to_id['close'], 
                self.token_to_id['open'], self.token_to_id['add']]
        self.assertEqual(self.generator._calculate_stack_size(state), 1)
        
        # 测试无效状态（栈不足）
        state = [self.token_to_id['<BEG>'], self.token_to_id['add']]
        self.assertEqual(self.generator._calculate_stack_size(state), -1)
        
    def test_get_scale_stack(self):
        """测试数量级栈"""
        # 测试特征数量级
        state = [self.token_to_id['<BEG>'], self.token_to_id['close']]
        scale_stack = self.generator._get_scale_stack(state)
        self.assertEqual(scale_stack, [100.0])
        
        # 测试操作后的数量级
        state = [self.token_to_id['<BEG>'], self.token_to_id['close'], 
                self.token_to_id['sma5']]
        scale_stack = self.generator._get_scale_stack(state)
        self.assertEqual(len(scale_stack), 1)
        
        # 测试二元操作
        state = [self.token_to_id['<BEG>'], self.token_to_id['close'], 
                self.token_to_id['open'], self.token_to_id['add']]
        scale_stack = self.generator._get_scale_stack(state)
        self.assertEqual(len(scale_stack), 1)
        
    def test_compute_result_scale(self):
        """测试结果数量级计算"""
        # 测试一元操作
        scale = self.generator._compute_result_scale('sma5', [100.0])
        self.assertEqual(scale, 100.0)
        
        scale = self.generator._compute_result_scale('rank', [100.0])
        self.assertEqual(scale, 1.0)
        
        # 测试二元操作
        scale = self.generator._compute_result_scale('add', [100.0, 100.0])
        self.assertEqual(scale, 100.0)
        
        scale = self.generator._compute_result_scale('mul', [100.0, 2.0])
        self.assertEqual(scale, 200.0)
        
        scale = self.generator._compute_result_scale('div', [100.0, 2.0])
        self.assertEqual(scale, 50.0)
        
    def test_is_operator_scale_compatible(self):
        """测试操作符数量级兼容性"""
        # 测试栈不足
        result = self.generator._is_operator_scale_compatible('add', [100.0])
        self.assertFalse(result)
        
        # 测试any规则
        result = self.generator._is_operator_scale_compatible('add', [100.0, 1000.0])
        self.assertTrue(result)
        
        # 测试similar_only规则 - 兼容
        result = self.generator._is_operator_scale_compatible('mul', [100.0, 150.0])
        self.assertTrue(result)
        
        # 测试similar_only规则 - 不兼容
        result = self.generator._is_operator_scale_compatible('mul', [100.0, 100000.0])
        self.assertFalse(result)
        
    def test_get_valid_actions(self):
        """测试有效动作获取"""
        # 测试初始状态
        state = [self.token_to_id['<BEG>']]
        valid_types, valid_actions_by_type = self.generator._get_valid_actions(state)
        
        # 初始状态应该只能添加特征
        self.assertIn(0, valid_types)  # 特征类型
        self.assertIn(0, valid_actions_by_type)
        
        # 测试有一个特征后的状态
        state = [self.token_to_id['<BEG>'], self.token_to_id['close']]
        valid_types, valid_actions_by_type = self.generator._get_valid_actions(state)
        
        # 应该可以添加特征或一元操作符
        self.assertIn(0, valid_types)
        self.assertIn(1, valid_types)
        
        # 测试栈平衡状态（应该强制结束）
        state = [self.token_to_id['<BEG>'], self.token_to_id['close'], 
                self.token_to_id['sma5']]
        valid_types, valid_actions_by_type = self.generator._get_valid_actions(state)
        
        # stack=1且len>=4，应该只能结束
        if len(state) >= 4:
            self.assertEqual(valid_types, [2])
            self.assertIn(2, valid_actions_by_type)
            
    def test_tokens_to_expression(self):
        """测试token转表达式"""
        # 测试有效表达式
        tokens = ['<BEG>', 'close', 'sma5', '<SEP>']
        expr = self.generator.tokens_to_expression(tokens)
        self.assertEqual(expr, 'sma5(close)')
        
        # 测试二元操作
        tokens = ['<BEG>', 'close', 'open', 'add', '<SEP>']
        expr = self.generator.tokens_to_expression(tokens)
        self.assertEqual(expr, 'add(close, open)')
        
        # 测试复杂表达式
        tokens = ['<BEG>', 'close', 'sma5', 'open', 'add', '<SEP>']
        expr = self.generator.tokens_to_expression(tokens)
        self.assertEqual(expr, 'add(sma5(close), open)')
        
        # 测试无效表达式 - 缺少开始标记
        tokens = ['close', 'sma5', '<SEP>']
        expr = self.generator.tokens_to_expression(tokens)
        self.assertEqual(expr, 'INVALID_EXPRESSION')
        
        # 测试无效表达式 - 栈不平衡
        tokens = ['<BEG>', 'close', 'open', '<SEP>']
        expr = self.generator.tokens_to_expression(tokens)
        self.assertEqual(expr, 'INVALID_EXPRESSION')
        
    def test_generate_expression_batch(self):
        """测试批量生成表达式"""
        batch_size = 4
        results = self.generator.generate_expression_batch(batch_size)
        
        # 检查返回结果数量
        self.assertEqual(len(results), batch_size)
        
        # 检查每个结果的结构
        for tokens, state_ids, trajectory in results:
            # tokens应该以<BEG>开始
            self.assertEqual(tokens[0], '<BEG>')
            
            # 检查trajectory结构
            self.assertIn('states', trajectory)
            self.assertIn('actions', trajectory)
            self.assertIn('types', trajectory)
            self.assertIn('type_log_probs', trajectory)
            self.assertIn('action_log_probs', trajectory)
            self.assertIn('values', trajectory)
            
            # 检查各个列表长度一致
            traj_len = len(trajectory['actions'])
            self.assertEqual(len(trajectory['states']), traj_len)
            self.assertEqual(len(trajectory['types']), traj_len)
            self.assertEqual(len(trajectory['type_log_probs']), traj_len)
            self.assertEqual(len(trajectory['action_log_probs']), traj_len)
            self.assertEqual(len(trajectory['values']), traj_len)
            
            # 尝试转换为可读表达式
            expr = self.generator.tokens_to_expression(tokens)
            print(f"Generated: {' '.join(tokens)} -> {expr}")
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 测试接近最大长度
        state = [self.token_to_id['<BEG>']]
        for i in range(self.max_expr_len - 4):
            state.append(self.token_to_id['close'])
        
        valid_types, valid_actions_by_type = self.generator._get_valid_actions(state)
        # 应该被强制结束
        print(f"Near max length, valid_types: {valid_types}")
        
        # 测试零数量级的除法
        scale = self.generator._compute_result_scale('div', [100.0, 0.0])
        self.assertEqual(scale, 100.0)  # 应该返回分子的数量级
        
        # 测试空scale_stack
        result = self.generator._is_operator_scale_compatible('add', [])
        self.assertFalse(result)


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def setUp(self):
        """设置集成测试环境"""
        self.feature_names = ['close', 'open', 'volume']
        self.operator_names = ['add', 'sma5']
        special_tokens = ['<PAD>', '<BEG>', '<SEP>']

        self.vocab = special_tokens + self.feature_names + self.operator_names
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

        self.operators = {
            'add': {'arity': 2, 'scale_rule': 'any'},
            'sma5': {'arity': 1, 'scale_rule': 'any'},
        }

        self.feature_scales = {
            'close': 100.0,
            'open': 100.0,
            'volume': 1000000.0,
        }

        self.device = torch.device('cpu')
        self.mock_actor_critic = MockActorCritic(len(self.vocab), self.device)

        try:
            from expression_generator import ExpressionGenerator
            self.generator = ExpressionGenerator(
                actor_critic=self.mock_actor_critic,
                vocab=self.vocab,
                token_to_id=self.token_to_id,
                id_to_token=self.id_to_token,
                operators=self.operators,
                feature_names=self.feature_names,
                feature_scales=self.feature_scales,
                max_expr_len=20,
                device=self.device,
                use_amp=False
            )
        except ImportError:
            self.skipTest("Cannot import ExpressionGenerator")

    def test_full_generation_workflow(self):
        """测试完整的生成流程"""
        print("\n=== Full Generation Workflow Test ===")

        # 生成多个批次
        for batch_idx in range(3):
            print(f"\nBatch {batch_idx + 1}:")
            results = self.generator.generate_expression_batch(batch_size=2)

            for idx, (tokens, state_ids, trajectory) in enumerate(results):
                expr = self.generator.tokens_to_expression(tokens)
                print(f"  Expression {idx + 1}: {expr}")
                print(f"    Tokens: {' '.join(tokens)}")
                print(f"    Steps: {len(trajectory['actions'])}")

                # 验证表达式有效性
                if expr != 'INVALID_EXPRESSION':
                    self.assertIn('<BEG>', tokens)
                    self.assertIn('<SEP>', tokens)


class TestFactorValidation(unittest.TestCase):
    """因子合法性检验测试"""

    def setUp(self):
        """设置验证测试环境"""
        # 基础配置
        self.feature_names = ['close', 'open', 'high', 'low', 'volume']
        self.operator_names = ['add', 'sub', 'mul', 'div', 'sma5', 'rank', 'delay1']
        special_tokens = ['<PAD>', '<BEG>', '<SEP>']

        self.vocab = special_tokens + self.feature_names + self.operator_names
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

        # 定义操作符函数
        def safe_div(a, b):
            """安全除法"""
            result = a / b.replace(0, np.nan)
            return result.replace([np.inf, -np.inf], np.nan)

        self.operators = {
            'add': {
                'arity': 2,
                'scale_rule': 'any',
                'func': lambda a, b: a + b
            },
            'sub': {
                'arity': 2,
                'scale_rule': 'any',
                'func': lambda a, b: a - b
            },
            'mul': {
                'arity': 2,
                'scale_rule': 'similar_only',
                'scale_threshold': 100.0,
                'func': lambda a, b: a * b
            },
            'div': {
                'arity': 2,
                'scale_rule': 'similar_only',
                'scale_threshold': 100.0,
                'func': safe_div
            },
            'sma5': {
                'arity': 1,
                'scale_rule': 'any',
                'func': lambda a: a.rolling(window=5, min_periods=1).mean()
            },
            'rank': {
                'arity': 1,
                'scale_rule': 'any',
                'func': lambda a: a.rank(pct=True)
            },
            'delay1': {
                'arity': 1,
                'scale_rule': 'any',
                'func': lambda a: a.shift(1)
            },
        }

        self.feature_scales = {
            'close': 100.0,
            'open': 100.0,
            'high': 100.0,
            'low': 100.0,
            'volume': 1000000.0,
        }

        self.device = torch.device('cpu')

        # 创建模拟数据
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        self.mock_data = pd.DataFrame({
            'close': 100 + np.random.randn(100).cumsum(),
            'open': 100 + np.random.randn(100).cumsum() * 0.98,
            'high': 100 + np.random.randn(100).cumsum() * 1.02,
            'low': 100 + np.random.randn(100).cumsum() * 0.96,
            'volume': 1000000 + np.random.randn(100) * 100000,
        }, index=dates)

        # 创建生成器
        try:
            from expression_generator import ExpressionGenerator
            self.mock_actor_critic = MockActorCritic(len(self.vocab), self.device)
            self.generator = ExpressionGenerator(
                actor_critic=self.mock_actor_critic,
                vocab=self.vocab,
                token_to_id=self.token_to_id,
                id_to_token=self.id_to_token,
                operators=self.operators,
                feature_names=self.feature_names,
                feature_scales=self.feature_scales,
                max_expr_len=20,
                device=self.device,
                use_amp=False
            )

            # 创建评估器
            from factor_evaluator import FactorEvaluator

            # 简单的组合模型mock
            class MockCombinationModel:
                def __init__(self):
                    self.alpha_pool = []
                    self.base_train_score = 0.0
                    self.base_val_score = 0.0
                    self.current_weights = None

                    class Config:
                        ic_threshold = 0.01
                    self.config = Config()

                def evaluate_new_factor(self, alpha_info, train_factor, val_factor):
                    """模拟评估新因子"""
                    return {
                        'train_incremental_sharpe': 0.02,
                        'train_stats': {
                            'sharpe': 0.5,
                            'composite_score': 0.5
                        },
                        'val_stats': {
                            'sharpe': 0.4,
                            'composite_score': 0.4
                        }
                    }

                def add_alpha_and_optimize(self, alpha_info, train_factor, val_factor):
                    """模拟添加因子"""
                    self.alpha_pool.append(alpha_info)
                    return {
                        'pool_size': len(self.alpha_pool),
                        'current_train_score': 0.5,
                        'current_val_score': 0.4
                    }

            self.combination_model = MockCombinationModel()

            # 划分训练和验证数据
            split_idx = 70
            self.train_data = self.mock_data.iloc[:split_idx]
            self.val_data = self.mock_data.iloc[split_idx:]
            self.train_target = pd.Series(np.random.randn(split_idx), index=self.train_data.index)
            self.val_target = pd.Series(np.random.randn(len(self.val_data)), index=self.val_data.index)

            self.evaluator = FactorEvaluator(
                operators=self.operators,
                feature_names=self.feature_names,
                combination_model=self.combination_model,
                train_data=self.train_data,
                val_data=self.val_data,
                train_target=self.train_target,
                val_target=self.val_target
            )

        except ImportError as e:
            self.skipTest(f"Cannot import required modules: {e}")

    def test_factor_computation_basic(self):
        """测试基础因子计算"""
        print("\n=== Test Basic Factor Computation ===")

        # 测试简单因子: close
        tokens = ['<BEG>', 'close', '<SEP>']
        expr_tokens = tokens[1:-1]

        result = self.evaluator.compute_factor_values(expr_tokens, self.train_data, is_training=True)
        self.assertIsNotNone(result, "Simple factor 'close' should compute successfully")
        self.assertEqual(len(result), len(self.train_data))
        print(f"✓ Factor 'close' computed: shape={result.shape}, mean={result.mean():.4f}, std={result.std():.4f}")

    def test_factor_computation_unary_op(self):
        """测试一元操作符因子"""
        print("\n=== Test Unary Operator Factor ===")

        # 测试 sma5(close)
        tokens = ['<BEG>', 'close', 'sma5', '<SEP>']
        expr_tokens = tokens[1:-1]

        result = self.evaluator.compute_factor_values(expr_tokens, self.train_data, is_training=True)
        self.assertIsNotNone(result, "Factor 'sma5(close)' should compute successfully")
        print(f"✓ Factor 'sma5(close)' computed: mean={result.mean():.4f}, std={result.std():.4f}")

    def test_factor_computation_binary_op(self):
        """测试二元操作符因子"""
        print("\n=== Test Binary Operator Factor ===")

        # 测试 add(close, open)
        tokens = ['<BEG>', 'close', 'open', 'add', '<SEP>']
        expr_tokens = tokens[1:-1]

        result = self.evaluator.compute_factor_values(expr_tokens, self.train_data, is_training=True)
        self.assertIsNotNone(result, "Factor 'add(close, open)' should compute successfully")
        print(f"✓ Factor 'add(close, open)' computed: mean={result.mean():.4f}, std={result.std():.4f}")

    def test_factor_computation_complex(self):
        """测试复杂因子"""
        print("\n=== Test Complex Factor ===")

        # 测试 add(sma5(close), open)
        tokens = ['<BEG>', 'close', 'sma5', 'open', 'add', '<SEP>']
        expr_tokens = tokens[1:-1]

        result = self.evaluator.compute_factor_values(expr_tokens, self.train_data, is_training=True)
        self.assertIsNotNone(result, "Complex factor should compute successfully")
        expr = self.generator.tokens_to_expression(tokens)
        print(f"✓ Factor '{expr}' computed: mean={result.mean():.4f}, std={result.std():.4f}")

    def test_factor_validation_invalid_format(self):
        """测试无效格式"""
        print("\n=== Test Invalid Format ===")

        # 缺少<BEG>
        tokens1 = ['close', 'sma5', '<SEP>']
        result1 = self.evaluator.evaluate_expression(tokens1, trial_only=True)
        self.assertFalse(result1['valid'])
        self.assertEqual(result1['reason'], 'invalid_format')
        print("✓ Missing <BEG> detected")

        # 缺少<SEP>
        tokens2 = ['<BEG>', 'close', 'sma5']
        result2 = self.evaluator.evaluate_expression(tokens2, trial_only=True)
        self.assertFalse(result2['valid'])
        self.assertEqual(result2['reason'], 'invalid_format')
        print("✓ Missing <SEP> detected")

    def test_generated_factors_validation(self):
        """测试生成因子的合法性检验"""
        print("\n=== Test Generated Factors Validation ===")

        # 生成多个因子并验证
        results = self.generator.generate_expression_batch(batch_size=5)

        valid_count = 0
        invalid_count = 0

        for idx, (tokens, state_ids, trajectory) in enumerate(results):
            expr = self.generator.tokens_to_expression(tokens)

            # 使用评估器验证因子
            eval_result = self.evaluator.evaluate_expression(tokens, trial_only=True)

            print(f"\nFactor {idx + 1}:")
            print(f"  Expression: {expr}")
            print(f"  Tokens: {' '.join(tokens[:15])}{'...' if len(tokens) > 15 else ''}")
            print(f"  Valid: {eval_result['valid']}")

            if eval_result['valid']:
                valid_count += 1
                print(f"  ✓ Can compute on data")
                # 检查返回的统计信息
                self.assertIn('train_factor', eval_result)
                self.assertIn('val_factor', eval_result)
                train_factor = eval_result['train_factor']
                val_factor = eval_result['val_factor']
                print(f"    Train: shape={train_factor.shape}, mean={train_factor.mean():.4f}, std={train_factor.std():.4f}")
                print(f"    Val: shape={val_factor.shape}, mean={val_factor.mean():.4f}, std={val_factor.std():.4f}")
            else:
                invalid_count += 1
                reason = eval_result.get('reason', 'unknown')
                print(f"  ✗ Cannot compute: {reason}")

        print(f"\n=== Summary ===")
        print(f"Valid factors: {valid_count}/{len(results)}")
        print(f"Invalid factors: {invalid_count}/{len(results)}")

        # 至少应该有一些有效的因子
        self.assertGreater(valid_count, 0, "Should generate at least some valid factors")

    def test_scale_compatibility(self):
        """测试数量级兼容性检查"""
        print("\n=== Test Scale Compatibility ===")

        # 测试兼容的数量级: mul(close, high) - 两者都是价格级别
        # 注意：我们需要在应用mul之前检查栈
        tokens1 = ['<BEG>', 'close', 'high']
        state1 = [self.token_to_id[t] for t in tokens1]

        scale_stack1 = self.generator._get_scale_stack(state1)
        compatible1 = self.generator._is_operator_scale_compatible('mul', scale_stack1)
        print(f"mul(close, high): scale_stack={scale_stack1}, compatible={compatible1}")
        self.assertTrue(compatible1, "Close and high should be scale-compatible for mul")

        # 测试不兼容的数量级: mul(close, volume) - 价格 vs 成交量
        tokens2 = ['<BEG>', 'close', 'volume']
        state2 = [self.token_to_id[t] for t in tokens2]

        scale_stack2 = self.generator._get_scale_stack(state2)
        compatible2 = self.generator._is_operator_scale_compatible('mul', scale_stack2)
        print(f"mul(close, volume): scale_stack={scale_stack2}, compatible={compatible2}")
        self.assertFalse(compatible2, "Close and volume should NOT be scale-compatible for mul")

    def test_train_val_consistency(self):
        """测试训练集和验证集的统计量一致性"""
        print("\n=== Test Train/Val Consistency ===")

        tokens = ['<BEG>', 'close', 'sma5', '<SEP>']
        expr_tokens = tokens[1:-1]

        # 计算训练集因子值（会保存统计量）
        train_result = self.evaluator.compute_factor_values(expr_tokens, self.train_data, is_training=True)
        self.assertIsNotNone(train_result)

        # 保存训练集统计量
        train_stats = self.evaluator.current_factor_stats.copy()
        print(f"Train stats: mean={train_stats['mean']:.4f}, std={train_stats['std']:.4f}")

        # 计算验证集因子值（应使用训练集统计量）
        val_result = self.evaluator.compute_factor_values(expr_tokens, self.val_data, is_training=False)
        self.assertIsNotNone(val_result)

        # 验证统计量未改变
        val_stats = self.evaluator.current_factor_stats
        print(f"Val stats (should be same): mean={val_stats['mean']:.4f}, std={val_stats['std']:.4f}")

        self.assertEqual(train_stats['mean'], val_stats['mean'], "Mean should not change for validation set")
        self.assertEqual(train_stats['std'], val_stats['std'], "Std should not change for validation set")
        print("✓ Statistics consistency maintained")


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加测试
    suite.addTests(loader.loadTestsFromTestCase(TestExpressionGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestFactorValidation))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 返回测试结果
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)