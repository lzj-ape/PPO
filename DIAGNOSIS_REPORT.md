---
noteId: "3fd918a0c84b11f08c769fd60f0ff98b"
tags: []

---

# 训练集计算失败问题诊断报告

## 问题描述

从日志中看到：
```
2025-11-23 16:53:27,065 - INFO - ❌ Batch iteration 2: NO valid candidates
2025-11-23 16:53:27,069 - INFO -    All 16 expressions failed validation
2025-11-23 16:53:27,070 - INFO -    Failure breakdown:
2025-11-23 16:53:27,070 - INFO -      train_computation_failed: 9/16
```

**核心问题**: 约56% (9/16) 的表达式在训练集上计算失败(`train_computation_failed`)

## 可能的原因分析

### 1. RPN栈不平衡问题 ⭐⭐⭐⭐⭐

**最可能的原因**

虽然`ExpressionGenerator._get_valid_actions()`有严格的栈平衡约束，但可能存在边界情况:

#### 问题点A: 循环结束时的强制<SEP>添加
[expression_generator.py:158-173](expression_generator.py#L158-L173)

```python
# 🔥 修复：循环结束后，为所有未完成的表达式强制添加 <SEP>
for i in range(batch_size):
    if not batch_finished[i]:
        # 检查是否有至少1个有效token（除了<BEG>）
        if len(batch_tokens[i]) < 2:
            # 极端情况：只有<BEG>，添加一个默认特征
            default_feature = 'close' if 'close' in self.feature_names else self.feature_names[0]
            batch_tokens[i].append(default_feature)
            batch_states[i].append(self.token_to_id[default_feature])
            logger.warning(f"Expression {i} had only <BEG>, added default feature '{default_feature}'")

        # 添加<SEP>
        batch_tokens[i].append('<SEP>')
        batch_states[i].append(self.token_to_id['<SEP>'])
```

**问题**: 如果循环结束时栈大小不是1，强制添加`<SEP>`会导致不平衡的表达式。

**例子**:
- 生成过程中产生了: `<BEG> close high` (栈大小=2)
- 循环结束，强制添加: `<BEG> close high <SEP>` ❌ **栈大小2,不平衡!**

#### 问题点B: 无有效动作时的强制结束
[expression_generator.py:256-261](expression_generator.py#L256-L261)

```python
# 🔥 约束4: 如果没有有效动作,强制结束(兜底,防止死锁)
if not valid_types:
    logger.warning(f"No valid actions at state len={current_len}, stack={stack_size}, "
                 f"remaining_space={remaining_space}, forcing <SEP> (will be INVALID)")
    return [2], {2: [self.token_to_id['<SEP>']]}
```

这里明确标注了"will be INVALID"，说明生成器自己知道这种情况会产生无效表达式。

### 2. 数据长度不足问题 ⭐⭐⭐⭐

某些操作符需要较长的滚动窗口，但训练数据可能不够长。

#### 问题算子识别:

从[operators.py](factor/operators.py)看，以下算子需要较大窗口:

- `std20`: 需要20个数据点 (min_periods=10)
- `sma20`: 需要20个数据点 (min_periods=10)
- `zscore20`: 需要20个数据点
- `variance20`: 需要20个数据点
- `mad20`: 需要20个数据点
- `rsi14`: 需要14个数据点
- `macd`: 需要26个数据点

**数据分割验证**:
查看[miner_core.py](PPO/miner_core.py)中的数据分割:
```python
# 需要检查 self.train_data 的实际长度
```

如果训练集只有100-200条数据，那么:
- 前20条会产生大量NaN (用于20窗口的算子)
- 有效数据可能不足以计算Sharpe等指标

### 3. NaN/Inf传播问题 ⭐⭐⭐

#### 问题链条:
```
输入特征有NaN → 操作符计算 → 结果有NaN → 下一个操作符 → 更多NaN
```

#### 关键代码检查:

[operators.py:128](factor/operators.py#L128) - SMA实现:
```python
def sma(x: pd.Series, window: int = 5) -> pd.Series:
    """简单移动平均"""
    min_periods = max(window // 2, 2)  # 至少需要一半窗口或2个数据点
    return x.rolling(window=window, min_periods=min_periods).mean().fillna(method='bfill').fillna(0)
```

✅ 使用了`fillna(method='bfill').fillna(0)`，理论上应该处理NaN

[operators.py:163](factor/operators.py#L163) - STD实现:
```python
def std(x: pd.Series, window: int = 20) -> pd.Series:
    """标准差"""
    min_periods = max(window // 2, 3)  # 标准差至少需要3个点
    return x.rolling(window=window, min_periods=min_periods).std().fillna(method='bfill').fillna(0)
```

✅ 也使用了`fillna()`处理

**但是**: `fillna(method='bfill')`在pandas新版本可能被弃用，应该使用`bfill()`

### 4. FactorEvaluator的计算流程问题 ⭐⭐⭐

查看[factor/factor_evaluator.py](factor/factor_evaluator.py)中的`compute_factor_train()`方法:

需要检查:
1. 表达式是否正确转换为计算
2. 异常是否被正确捕获和记录
3. 返回值是否正确处理

### 5. 操作符函数本身的Bug ⭐⭐

某些操作符可能在特定输入下崩溃。

#### 高风险算子:
- **除法算子** (`div`): 除零问题
- **对数算子** (`log`): 负数问题
- **开方算子** (`sqrt`): 负数问题
- **相关性算子** (`corr20`): 需要足够的数据点

检查[operators.py:30-32](factor/operators.py#L30-L32):
```python
def div(x: pd.Series, y: pd.Series) -> pd.Series:
    """除法（安全）"""
    return (x / (y.replace(0, np.nan) + 1e-8)).fillna(0).replace([np.inf, -np.inf], 0)
```

✅ 已经做了安全处理

## 诊断测试脚本

我已经创建了三个测试脚本:

### 1. 完整诊断 (`diagnose_train_computation_failure.py`)
- 测试1: RPN栈平衡验证
- 测试2: 操作符计算测试
- 测试3: 数据长度要求
- 测试4: 特征数据质量
- 测试5: 端到端计算测试

### 2. 简化测试 (`test_computation_simple.py`)
- 专注于核心问题
- 测试不同数据长度的影响
- 测试边界情况

### 3. 快速测试 (`quick_test.py`)
- 最基础的算子测试

## 建议的修复方案

### 🔥 修复1: 严格保证RPN栈平衡 (优先级: 最高)

修改`ExpressionGenerator.generate_expression_batch()`:

```python
# 在 max_expr_len-1 循环中:
for step in range(self.max_expr_len - 1):
    # ...现有逻辑...

# 循环结束后的处理:
for i in range(batch_size):
    if not batch_finished[i]:
        # 🔥 新逻辑: 检查栈大小,添加必要的操作符以平衡栈
        current_stack = self._calculate_stack_size(batch_states[i])

        if current_stack == 0:
            # 栈为空,添加默认特征
            default_feature = 'close' if 'close' in self.feature_names else self.feature_names[0]
            batch_tokens[i].append(default_feature)
            batch_states[i].append(self.token_to_id[default_feature])
            current_stack = 1

        # 添加操作符使栈平衡到1
        while current_stack > 1:
            # 找到能消耗最多栈的操作符
            best_arity = min(current_stack, 3)  # 最多三元操作符

            # 找合适的操作符
            suitable_ops = [
                op_name for op_name, op_info in self.operators.items()
                if op_info['arity'] == best_arity
            ]

            if suitable_ops:
                chosen_op = np.random.choice(suitable_ops)
                batch_tokens[i].append(chosen_op)
                batch_states[i].append(self.token_to_id[chosen_op])
                current_stack = current_stack - best_arity + 1
            else:
                break

        # 最后添加<SEP>
        batch_tokens[i].append('<SEP>')
        batch_states[i].append(self.token_to_id['<SEP>'])
```

### 🔥 修复2: 在FactorEvaluator中添加详细日志

在计算失败时记录具体原因:

```python
def compute_factor_train(self, tokens: List[str]) -> Optional[pd.Series]:
    """计算训练集因子值"""
    try:
        # 1. 检查栈平衡
        stack_size = self._calculate_stack_size(tokens)
        if stack_size != 1:
            logger.warning(f"Invalid stack size: {stack_size}, tokens: {' '.join(tokens)}")
            return None

        # 2. 计算
        result = self._compute_from_rpn(tokens)

        # 3. 检查结果质量
        if result is None or len(result) == 0:
            logger.warning(f"Empty result for tokens: {' '.join(tokens)}")
            return None

        valid_ratio = (~result.isna()).sum() / len(result)
        if valid_ratio < 0.5:
            logger.warning(f"Low valid ratio ({valid_ratio:.2%}) for tokens: {' '.join(tokens)}")
            return None

        return result

    except Exception as e:
        logger.error(f"Computation failed for tokens: {' '.join(tokens)}, error: {e}")
        return None
```

### 🔥 修复3: 增加min_periods的灵活性

修改operators.py中的滚动窗口操作:

```python
def std(x: pd.Series, window: int = 20) -> pd.Series:
    """标准差 - 自适应min_periods"""
    data_len = len(x)

    # 自适应min_periods: 数据短时用更小的min_periods
    if data_len < window:
        min_periods = max(3, data_len // 2)
    else:
        min_periods = max(window // 2, 3)

    result = x.rolling(window=window, min_periods=min_periods).std()

    # 向前填充 + 填0
    result = result.bfill().fillna(0)

    return result
```

### 🔥 修复4: 在数据准备阶段检查长度

在FactorMinerCore初始化时:

```python
def _split_data(self, data: pd.DataFrame):
    """数据分割"""
    # ...现有分割逻辑...

    # 检查训练集长度
    min_required_length = 200  # 最少需要200条数据
    if len(self.train_data) < min_required_length:
        logger.warning(
            f"Train data length ({len(self.train_data)}) is less than "
            f"recommended minimum ({min_required_length}). "
            f"Some operators may produce excessive NaN values."
        )
```

## 下一步行动

1. **运行测试脚本**确认问题具体位置:
   ```bash
   python quick_test.py
   python test_computation_simple.py
   python diagnose_train_computation_failure.py
   ```

2. **查看实际的失败表达式**: 修改miner代码,在计算失败时打印完整的token序列

3. **应用修复方案**: 根据测试结果,优先应用修复1和修复2

4. **验证修复效果**: 重新运行挖掘流程,观察`train_computation_failed`的比例是否下降

## 总结

最可能的问题是**RPN栈不平衡**,特别是在表达式生成循环结束时强制添加`<SEP>`的逻辑。

建议:
1. ✅ 优先修复栈平衡问题
2. ✅ 添加详细的失败日志
3. ✅ 检查数据长度要求
4. ✅ 增强操作符的鲁棒性
