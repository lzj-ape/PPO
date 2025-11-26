---
noteId: "73b6d630c84b11f08c769fd60f0ff98b"
tags: []

---

# 训练集计算失败问题 - 诊断工具使用指南

## 问题概述

你遇到的问题：约56% (9/16) 的表达式在训练集上计算失败(`train_computation_failed`)

## 我为你创建的工具

### 1. 📋 诊断报告 (`DIAGNOSIS_REPORT.md`)
- **详细的问题分析**
- **可能的原因及优先级**
- **具体的修复方案**
- **推荐阅读这个文件了解问题的本质**

### 2. 🔧 诊断工具 (`diagnose_utils.py`)
**这是最实用的工具！** 可以直接在你的代码中使用。

#### 使用方法:

在你的notebook或训练脚本中添加:

```python
# 导入诊断工具
from diagnose_utils import diagnose_failed_expressions, diagnose_single_expression

# 方法1: 在训练循环中收集失败的表达式
failed_expressions = []

# 在你的batch处理循环中:
for tokens, state_ids, trajectory in batch_results:
    factor_values = miner.factor_evaluator.compute_factor_train(tokens)
    if factor_values is None:
        failed_expressions.append(tokens)

# 批量诊断
if len(failed_expressions) > 0:
    diagnose_failed_expressions(failed_expressions, miner)

# 方法2: 诊断单个表达式
tokens = ['<BEG>', 'close', 'high', 'add', '<SEP>']  # 某个失败的表达式
result = diagnose_single_expression(tokens, miner, verbose=True)
```

**诊断工具会告诉你**:
- ✅ 栈是否平衡
- ✅ 在哪个token处失败
- ✅ 计算结果的质量(有效率、NaN率、Inf率)
- ✅ 失败的具体原因

### 3. 🧪 测试脚本

#### `test_computation_simple.py` - 简化测试
测试基础功能是否正常:
- 操作符是否能正常计算
- RPN表达式端到端计算
- 数据长度的影响
- 边界情况

#### `diagnose_train_computation_failure.py` - 完整诊断
5个系统性的测试:
1. RPN栈平衡验证
2. 操作符计算测试
3. 数据长度要求
4. 特征数据质量
5. 端到端计算测试

#### `quick_test.py` - 快速测试
最基础的功能验证

## 立即可以做的事

### 步骤1: 使用诊断工具找出问题 ⭐⭐⭐⭐⭐

在你的training notebook中添加诊断代码（见上面的使用方法）。

这会告诉你:
- 到底是哪些表达式失败了
- 是栈不平衡还是计算错误
- 具体在哪个操作符失败

### 步骤2: 根据诊断结果查看修复方案

打开[DIAGNOSIS_REPORT.md](DIAGNOSIS_REPORT.md)，根据诊断工具的输出，应用对应的修复方案。

**最可能需要的修复** (从报告中):

#### 修复A: 栈平衡问题
如果诊断显示"栈不平衡"，需要修改`expression_generator.py`中的循环结束逻辑。

#### 修复B: 数据长度问题
如果诊断显示"大量NaN"，需要:
1. 检查训练数据长度
2. 调整操作符的min_periods

#### 修复C: 操作符问题
如果特定操作符总是失败，需要修改对应的操作符实现。

## 示例: 在你的代码中添加诊断

找到你的训练循环（可能在notebook或`miner_core.py`中），修改如下:

```python
# 在文件开头添加
from diagnose_utils import diagnose_failed_expressions

# 在训练循环中 (通常在 mine_factors 或类似函数中):
def mine_factors(self, ...):
    # ... 现有代码 ...

    # 🆕 添加: 收集失败的表达式
    failed_expressions = []

    for iteration in range(num_iterations):
        # 生成表达式
        batch_results = self.expr_generator.generate_expression_batch(batch_size)

        valid_candidates = []
        for tokens, state_ids, trajectory in batch_results:
            # 计算因子
            factor_values = self.factor_evaluator.compute_factor_train(tokens)

            if factor_values is None:
                # 🆕 添加: 记录失败的表达式
                failed_expressions.append(tokens)
                continue

            # ... 现有的评估逻辑 ...

        # 🆕 添加: 每隔N次迭代,诊断一次
        if iteration % 10 == 0 and len(failed_expressions) > 0:
            logger.info(f"\n诊断最近的失败表达式 (共{len(failed_expressions)}个):")
            diagnose_failed_expressions(failed_expressions[-16:], self)  # 只诊断最近16个
            failed_expressions = []  # 清空
```

## 预期的诊断输出

运行后你会看到类似:

```
================================================================================
🔍 开始诊断失败的表达式
================================================================================
失败表达式数量: 9

================================================================================
表达式 1/9
Tokens: <BEG> close high add <SEP>
✅ 栈平衡检查通过 (栈大小=1)
✅ 计算成功:
   有效率: 100.0%
   NaN率: 0.0%
   Inf率: 0.0%
   均值: 201.2345
   标准差: 5.6789

================================================================================
表达式 2/9
Tokens: <BEG> close high low <SEP>
❌ 栈平衡检查失败:
   Final stack size is 3, expected 1
   错误位置: 第4个token
   最终栈大小: 3
...
```

从输出中你就能清楚地知道:
1. 有多少是栈不平衡
2. 有多少是计算错误
3. 具体是哪些token组合有问题

## 下一步

1. **立即行动**: 在你的代码中添加诊断工具
2. **查看输出**: 看到底是什么类型的失败
3. **应用修复**: 根据`DIAGNOSIS_REPORT.md`中的对应方案修复
4. **验证效果**: 重新运行，观察失败率是否下降

## 需要帮助?

如果诊断工具显示的问题不在报告的修复方案中，把诊断输出发给我，我会帮你分析具体的修复方法。

---

**重要**: 先运行诊断工具，不要盲目修改代码！诊断会告诉你真正的问题所在。
