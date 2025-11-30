# HSA 工作流与数据流转说明

本文档详细说明数据如何在 HSA 增强版的 LLaMA-Factory 中流转。

## 1. 数据预处理阶段 (Preprocessing)

**文件**: `src/llamafactory/data/preprocess.py`

在此阶段，原始的对话数据被转换为模型可理解的 ID 序列。

### 输入示例
```json
{
    "messages": [
        {"role": "user", "content": "Task: Pickup apple", "semantic_type": "GLOBAL_CONDITION"},
        {"role": "assistant", "content": "Moving forward...", "semantic_type": "PROCESS_NOISE"}
    ]
}
```

### 处理逻辑
1.  **映射**: 将字符串标签映射为整数 ID。
    *   `GLOBAL_CONDITION` -> `0`
    *   `KEY_EVENT` -> `1`
    *   `PROCESS_NOISE` -> `2`
2.  **广播 (Broadcasting)**:
    *   假设 "Task: Pickup apple" 被切分为 `[101, 205, 309]` (3个Token)。
    *   对应的语义 ID `0` 被复制 3 次：`[0, 0, 0]`。
3.  **拼接**: 将所有消息的 ID 拼接成完整的序列。

### 输出特征
*   `input_ids`: `[101, 205, 309, ...]`
*   `semantic_ids`: `[0, 0, 0, ...]`

---

## 2. 数据整理阶段 (Collation)

**文件**: `src/llamafactory/data/collator.py`

在此阶段，不同长度的样本被打包成一个 Batch。

### 关键操作：Padding
由于不同样本长度不一，需要填充。
*   **Input IDs Padding**: 通常使用 `pad_token_id`。
*   **Semantic IDs Padding**: **必须使用 `2` (NOISE)**。
    *   这确保了 Padding 部分在 Attention 计算中会被视为噪声，从而被快速衰减或 Mask 掉，不会干扰正常计算。

### 输出 Tensor 形状
假设 Batch Size = 4, Max Seq Len = 1024:
*   `input_ids`: `[4, 1024]`
*   `semantic_ids`: `[4, 1024]`

---

## 3. 训练运行时阶段 (Training Runtime)

**文件**: `src/llamafactory/model/hsa_patch.py`

在此阶段，数据进入 GPU，模型开始前向传播。

### 步骤 1: 参数透传
HuggingFace Trainer 默认会过滤未定义的参数。通过设置 `remove_unused_columns=False`，`semantic_ids` 能够顺利传入模型的 `forward` 函数。

### 步骤 2: 拦截与计算
1.  **拦截**: Monkey Patch 拦截了 Attention 层的调用。
2.  **提取**: 从 `kwargs` 中提取 `semantic_ids`。
3.  **计算 Bias**:
    *   构建相对距离矩阵 $D$ ($N \times N$)。
    *   根据 `semantic_ids` 构建类型掩码。
    *   应用公式计算 $\mathbf{M}_{\text{semantic}}$。
    *   **优化**: 为了节省显存，Bias 计算应在 `torch.no_grad()` 上下文中进行（如果它是固定的），或者作为计算图的一部分（如果需要梯度，通常不需要）。

### 步骤 3: 融合
*   xxxxxxxxxx [HSA] Enabling Hierarchical Semantic Attention...text
*   调用原始的 Attention 实现（如 `torch.nn.functional.scaled_dot_product_attention` 或其他实现）。

---

## 4. 调试与验证

为确保工作流正常，建议在 `preprocess.py` 中添加断点或日志：

```python
# Debug snippet
if len(input_ids) != len(semantic_ids):
    logger.error(f"Mismatch! Input: {len(input_ids)}, Semantic: {len(semantic_ids)}")
```

在 `hsa_patch.py` 中打印 Bias 的统计信息：
```python
# Debug snippet
print(f"Bias Mean: {bias.mean().item()}, Max: {bias.max().item()}")
```
