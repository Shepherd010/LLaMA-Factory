# Hierarchical Semantic Attention (HSA) 机制与实现详解

本文档详细介绍了 Hierarchical Semantic Attention (HSA) 机制的原理、设计动机以及在 LLaMA-Factory 中的具体实现细节。

## 1. 什么是 HSA？

**Hierarchical Semantic Attention (HSA)** 是一种针对长上下文 Agent 任务（特别是包含大量思维链/Chain-of-Thought 推理的任务）设计的注意力机制。

### 1.1 动机 (Motivation)
在复杂的 Agent 交互中，模型通常会产生大量的中间推理步骤（Process Noise）。这些推理步骤对于当前的局部决策是必要的，但对于长期的上下文记忆来说，它们往往是“噪声”，会稀释关键信息（如用户指令、关键决策点）的注意力权重。

HSA 的核心思想是根据 Token 的**语义类型 (Semantic Type)** 动态调整注意力机制，从而实现：
*   **全局关注**关键指令和定义。
*   **稀疏关注**关键事件（Landmarks）。
*   **局部关注**中间推理过程（Noise）。

### 1.2 语义类型 (Semantic Types)

HSA 将上下文中的 Token 分为三种语义类型：

| ID | 类型名称 | 描述 | 注意力行为 |
| :--- | :--- | :--- | :--- |
| **0** | **Global Condition** | 全局指令、系统提示、任务描述 | **全局可见**。所有 Token 都可以关注到它。 |
| **1** | **Key Event (Landmark)** | 关键决策、重要状态变更 | **长程衰减**。随距离增加线性衰减，保留长期记忆线索。 |
| **2** | **Process Noise** | 中间推理、思维链 (CoT) | **局部窗口**。仅在短距离窗口内可见，超出窗口即被 Mask。 |

## 2. 核心机制 (Mechanism)

HSA 通过在标准 Attention Score 上叠加一个 **Bias Matrix** 来实现。

$$ \text{Attention}(Q, K, V) = \text{Softmax}(\frac{QK^T}{\sqrt{d_k}} + M + \text{Bias}) V $$

其中 `Bias` 矩阵 $B_{ij}$ (第 $i$ 个 Query 对第 $j$ 个 Key 的偏置) 根据 Key ($j$) 的语义类型 $S_j$ 和距离 $d = |i - j|$ 计算：

1.  **若 $S_j = 0$ (Global)**:
    $$ B_{ij} = 0 $$
    (无偏置，标准注意力)

2.  **若 $S_j = 1$ (Landmark)**:
    $$ B_{ij} = -\lambda_1 \cdot d $$
    (线性距离衰减，$\lambda_1$ 为衰减系数)

3.  **若 $S_j = 2$ (Noise)**:
    *   若 $d \le W$ (在窗口内):
        $$ B_{ij} = -\lambda_2 \cdot d $$
    *   若 $d > W$ (超出窗口):
        $$ B_{ij} = -\infty $$
        (完全屏蔽)

**参数设置 (当前实现):**
*   $\lambda_1 = 0.001$
*   $\lambda_2 = 0.5$
*   $W = 50$ (窗口大小)

---

## 3. 代码实现 (Implementation)

HSA 的实现涉及数据处理管道的各个环节，从数据集加载到模型前向传播。

### 3.1 数据层 (Data Pipeline)

#### 3.1.1 数据集格式 (`data/hsa_test.json`)
数据集中的每条 Message 都包含一个额外的 `semantic_type` 字段：
```json
{
  "role": "system",
  "content": "...",
  "semantic_type": "GLOBAL_CONDITION"  // -> ID 0
},
{
  "role": "assistant",
  "content": "Hmm... let me think...",
  "semantic_type": "PROCESS_NOISE"     // -> ID 2
}
```

#### 3.1.2 模板解析 (`src/llamafactory/data/template.py`)
在 `Template._encode_with_semantic` 方法中，解析 `semantic_type` 并映射为 `semantic_ids`：
*   `GLOBAL_CONDITION` -> 0
*   `KEY_EVENT` -> 1
*   `PROCESS_NOISE` -> 2 (默认值)

这些 ID 会与 Token ID 一一对应，生成 `semantic_ids` 序列。

#### 3.1.3 数据预处理 (`src/llamafactory/data/processor/supervised.py`)
`SupervisedDatasetProcessor` 被修改以支持 `semantic_ids` 的传递。它将 `semantic_ids` 打包进 `model_inputs` 字典中。

#### 3.1.4 数据整理 (`src/llamafactory/data/collator.py`)
`MultiModalDataCollatorForSeq2Seq` 负责将 Batch 中的 `semantic_ids` 进行 Padding。
*   **关键修正**：由于 Qwen2-VL 等模型使用 **Left Padding**，标准的 `pad_sequence` (默认 Right Padding) 会导致 `semantic_ids` 与 `input_ids` 错位。
*   **实现**：添加了自定义 Padding 逻辑，根据 `tokenizer.padding_side` 动态选择填充方向，填充值为 `2` (Process Noise，即 Padding 被视为噪声)。

### 3.2 模型层 (Model Layer)

#### 3.2.1 HSA Patch (`src/llamafactory/model/hsa_patch.py`)
这是核心实现文件。它定义了 `apply_hsa_patch` 函数，该函数使用 **Monkey Patch** 技术替换了 `Qwen2VLAttention` 类的 `forward` 方法。

新的 `hsa_forward` 方法：
1.  从 `kwargs` 中提取 `semantic_ids`。
2.  基于上述公式计算 `bias_matrix`。
3.  将 `bias_matrix` 叠加到原始 `attention_mask` 上。
4.  调用原始的 `forward` 逻辑。

#### 3.2.2 Patch 注入 (`src/llamafactory/model/patcher.py`)
在模型加载时，检查环境变量 `USE_HSA`。如果为 `true`，则调用 `apply_hsa_patch`。

## 4. 如何运行

### 4.1 环境准备
确保安装了 LLaMA-Factory 及其依赖。

### 4.2 运行 Demo
使用提供的脚本 `run_hsa_demo.sh`：

```bash
export USE_HSA=true
bash run_hsa_demo.sh
```

该脚本会：
1.  设置 `USE_HSA=true` 环境变量。
2.  使用 `hsa_test` 数据集启动 `llamafactory-cli train`。
3.  加载 Qwen2-VL 模型并应用 HSA Patch。
4.  执行训练（演示目的，Steps 较少）。

## 5. 总结

HSA 通过在 Attention 层引入语义感知的偏置，有效地管理了长上下文中的信息流。在 LLaMA-Factory 中的实现是非侵入式的（通过 Patch），并且完全集成了数据处理管道，支持从 JSON 数据集直接驱动注意力机制的行为。
