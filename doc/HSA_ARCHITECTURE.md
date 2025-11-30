# 分层语义注意力 (HSA) 系统架构文档

## 1. 核心理念：三级记忆金字塔

具身智能的记忆机制不应仅依赖时间衰减，而应建立在**语义分层（Semantic Stratification）** 之上。HSA 将序列中的 Token 按其语义角色划分为三个层级，每层对应不同的注意力保留策略。

| 层级 | 名称 (Code) | 语义定义 | 典型内容 | 衰减策略 |
| :--- | :--- | :--- | :--- | :--- |
| **L0** | `GLOBAL` | 全局锚点 | 任务定义、系统指令 | **永恒关注** ($f(t)=1$) |
| **L1** | `LANDMARK` | 状态路标 | 关键动作 (Pickup, Open) | **长时记忆** (慢衰减 $\lambda_1$) |
| **L2** | `NOISE` | 过程噪声 | 导航、冗余观察 | **工作记忆** (快衰减 $\lambda_2$ + 截断) |

---

## 2. 数学模型

在标准 Transformer Self-Attention 基础上，HSA 引入了**加性语义偏置 (Additive Semantic Bias)**。

### 原始 Attention
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### HSA Attention
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \mathbf{M}_{\text{semantic}}\right)V
$$

### 偏置矩阵定义
偏置矩阵 $\mathbf{M}_{i,j}$（Query 位置 $i$，Key 位置 $j$）由 Key 的语义类型 `Semantic[j]` 和相对距离 $|i - j|$ 决定：

$$
\mathbf{M}_{i,j} =
\begin{cases}
0 & \text{if } \text{Semantic}[j] = \text{Global} \\
-\lambda_1 \cdot |i - j| & \text{if } \text{Semantic}[j] = \text{Landmark} \\
-\lambda_2 \cdot |i - j| & \text{if } \text{Semantic}[j] = \text{Noise and } |i - j| \leq W \\
-\infty & \text{if } \text{Semantic}[j] = \text{Noise and } |i - j| > W
\end{cases}
$$

#### 推荐参数
*   $\lambda_1 = 0.001$：用于关键事件，允许跨越数千个 Token 保持关注。
*   $\lambda_2 = 0.5$：用于过程噪声，迅速衰减以释放注意力带宽。
*   $W = 50$：噪声滑动窗口大小，超过此距离的噪声将被强制 Mask (设为 $-\infty$)。

---

## 3. 系统架构设计

HSA 的集成涉及 LLaMA-Factory 的三个主要子系统：

### 3.1 数据管道 (Data Pipeline)
负责将离散的语义标签转换为与 Token 对齐的张量。
*   **输入**: 带有 `semantic_type` 字段的 JSON 数据。
*   **处理**: Tokenizer 切分文本时，同步广播语义标签 ID。
*   **输出**: `input_ids` (Token序列) 和 `semantic_ids` (语义ID序列)。

### 3.2 批次整理 (Data Collation)
负责处理变长序列的对齐。
*   **机制**: 使用 `NOISE` 类型作为 Padding Value。
*   **理由**: Padding 部分应被视为无意义噪声，应用最激进的遗忘策略。

### 3.3 模型注入 (Model Injection)
负责在运行时修改计算图。
*   **技术**: Runtime Monkey Patching (运行时热补丁)。
*   **位置**: 拦截 `Qwen2VLAttention` (或其他目标模型) 的 `forward` 方法。
*   **操作**: 根据传入的 `semantic_ids` 动态计算 $\mathbf{M}_{\text{semantic}}$ 并叠加到 Attention Mask 上。

---

## 4. 架构图示

```mermaid
graph TD
    A[Raw Dataset (JSON)] -->|Preprocess| B(Tokenized Features)
    B -->|Collate| C{Batch Tensor}
    C -->|input_ids| D[Model Embedding]
    C -->|semantic_ids| E[HSA Bias Generator]
    
    subgraph Transformer Layer
    D --> Q[Query]
    D --> K[Key]
    D --> V[Value]
    E -->|Bias Matrix M| Attn[Self-Attention Core]
    Q --> Attn
    K --> Attn
    V --> Attn
    end
    
    Attn --> F[Output Hidden States]
```
