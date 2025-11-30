# HSA 数据工程：自动语义标注 (Auto-Labeling)

为避免昂贵且低效的人工标注，我们在数据预处理阶段引入**规则引擎**，根据具身智能任务的特性自动生成 `semantic_ids`。

## 1. 核心理念

通过解析对话中的角色（Role）和内容（Content），我们可以推断出每个 Token 在任务执行中的语义重要性。

*   **System/Instruction**: 定义了任务的根本目标，必须全程记住。
*   **Critical Actions**: 改变了物理环境状态（如拿起物体），是状态跟踪的关键锚点。
*   **Navigation/Observation**: 主要是为了达成动作的中间过程，包含大量冗余信息，应当被适度遗忘。

---

## 2. 语义标签定义

我们在 Python 代码中定义如下常量映射：

```python
SEMANTIC_GLOBAL = 0    # 全局锚点 (Global Anchor)
SEMANTIC_LANDMARK = 1  # 状态路标 (State Landmarks)
SEMANTIC_NOISE = 2     # 过程噪声 (Process Noise)
```

---

## 3. 规则映射引擎

我们实现了一个启发式的规则函数 `assign_semantic_label`，用于为每一条 Message 分配语义标签。

### 3.1 规则逻辑代码

```python
def assign_semantic_label(message, task_instruction_step=0):
    """
    根据消息内容和角色分配语义标签。
    """
    role = message["role"]
    content = message["content"]
    
    # 规则 1: System Prompt 或初始任务指令 → 全局锚点 (L0)
    # System Prompt 包含世界观和基本约束
    # 初始指令 (通常是第一条 User Message) 包含具体任务目标
    if role == "system" or message.get("step_index") == task_instruction_step:
        return SEMANTIC_GLOBAL
    
    # 规则 2: 判断是否为关键动作 → 状态路标 (L1)
    # 具身智能中的关键动作通常改变了环境状态
    if "<DecisionMaking>" in content:
        # 提取动作动词 (需配合具体的解析逻辑)
        # 假设 extract_action 能从 "<DecisionMaking>pickup apple</DecisionMaking>" 提取出 "pickup"
        action = extract_action(content)  
        
        # 定义关键动作集合
        critical_actions = {"pickup", "put", "open", "close", "toggle", "slice"}
        
        # 如果包含关键动作，标记为 Landmark
        if any(act in action for act in critical_actions):
            return SEMANTIC_LANDMARK
    
    # 规则 3: 其他所有内容 → 过程噪声 (L2)
    # 包括导航 (navigate), 观察 (observe), 思考 (thought) 等
    return SEMANTIC_NOISE
```

### 3.2 辅助函数示例

```python
import re

def extract_action(content):
    """
    从 DecisionMaking 标签中提取动作文本。
    """
    match = re.search(r"<DecisionMaking>(.*?)</DecisionMaking>", content)
    if match:
        return match.group(1).lower()
    return ""
```

---

## 4. 数据构造结果

经过预处理管道后，输入模型的数据将包含两个对齐的张量。

### 4.1 张量结构
*   **`input_ids`**: 原始文本经过 Tokenizer 切分后的整数序列。
*   **`semantic_ids`**: 与 `input_ids` 维度完全一致的语义标签序列。

### 4.2 构造示例

假设一段对话如下：
1.  **System**: "You are a robot."
2.  **User**: "Put apple in drawer."
3.  **Assistant**: "navigate to table" (过程)
4.  **Assistant**: "pickup apple" (关键动作)

**模型输入视角的对齐展示**：

| 原始文本片段 | Token 序列 (input_ids) | 语义标签 (semantic_ids) | 类别说明 |
| :--- | :--- | :--- | :--- |
| `[SYS] You are a robot...` | `[1, 582, 338, 263, 892...]` | `[0, 0, 0, 0, 0...]` | **Global** |
| `[USER] Put apple in...` | `[2, 821, 442, 112...]` | `[0, 0, 0, 0...]` | **Global** |
| `[AST] navigate to...` | `[3, 991, 221, 552...]` | `[2, 2, 2, 2...]` | **Noise** |
| `[AST] pickup apple...` | `[3, 772, 442...]` | `[1, 1, 1...]` | **Landmark** |

> **注意**: 在实际实现中，`semantic_ids` 是通过将单条消息的 Label 广播 (Broadcast) 到该消息的所有 Token 上生成的。
