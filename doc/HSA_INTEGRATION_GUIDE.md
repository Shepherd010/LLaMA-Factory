# HSA 集成指南 (Integration Guide)

本指南将指导您如何将 HSA 机制集成到现有的 LLaMA-Factory 代码库中。

## 0. 准备工作

### 数据集格式要求
确保您的数据集 JSON 文件中，`messages` 列表里的每个对象都包含 `semantic_type` 字段。
有效值为：
*   `GLOBAL_CONDITION`
*   `KEY_EVENT`
*   `PROCESS_NOISE` (默认)

如果缺少此字段，系统将默认将其视为 `PROCESS_NOISE`。

---

## 1. 代码修改清单

您需要修改或创建以下文件：

1.  `src/llamafactory/data/preprocess.py` (修改)
2.  `src/llamafactory/data/collator.py` (修改)
3.  `src/llamafactory/model/hsa_patch.py` (新建)
4.  `src/llamafactory/model/patcher.py` (修改)

### 1.1 修改 `preprocess.py`

**目标**: 生成 `semantic_ids`。

找到 `preprocess_supervised_dataset` 函数。在构建 `model_inputs` 的循环中，**增加环境变量检查**：

```python
import os

# 定义常量映射
SEMANTIC_MAP = {
    "GLOBAL_CONDITION": 0,
    "KEY_EVENT": 1,
    "PROCESS_NOISE": 2
}

# 检查开关
use_hsa = os.environ.get("USE_HSA", "false").lower() == "true"

# ... 在处理 messages 的循环内 ...
semantic_ids = []
for msg in messages:
    # 仅在开启 HSA 时处理，否则跳过或生成默认值
    if use_hsa:
        # 获取类型 ID
        sem_type = SEMANTIC_MAP.get(msg.get("semantic_type"), 2)
        
        # 获取 Token 长度 (需与 tokenizer.encode 逻辑一致)
        msg_tokens = tokenizer.encode(msg["content"], add_special_tokens=False)
        
        # 广播 ID
        semantic_ids.extend([sem_type] * len(msg_tokens))
    else:
        # 如果未开启，可以选择不生成 semantic_ids，或者生成全 2
        pass

# 将 semantic_ids 添加到 model_inputs (仅当 use_hsa 为 True 时)
if use_hsa:
    model_inputs["semantic_ids"].append(semantic_ids)
```

### 1.2 修改 `collator.py`

**目标**: Padding `semantic_ids`。

在 `DataCollatorForSeq2Seq` 的 `__call__` 方法中：

```python
# ... 现有代码 ...
features = super().__call__(features, return_tensors) # 或手动处理

if "semantic_ids" in features[0]:
    # 提取列表
    sem_ids = [torch.tensor(f["semantic_ids"], dtype=torch.long) for f in features]
    # Padding (Value = 2)
    padded_sem_ids = torch.nn.utils.rnn.pad_sequence(
        sem_ids, batch_first=True, padding_value=2
    )
    # 存回 features
    features["semantic_ids"] = padded_sem_ids
```

### 1.3 新建 `hsa_patch.py`

**目标**: 实现 Attention Bias 计算逻辑。

```python
import torch
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLAttention

def apply_hsa_patch(model):
    # 定义新的 forward 函数
    original_forward = Qwen2VLAttention.forward
    
    def hsa_forward(self, hidden_states, attention_mask=None, **kwargs):
        semantic_ids = kwargs.get("semantic_ids")
        
        if semantic_ids is not None:
            # 1. 计算 Bias Matrix (参考 HSA_ARCHITECTURE.md 中的公式)
            # ... 实现代码 ...
            
            # 2. 融合 Mask
            if attention_mask is None:
                attention_mask = bias_matrix
            else:
                attention_mask = attention_mask + bias_matrix
                
        return original_forward(self, hidden_states, attention_mask=attention_mask, **kwargs)

    # 应用 Patch
    Qwen2VLAttention.forward = hsa_forward
    print("HSA Patch Applied!")
```

### 1.4 修改 `patcher.py`

**目标**: 在模型加载时激活 Patch。

在 `patch_model` 函数末尾，**增加环境变量检查**：

```python
import os
from .hsa_patch import apply_hsa_patch

def patch_model(...):
    # ... 现有逻辑 ...
    
    # 激活 HSA (仅当环境变量 USE_HSA=true 时)
    if os.environ.get("USE_HSA", "false").lower() == "true":
        print("[HSA] Enabling Hierarchical Semantic Attention...")
        apply_hsa_patch(model)
    
    return model
```

---

## 2. 训练配置 (Training Configuration)

在启动训练时，必须配置以下参数：

### 2.1 `remove_unused_columns`
**必须设置为 `False`**。
*   原因：HuggingFace Trainer 默认会检查模型签名，移除 Dataset 中多余的列。由于我们是通过 `kwargs` 隐式传递 `semantic_ids`，如果不关闭此选项，`semantic_ids` 会在进入模型前被丢弃。

### 2.2 Attention Implementation
**建议设置为 `eager`** (或确保 Patch 兼容 SDPA)。
*   原因：Flash Attention 2 (`flash_attn_varlen_func`) 接口通常不支持加性 Bias 矩阵。
*   设置方法：在 `TrainingArguments` 或命令行中指定 `--attn_implementation eager` (具体参数名视 LLaMA-Factory 版本而定，通常是 `--flash_attn disabled` 或类似)。

---

## 3. 验证步骤

1.  **数据检查**: 运行预处理脚本，检查生成的缓存文件或打印 `model_inputs`，确认 `semantic_ids` 存在且长度正确。
2.  **前向测试**: 编写一个简单的推理脚本，加载 Patch 后的模型，输入带有 `semantic_ids` 的数据，观察是否报错。
3.  **显存监控**: 启用 HSA 后，显存占用会增加（因为 $N \times N$ 矩阵）。请密切关注 OOM 风险，必要时减小 Batch Size 或 Sequence Length。

---

## 4. Web UI 使用指南

如果您习惯使用 Web UI (`src/webui.py`) 进行训练，**无需修改 Web UI 的任何代码**。我们通过环境变量来控制 HSA 的开启与关闭。

### 4.1 启动方式

在启动 Web UI 前，设置 `USE_HSA` 环境变量。

#### Windows (PowerShell)
```powershell
# 开启 HSA
$env:USE_HSA = "true"
python src/webui.py
```

#### Windows (CMD)
```cmd
:: 开启 HSA
set USE_HSA=true
python src/webui.py
```

#### Linux / Mac
```bash
# 开启 HSA
export USE_HSA=true
python src/webui.py
```

### 4.2 关闭 HSA
如果不设置该变量，或者将其设为 `false`，系统将以原生 LLaMA-Factory 模式运行，完全不受 HSA 代码影响。

```powershell
# 关闭 HSA
$env:USE_HSA = "false"
python src/webui.py
```

### 4.3 验证是否生效
启动训练后，观察终端（Console）输出。如果看到类似以下的日志，说明 HSA 已成功注入：
```text
[HSA] Enabling Hierarchical Semantic Attention...
```
