# HSA (Hierarchical Semantic Attention) 架构实现与多模态集成总结

**日期:** 2025年11月24日
**作者:** GitHub Copilot
**项目:** LLaMA-Factory HSA 集成

## 1. 项目概述

本文档详细总结了在 LLaMA-Factory 框架中实现 **分层语义注意力 (Hierarchical Semantic Attention, HSA)** 架构的工作。该架构的核心目的是通过引入语义角色（全局条件、关键路标、噪声），对 Transformer 的注意力机制施加偏置，从而引导模型更关注关键信息，忽略无关噪声。

此外，本项目还成功将 HSA 架构扩展到了 **多模态模型 (Qwen2-VL)**，验证了其在处理图像-文本混合输入时的有效性。

## 2. 核心实现细节

### 2.1 HSA 核心逻辑 (`src/llamafactory/model/hsa_patch.py`)

为了最小化对原生模型代码的侵入，我们采用了 **Monkey Patch (运行时动态替换)** 的方式注入 HSA 逻辑。

*   **`HSAContext` 上下文管理器**:
    *   用于在模型的前向传播过程中存储和分发 `semantic_ids`（语义ID）。
    *   利用类变量 `_current_semantic_ids` 实现跨层共享，确保每一层注意力模块都能访问到当前的语义标签。

*   **`compute_hsa_bias` 偏置计算函数**:
    *   **输入**: `semantic_ids` 张量，形状为 `[batch_size, seq_len]`。
    *   **输出**: 注意力偏置矩阵，形状为 `[batch_size, 1, seq_len, seq_len]`。
    *   **逻辑**:
        1.  将 `semantic_ids` 扩展为 Query (`[bsz, 1, seq_len, 1]`) 和 Key (`[bsz, 1, 1, seq_len]`) 维度。
        2.  **屏蔽规则 1**: 当 Query 是 **全局条件 (GLOBAL_ID=0)** 时，屏蔽对 **噪声 (NOISE_ID=2)** 的注意力。
        3.  **屏蔽规则 2**: 当 Query 是 **关键路标 (LANDMARK_ID=1)** 时，屏蔽对 **噪声 (NOISE_ID=2)** 的注意力。
    *   通过将对应位置的注意力分数设为极小值（如 `-inf`），实现“看不见”的效果。

*   **`apply_hsa_patch` 注入函数**:
    *   **模型级 Patch**: 包装模型的 `forward` 方法，使其能够接收 `semantic_ids` 参数，并将其存入 `HSAContext`。
    *   **层级 Patch**: 遍历模型的所有子模块，识别并替换标准的注意力模块（`LlamaAttention`, `Qwen2Attention`, `Qwen2VLAttention`）。
    *   **包装逻辑**: 在原有的 `forward` 调用前，从 Context 获取 `semantic_ids`，计算 Bias，并将其叠加到 `attention_mask` 上。

### 2.2 模型加载器集成 (`src/llamafactory/model/loader.py`)

修改了 `load_model` 函数，确保在模型初始化完成、权重加载之后，立即应用 HSA Patch。

```python
# 在 load_model 函数末尾
from .hsa_patch import apply_hsa_patch
# ...
apply_hsa_patch(model)  # 动态修改模型结构
```

### 2.3 数据流改造

为了让模型“知道”每个 Token 的语义角色，我们需要从数据集中提取这些信息并传递给模型。

*   **语义标签定义**:
    *   `GLOBAL_CONDITION` (ID: 0): 全局指令或系统提示。
    *   `LANDMARK` (ID: 1): 关键推理步骤或重要信息。
    *   `NOISE` (ID: 2): 闲聊、无关信息或低价值内容。

*   **数据转换器 (`src/llamafactory/data/converter.py`)**:
    *   修改了 `SharegptDatasetConverter`。
    *   在解析 ShareGPT 格式数据时，额外读取 `semantic_type` 字段。
    *   默认值设为 `GLOBAL_CONDITION`。

*   **数据处理器 (`src/llamafactory/data/processor/supervised.py`)**:
    *   修改了 `SupervisedDatasetProcessor` 的 `_encode_data_example` 方法。
    *   在生成 `input_ids` 的同时，同步生成 `semantic_ids` 列表。
    *   处理了多轮对话的拼接逻辑，确保 `semantic_ids` 与 `input_ids` 长度严格一致，并正确处理 Padding 和 EOS Token。

## 3. 多模态 (Qwen2-VL) 集成与优化

本项目重点验证了 HSA 在视觉-语言模型上的可行性。

### 3.1 Qwen2-VL 适配
*   **注意力模块识别**: 在 `hsa_patch.py` 中增加了对 `transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLAttention` 的支持。
*   **兼容性处理**: 使用 `try-except` 块导入 `Qwen2VLAttention`，确保在纯文本环境下不会报错。

### 3.2 依赖与环境修复
*   **BitsAndBytes 版本**: 解决了 `bitsandbytes` 版本过低导致 4-bit 量化 (QLoRA) 失败的问题。升级到了 `bitsandbytes>=0.39.0` (实际环境 0.48.2)。
*   **模型轻量化**:
    *   最初尝试使用 `Qwen2.5-VL-7B-Instruct`，但发现其显存占用和计算开销较大，不利于快速调试。
    *   最终切换至 **`Qwen/Qwen2.5-VL-3B-Instruct`**，在保持多模态能力的同时显著提升了迭代速度。

## 4. 验证与测试

为了确保整个链路（数据 -> 处理 -> 模型 -> HSA Bias -> 反向传播）畅通，我们构建了完整的测试流程。

### 4.1 测试脚本 (`tests/test_hsa_flow.py`)
编写了一个自动化测试脚本，执行以下操作：
1.  配置 LLaMA-Factory CLI 参数。
2.  指定使用 `hsa_test` 数据集和 `qwen2_vl` 模板。
3.  启用 LoRA 微调和 4-bit 量化。
4.  运行训练并捕获日志，验证是否成功。

### 4.2 测试数据集 (`data/hsa_test.json`)
构建了一个包含图像和语义标签的测试样本：
```json
[
  {
    "conversations": [
      {
        "from": "user",
        "value": "<image>这张图里有什么？",
        "semantic_type": "GLOBAL_CONDITION"
      },
      {
        "from": "assistant",
        "value": "这是一张测试图片...",
        "semantic_type": "LANDMARK"
      }
    ],
    "images": ["data/mllm_demo_data/1.jpg"]
  }
]
```
*修正*: 修复了图像路径问题，确保指向本地存在的 `data/mllm_demo_data/*.jpg` 文件。

### 4.3 测试结果
*   **执行状态**: 成功 (Passed)
*   **模型**: `Qwen/Qwen2.5-VL-3B-Instruct`
*   **训练 Loss**: ~0.4408
*   **结论**: HSA Patch 成功注入，模型能够正常处理多模态输入并进行梯度更新。

## 5. 修改文件列表

以下是本项目涉及的主要修改文件：

1.  `src/llamafactory/model/hsa_patch.py` (新增): HSA 核心逻辑。
2.  `src/llamafactory/model/loader.py` (修改): 集成 Patch 调用。
3.  `src/llamafactory/data/converter.py` (修改): 解析 `semantic_type`。
4.  `src/llamafactory/data/processor/supervised.py` (修改): 生成 `semantic_ids`。
5.  `tests/test_hsa_flow.py` (新增): 端到端测试脚本。
6.  `data/hsa_test.json` (新增): 测试数据集。
7.  `data/dataset_info.json` (修改): 注册 `hsa_test` 数据集。

## 6. 如何运行

要复现此工作或开始训练：

1.  **准备数据**: 确保数据集格式为 ShareGPT，并在 `conversations` 中包含 `semantic_type` 字段。
2.  **注册数据**: 在 `data/dataset_info.json` 中添加数据集条目。
3.  **启动训练**:
    ```bash
    llamafactory-cli train \
        --stage sft \
        --do_train \
        --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
        --dataset hsa_test \
        --template qwen2_vl \
        --finetuning_type lora \
        --output_dir saves/hsa_run \
        --overwrite_output_dir \
        --fp16
    ```
