# HSA 与 Feedback 机制迁移与测试指南

**目标**: 将现有的 **HSA (Hierarchical Semantic Attention)** 架构与 **Feedback (反射轨迹反馈)** 机制迁移至更高版本的 LLaMA-Factory (支持 Qwen3/Qwen2.5-VL 等新模型)，并确保两者在同一个代码库中完美融合。

**背景**:
1.  **HSA**: 通过 `semantic_ids` 对注意力机制施加偏置，区分 Global/Landmark/Noise token。
2.  **Feedback**: 在迭代训练的第 3 阶段，仅训练 `<|feedback|>` 标记之后的反射轨迹。需要在 Data Collator 中动态屏蔽该标记之前的 Label。

---

## 1. 核心组件迁移清单

请按照以下顺序逐一迁移和修改文件。

### 1.1 模型层 (Model Layer)

#### **文件**: `src/llamafactory/model/hsa_patch.py` (新增)
*   **操作**: 将原有的 `hsa_patch.py` 完整复制到新版本。
*   **适配检查**:
    *   检查 `transformers` 库中新模型（如 Qwen3, Qwen2.5-VL）的 Attention 类名。
    *   确保 `apply_hsa_patch` 函数中的 `target_classes` 包含了新模型的 Attention 类 (例如 `Qwen2VLAttention`, `Qwen2_5_VLAttention` 或未来的 `Qwen3Attention`)。
    *   **关键代码**:
        ```python
        # 确保包含所有目标 Attention 类
        target_classes = (LlamaAttention, Qwen2Attention, Qwen2VLAttention)
        ```

#### **文件**: `src/llamafactory/model/loader.py` (修改)
*   **操作**: 在 `load_model` 函数的末尾（返回 model 之前）注入 HSA Patch。
*   **代码**:
    ```python
    from .hsa_patch import apply_hsa_patch
    # ... 在 load_model 函数结束前 ...
    apply_hsa_patch(model)
    return model
    ```

### 1.2 数据处理层 (Data Processing Layer)

#### **文件**: `src/llamafactory/data/converter.py` (修改)
*   **操作**: 修改 `SharegptDatasetConverter` 类。
*   **逻辑**: 在解析 `messages` 时，提取 `semantic_type` 字段。
*   **代码片段**:
    ```python
    # 在 SharegptDatasetConverter.__call__ 中
    aligned_messages.append({
        "role": tag_mapping[message[self.dataset_attr.role_tag]],
        "content": message[self.dataset_attr.content_tag],
        # 新增: 提取语义类型，默认为 GLOBAL_CONDITION
        "semantic_type": message.get(self.dataset_attr.semantic_tag, "GLOBAL_CONDITION"),
    })
    ```

#### **文件**: `src/llamafactory/data/processor/supervised.py` (修改)
*   **操作**: 修改 `SupervisedDatasetProcessor._encode_data_example`。
*   **逻辑**:
    1.  建立 `semantic_type` 到 ID 的映射 (Global=0, Landmark=1, Noise=2)。
    2.  在构建 `input_ids` 的循环中，同步构建 `semantic_ids`。
    3.  确保 `semantic_ids` 的长度、截断、Padding 逻辑与 `input_ids` 完全一致。
    4.  **注意**: 多轮对话拼接时，`semantic_ids` 也要正确拼接。

### 1.3 数据整理层 (Data Collator) - **核心融合点**

这是 HSA 和 Feedback 机制结合的关键位置。你需要修改 `src/llamafactory/data/collator.py` 中的 `MultiModalDataCollatorForSeq2Seq` 类。

#### **文件**: `src/llamafactory/data/collator.py` (修改)

**任务**: 将 Feedback 的 Label Masking 逻辑和 HSA 的 Semantic ID Padding 逻辑合并到 `__call__` 方法中。

**融合逻辑伪代码**:

```python
@dataclass
class MultiModalDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    # ... existing init ...

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        
        # --- [Feedback 机制集成] ---
        # 逻辑: 如果 input_ids 中包含 <|feedback|>，则将该 token 及其之前的所有 labels 设为 IGNORE_INDEX
        # 注意: 需确保 tokenizer 已包含此特殊 token，或者硬编码 ID (不推荐)，建议动态获取
        try:
            feedback_id = self.tokenizer.convert_tokens_to_ids("<|feedback|>")
            # 防止 tokenizer 没有该 token 时报错或返回 unk
            if feedback_id is not None and feedback_id != self.tokenizer.unk_token_id:
                for feature in features:
                    if feedback_id in feature['input_ids']:
                        feedback_idx = feature['input_ids'].index(feedback_id)
                        # Mask 掉 feedback token 及其之前的所有 label
                        for i in range(len(feature['labels'])):
                            if i <= feedback_idx:
                                feature['labels'][i] = IGNORE_INDEX
        except Exception as e:
            # 记录日志或忽略，视 tokenizer 配置而定
            pass

        # --- [HSA 机制集成: 提取 semantic_ids] ---
        # 在父类处理之前，先 pop 出 semantic_ids，防止 DataCollatorForSeq2Seq 报错
        semantic_ids_batch = [feature.pop("semantic_ids", None) for feature in features]

        # --- [标准处理] ---
        # 调用父类或原有逻辑处理 images/videos/audios 和 input_ids padding
        # ... (保留原有的多模态处理代码) ...
        features = super().__call__(features) # 或原有的手动处理逻辑

        # --- [HSA 机制集成: Padding semantic_ids] ---
        # 逻辑: 根据 input_ids 的 padding 情况，对 semantic_ids 进行同样的 padding (通常填 2/NOISE 或 0/GLOBAL)
        if semantic_ids_batch[0] is not None:
            padded_input_ids = features["input_ids"]
            bsz, seq_len = padded_input_ids.shape
            padded_semantic_ids = []
            
            for i, sem_ids in enumerate(semantic_ids_batch):
                if sem_ids is None: sem_ids = []
                
                # 计算需要 Pad 的长度
                # 注意: 需考虑 padding_side (left/right)
                # 注意: 如果有多模态 fake tokens (image placeholder)，也要同步处理 semantic_ids
                
                # ... (参考旧版 collator.py 中的 padding 逻辑) ...
                
                padded_semantic_ids.append(processed_sem_ids)
            
            features["semantic_ids"] = torch.tensor(padded_semantic_ids, dtype=torch.long)

        return features
```

---

## 2. 测试验证计划

迁移完成后，必须运行以下测试以确保两个机制正常工作。

### 2.1 测试脚本: `tests/test_hsa_feedback_migration.py`

请创建一个新的测试脚本，包含以下两个测试用例。

#### **测试用例 1: Feedback 机制验证**
*   **输入**: 构造一条包含 `<|feedback|>` token 的数据样本。
*   **预期**:
    *   Data Collator 返回的 `labels` 中，`<|feedback|>` 及其左侧的数值应全为 `-100` (IGNORE_INDEX)。
    *   `<|feedback|>` 右侧的 `labels` 应保留原值。

#### **测试用例 2: HSA 机制验证**
*   **输入**: 构造一条包含 `semantic_type` 的数据样本。
*   **预期**:
    *   Data Collator 返回的 `features` 中包含 `semantic_ids` 字段。
    *   `semantic_ids` 的形状与 `input_ids` 一致。
    *   运行一次模型 Forward (使用 `hsa_patch` 后的模型)，不应报错，且 Loss 正常计算。

### 2.2 运行命令
```bash
# 1. 确保环境准备就绪
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate llama-factory

# 2. 运行测试脚本
python tests/test_hsa_feedback_migration.py
```

---

## 3. 注意事项

1.  **Tokenizer**: 确保新模型的 Tokenizer 中添加了 `<|feedback|>` 特殊 Token，否则 `convert_tokens_to_ids` 可能失败。可以在 `template` 定义中添加，或者在微调前手动 resize tokenizer。
2.  **Qwen2.5-VL/Qwen3 兼容性**: 新模型可能引入了新的 Attention 实现（如 FlashAttention 变体）。务必检查 `hsa_patch.py` 是否能正确 hook 到这些新模块。
3.  **Padding Side**: LLaMA-Factory 可能会根据模型类型自动调整 padding side (left/right)。`collator.py` 中的 `semantic_ids` padding 逻辑必须动态适配 `tokenizer.padding_side`。

---

**致 Coder-LLM**:
请严格按照上述指南操作。你的首要任务是**合并** `converter_feedback.py` 中的 Feedback 逻辑和原有的 HSA 逻辑到新的 `collator.py` 中，并确保其他配套文件 (`converter.py`, `supervised.py`, `hsa_patch.py`) 同步更新。完成代码修改后，请务必运行测试脚本验证。
