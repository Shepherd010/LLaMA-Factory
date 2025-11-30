# Qwen3-VL-4B LoRA微调完全指南

## 🚀 快速开始（3分钟上手）

### 1. 环境准备
```bash
# 确保已安装LLaMA-Factory
cd /home/worku22/LLaMA-Factory
pip install -e ".[torch,metrics]"
```

### 2. 一键启动训练
```bash
# 基础训练命令
llamafactory-cli train examples/train_lora/qwen3_vl_4b_lora_sft.yaml

# 或使用Python直接运行
python src/train.py examples/train_lora/qwen3_vl_4b_lora_sft.yaml
```

---

## ⚡ 速度优化全攻略

### 核心优化参数说明

| 参数 | 作用 | 推荐值 | 说明 |
|------|------|--------|------|
| `overwrite_cache` | 缓存复用 | `false` | **首要优化项**，关闭后复用tokenizer缓存 |
| `preprocessing_num_workers` | 预处理并行 | CPU核心数×70% | 16核CPU设为12 |
| `dataloader_num_workers` | 数据加载并行 | 4-8 | 过大可能OOM |
| `streaming` | 流式加载 | 小数据集false | >100万样本开启 |
| `image_max_pixels` | 图像分辨率 | 262144(512×512) | 降低可加速但影响质量 |

### 场景化配置推荐

#### 场景1：小数据集（<1万条）- 追求简单快速
```yaml
streaming: false
overwrite_cache: false
preprocessing_num_workers: 8
dataloader_num_workers: 2
```

#### 场景2：中等数据集（1-100万条）- 平衡速度和内存
```yaml
streaming: false
overwrite_cache: false
preprocessing_num_workers: 16
dataloader_num_workers: 4
preprocessing_batch_size: 2000
tokenized_path: ./cache/my_dataset  # 指定缓存路径
```

#### 场景3：大规模数据集（>100万条）- 防止OOM
```yaml
streaming: true
overwrite_cache: false
preprocessing_num_workers: 16
dataloader_num_workers: 4
buffer_size: 65536
preprocessing_batch_size: 4000
mix_strategy: interleave_under
```

---

## 📁 数据集准备

### 1. 数据格式示例
在 `data/` 目录下创建你的数据集JSON文件：

```json
[
  {
    "messages": [
      {"role": "user", "content": "<image>请描述这张图片"},
      {"role": "assistant", "content": "这是一张..."}
    ],
    "images": ["path/to/image1.jpg"]
  }
]
```

### 2. 注册数据集
编辑 `data/dataset_info.json`，添加：

```json
{
  "my_dataset": {
    "file_name": "my_dataset.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    }
  }
}
```

### 3. 修改配置文件
```yaml
dataset: my_dataset  # 使用你注册的数据集名称
```

---

## 🔧 常用命令

### 单卡训练
```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/qwen3_vl_4b_lora_sft.yaml
```

### 多卡训练（DDP）
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train examples/train_lora/qwen3_vl_4b_lora_sft.yaml
```

### 指定参数覆盖配置
```bash
llamafactory-cli train examples/train_lora/qwen3_vl_4b_lora_sft.yaml \
  --num_train_epochs 5 \
  --learning_rate 5e-5 \
  --output_dir saves/my_experiment
```

### 断点续训
```bash
llamafactory-cli train examples/train_lora/qwen3_vl_4b_lora_sft.yaml \
  --resume_from_checkpoint saves/qwen3_vl-4b/lora/sft/checkpoint-500
```

---

## 💡 常见问题解决

### Q1: Tokenizer加载慢（5-6分钟）
**原因**：每次重新处理数据  
**解决**：
```yaml
overwrite_cache: false  # 关键！
```

### Q2: 内存溢出(OOM)
**解决方案**：
```yaml
per_device_train_batch_size: 1  # 降低batch size
gradient_accumulation_steps: 16  # 增大梯度累积
image_max_pixels: 131072  # 降低图像分辨率(256×512)
streaming: true  # 开启流式加载
```

### Q3: 训练速度慢
**检查清单**：
1. 确认使用了 `bf16: true`
2. 检查GPU利用率：`nvidia-smi -l 1`
3. 增加 `dataloader_num_workers`
4. 降低 `image_max_pixels`

### Q4: get_rope_index shape mismatch 错误
**原因**：cutoff_len截断了视觉token  
**解决**：增大cutoff_len或降低image_max_pixels
```yaml
cutoff_len: 4096  # 增大序列长度
image_max_pixels: 131072  # 或降低图像分辨率
```

---

## 📊 训练监控

### 查看训练日志
```bash
tail -f saves/qwen3_vl-4b/lora/sft/trainer_log.jsonl
```

### 查看loss曲线
训练完成后，loss图像保存在：
```
saves/qwen3_vl-4b/lora/sft/training_loss.png
```

### 使用TensorBoard
```yaml
report_to: tensorboard
```
```bash
tensorboard --logdir saves/qwen3_vl-4b/lora/sft
```

---

## 🎯 训练完成后

### 合并LoRA权重
```bash
llamafactory-cli export \
  --model_name_or_path Qwen/Qwen3-VL-4B-Instruct \
  --adapter_name_or_path saves/qwen3_vl-4b/lora/sft \
  --template qwen3_vl \
  --export_dir models/qwen3_vl_merged
```

### 测试对话
```bash
llamafactory-cli chat \
  --model_name_or_path Qwen/Qwen3-VL-4B-Instruct \
  --adapter_name_or_path saves/qwen3_vl-4b/lora/sft \
  --template qwen3_vl
```

---

## 📋 配置参数速查表

### 必须配置
| 参数 | 说明 |
|------|------|
| `model_name_or_path` | 模型路径 |
| `dataset` | 数据集名称 |
| `template` | 对话模板（qwen3_vl） |
| `output_dir` | 输出目录 |

### 速度相关
| 参数 | 默认值 | 加速建议 |
|------|--------|----------|
| `overwrite_cache` | true | **改为false** |
| `preprocessing_num_workers` | 1 | 改为16 |
| `bf16` | false | **改为true** |
| `image_max_pixels` | 768×768 | 可降至512×512 |

### 内存相关
| 参数 | OOM时调整 |
|------|-----------|
| `per_device_train_batch_size` | 降低为1 |
| `gradient_accumulation_steps` | 增大 |
| `cutoff_len` | 降低 |
| `streaming` | 改为true |

---

## 🎉 祝你训练顺利！

如有问题，可以：
1. 查看 `saves/*/trainer_log.jsonl` 日志
2. 检查GPU内存：`nvidia-smi`
3. 调整配置参数重试
