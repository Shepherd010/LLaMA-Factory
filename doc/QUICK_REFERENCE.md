# 数据生成三层架构 - 快速参考卡

## 🎯 一张图理解全部

```
层级3: 思维轨迹生成 o1StyleGenerate.py
       └─ 需要VLM (OpenAI/Claude/Qwen)
          ├─ API Key 配置: VLMCallapi_keys.py
          ├─ 成本: $0.01-0.40 per 轨迹
          ├─ 时间: 2-5分钟 per 轨迹
          └─ 输出: trajectory_0.json + images/
       
层级2: 任务模板生成 TaskGenerate.py
       └─ 无需VLM (纯Python逻辑)
          ├─ 时间: ~1小时 全部
          ├─ 成本: $0
          └─ 输出: {task_type}_task_metadata/

层级1: 场景元数据 taskgenerate/
       └─ AI2THOR 场景库
          ├─ 120个预置房间
          ├─ metadata.json (对象信息)
          └─ pick_up_and_put.json (兼容性规则)
```

---

## 快速配置 (5分钟)

### 步骤1: 获取API Key
```
访问 https://platform.openai.com/api-keys
创建新Key → 复制 (格式: sk-proj-xxxxx)
```

### 步骤2: 添加API Key
```python
# data_engine/VLMCallapi_keys.py
api_keys = ["sk-proj-your-key"]
```

### 步骤3: 运行测试
```bash
cd data_engine
python test_vlm.py  # 验证API连接
```

### 步骤4: 生成数据
```bash
python TaskGenerate.py           # 1小时，不需网络
python o1StyleGenerate.py        # 几天，需网络+VLM
```

---

## 三个关键文件

### ① taskgenerate/ - 场景库

**路径**: `z:\Code_Windows\embodied_reasoner\data_engine\taskgenerate\`

**包含**:
- `kitchens/FloorPlan{1-30}/metadata.json`
- `living_rooms/FloorPlan{201-230}/metadata.json`
- `bedrooms/FloorPlan{301-330}/metadata.json`
- `bathrooms/FloorPlan{401-430}/metadata.json`
- `pick_up_and_put.json` ← **物体兼容性表（可修改）**

**是什么**: 纯数据库，不需改动

---

### ② TaskGenerate.py - 任务生成器

**路径**: `z:\Code_Windows\embodied_reasoner\data_engine\TaskGenerate.py`

**核心参数** (第2350行):
```python
task_type = "single_search"  # 改这个选择任务类型

# 10种可选:
# "single_search", "single_search_from_closerep",
# "single_pickup", "single_pickup_from_closerep",
# "single_toggle", "pickup_and_put",
# "pickup_and_put_in_closerep", "pickup_from_closerep_and_put",
# "pickup_from_closerep_and_put_in_closerep",
# "ordered_pickup_two_object_and_put"
```

**是什么**: 纯规则引擎，**不需VLM**

---

### ③ o1StyleGenerate.py - 轨迹生成器

**路径**: `z:\Code_Windows\embodied_reasoner\data_engine\o1StyleGenerate.py`

**核心参数** (第2368行):
```python
model = "gpt-4o-2024-11-20"  # 改这个选择VLM模型

# 其他选项:
# "gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "claude-3-opus"

tasktype = "single_search"   # 改这个选择任务类型
```

**是什么**: VLM推理引擎，**需要API Key + 网络 + 钱**

---

## VLM 配置详解

### 配置文件: VLMCallapi_keys.py

```python
# 位置: data_engine/VLMCallapi_keys.py

# ❌ 错误方式
api_keys = []  # 空列表 → 会报错

# ✅ 正确方式1: 直接添加
api_keys = [
    "sk-proj-your-openai-key-here",
]

# ✅ 正确方式2: 从环境变量
import os
api_keys = [
    os.getenv("OPENAI_API_KEY"),
]

# ✅ 正确方式3: 从配置文件
import json
with open("config.json", "r") as f:
    config = json.load(f)
api_keys = config["keys"]
```

### API 端点修改: vlmCall.py

```python
# 当前 (第三方兼容API)
conn = http.client.HTTPSConnection("us.ifopen.ai")

# 改为官方OpenAI
conn = http.client.HTTPSConnection("api.openai.com")

# 改为阿里云DashScope (国内)
conn = http.client.HTTPSConnection("dashscope.aliyuncs.com")

# 改为本地部署 (免费)
conn = http.client.HTTPSConnection("localhost:8000")
```

---

## 10种任务类型速查表

| # | 任务类型 | 关键行动 | 复杂度 | 用途 |
|---|---------|--------|------|------|
| 1 | single_search | navigate→end | ⭐ | 寻找物体 |
| 2 | single_search_from_closerep | navigate→open→end | ⭐⭐ | 打开容器搜索 |
| 3 | single_pickup | navigate→pickup→end | ⭐ | 拿起物体 |
| 4 | single_pickup_from_closerep | navigate→open→pickup→close→end | ⭐⭐⭐ | 从容器拿出 |
| 5 | single_toggle | navigate→toggle→end | ⭐ | 切换开关 |
| 6 | pickup_and_put | navigate→pickup→navigate→put→end | ⭐⭐ | 转移物体 |
| 7 | pickup_and_put_in_closerep | navigate→pickup→navigate→open→put→end | ⭐⭐⭐ | 放入容器 |
| 8 | pickup_from_closerep_and_put | navigate→open→pickup→close→navigate→put→end | ⭐⭐⭐ | 从容器转移 |
| 9 | pickup_from_closerep_and_put_in_closerep | navigate→open→pickup→close→navigate→open→put→end | ⭐⭐⭐⭐ | 复杂转移 |
| 10 | ordered_pickup_two_object_and_put | ... (20+步) | ⭐⭐⭐⭐⭐ | 有序双物体 |

---

## VLM 模型选择指南

### 质量 vs 成本

```
质量越高 ────────────────────────────────────→ 成本越低

GPT-4-Turbo      GPT-4O         GPT-4O-Mini     本地开源
($0.40/K)        ($0.015/K)     ($0.0006/K)     ($0/本地)
最好              平衡            便宜             免费

推荐: 优先用 GPT-4O-Mini 生成完整数据集
    若质量不满意再用 GPT-4-Turbo 微调部分
```

### 模型特性对比

| 模型 | 输入价格 | 输出价格 | 速度 | 质量 | 中文支持 |
|------|---------|---------|------|------|---------|
| GPT-4-Turbo | $0.01 | $0.03 | 快 | 🌟🌟🌟🌟🌟 | ✅ |
| GPT-4O | $0.005 | $0.015 | 快 | 🌟🌟🌟🌟 | ✅ |
| GPT-4O-Mini | $0.00015 | $0.0006 | 快 | 🌟🌟🌟 | ✅ |
| Claude 3 | $0.015 | $0.075 | 中等 | 🌟🌟🌟🌟 | ✅ |
| Qwen-VL-Max | $0.002 | $0.002 | 快 | 🌟🌟🌟 | 🌟🌟🌟 |
| Llava (本地) | $0/本地 | $0/本地 | 慢 | 🌟🌟 | ✅ |

---

## 成本计算器

### 生成完整数据集的成本

```
参数:
- 总轨迹数: 9,300条
- 每条平均步数: 10步
- 每步平均VLM调用: 2次
- 每次调用平均token: 1,000 token

总VLM调用数:
= 9,300 × 10 × 2 = 186,000 次调用

总Token数 (估算):
= 186,000 × 1,000 = 186M token
= 平均分配 60% input + 40% output
= 111.6M input + 74.4M output

按模型计算成本:
┌──────────────────────────────────────┐
│ GPT-4-Turbo                          │
│ = 111.6M × $0.01 + 74.4M × $0.03    │
│ = $1,116 + $2,232                    │
│ = $3,348 ≈ 22,000 RMB ⚠️⚠️⚠️         │
├──────────────────────────────────────┤
│ GPT-4O                               │
│ = 111.6M × $0.005 + 74.4M × $0.015  │
│ = $558 + $1,116                      │
│ = $1,674 ≈ 11,000 RMB ⚠️             │
├──────────────────────────────────────┤
│ GPT-4O-Mini                          │
│ = 111.6M × $0.00015 + 74.4M × $0.0006│
│ = $16.74 + $44.64                    │
│ = $61 ≈ 400 RMB ✅ 推荐               │
├──────────────────────────────────────┤
│ Qwen-VL (国内)                       │
│ = 185.6M × $0.002                    │
│ = $371 ≈ 2,400 RMB ✅                │
├──────────────────────────────────────┤
│ 本地开源模型                          │
│ = $0 (但需GPU时间，约 5-10万RMB)     │
└──────────────────────────────────────┘

💡 建议: 
- 预算充足: GPT-4-Turbo (质量最好)
- 平衡方案: GPT-4O (综合最优)
- 预算紧张: GPT-4O-Mini (3分之1价格)
- 国内部署: 阿里云Qwen (支持商用)
```

---

## 常见问题速查

| 问题 | 答案 |
|------|------|
| TaskGenerate.py 需要VLM吗? | ❌ 否，纯逻辑 |
| o1StyleGenerate.py 需要VLM吗? | ✅ 是，必需 |
| taskgenerate/ 需要修改吗? | ❌ 否，原始数据 |
| VLMCallapi_keys.py 在哪? | `data_engine/VLMCallapi_keys.py` |
| 没有API Key可以运行吗? | TaskGenerate.py ✅，o1StyleGenerate.py ❌ |
| 能用国内VLM吗? | ✅ 可以，改 vlmCall.py 的端点 |
| 能用本地模型吗? | ✅ 可以，但需GPU(24GB+) |
| 生成一条轨迹多少钱? | $0.01-0.40 (取决于模型) |
| 一条轨迹多少时间? | 2-5分钟 (取决于步数) |

---

## 三条命令一键生成

```bash
# 全部默认配置
cd data_engine

# 1️⃣ 生成任务模板 (无需VLM，~1小时)
python TaskGenerate.py

# 2️⃣ 生成轨迹数据 (需VLM，会问询)
python o1StyleGenerate.py

# 3️⃣ 训练模型 (需LLaMA-Factory)
cd ..
bash scripts/train.sh
```

---

## 最后的最后

✅ **本质理解**:
- 层级1 = 场景资产库 (无需动)
- 层级2 = 逻辑规则引擎 (不需钱)
- 层级3 = VLM推理引擎 (需钱🤑)

✅ **快速开始**:
1. 获取API Key
2. 改2个配置文件
3. 跑2个Python脚本
4. 等待数据生成

✅ **更多文档**:
- 详细说明: `DATA_GENERATION_DETAILED.md`
- 配置指南: `VLM_CONFIG_GUIDE.md`
- 可视化图: `DATA_PIPELINE_VISUAL.md`
- 完整架构: `PROJECT_ARCHITECTURE.md`
