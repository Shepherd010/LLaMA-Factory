# Embodied-Reasoner 项目架构深度解读

## 🎯 项目核心使命

**Embodied-Reasoner** 是一个将 **o1风格深度推理能力** 扩展到 **具身交互任务** 的多模态模型。核心目标是让 AI 智能体在 AI2THOR 虚拟环境中执行复杂的长期交互任务（搜索隐藏物体、操纵和运输物品等）。

### 关键特性：
- ✅ 深度推理：分析、空间推理、反思、规划
- ✅ 多模态处理：长序列的图像-文本交错
- ✅ 环境交互：自主观察、探索、寻找隐藏物体
- ✅ 开源数据集：9.3k 轨迹、64K 图像、8M 思维token

---

## 📦 项目模块架构图

```
embodied_reasoner/
├── data_engine/          # 数据合成引擎 (任务+轨迹生成)
├── evaluate/             # 评估框架 (模型推理+性能评估)
├── inference/            # 推理服务 (模型部署)
├── finetune/             # 微调配置 (LoRA + 全参)
├── scripts/              # 执行脚本
└── data/                 # 数据资产 (测试集、训练集)
```

---

## 🔧 各模块详细解读

### 1️⃣ **data_engine/** - 数据合成引擎 (核心逻辑)

**目的**：自动合成 **观察-思考-行动** 轨迹数据集，用于模型训练。

#### 架构流程图：
```
TaskGenerate.py (任务模板生成)
          ↓
        [任务元数据] {taskname, key_actions, task_metadata}
          ↓
o1StyleGenerate.py (o1风格轨迹生成)
          ↓
        [思维轨迹] {观察→思考→行动→观察→...}
          ↓
[最终数据集] {图像序列 + 文本推理 + 行动}
```

#### 1.1 **TaskGenerate.py** - 任务生成器
**功能**：在 AI2THOR 场景中生成多样化的任务模板。

**关键逻辑**：
```
对每个场景 (120个AI2THOR房间):
  1. 加载场景元数据 (对象、容器、交互属性)
  2. 根据 task_type 筛选合适的对象:
     - single_search: 寻找单个物体
     - two_search: 寻找两个物体
     - pickup_and_put: 拿起并放入容器
     - ordered_pickup: 有序操纵多个物体
     ...等10种子任务
  3. 定义"关键行动序列":
     - navigate_to(位置)
     - open(容器)
     - pickup(物体)
     - put_in(容器)
     - end
  4. 为每个行动定义 reward=1，总reward=总步数
  5. 保存到 JSON 格式的任务元数据
```

**输出格式**：
```json
{
  "taskname": "Locate the Apple in the room.",
  "tasktype": "single_search",
  "actions": [
    {"action": "navigate to", "objectId": "CounterTop|...", "reward": 1},
    {"action": "end", "reward": 1}
  ],
  "totalreward": 2
}
```

#### 1.2 **o1StyleGenerate.py** - o1风格思维轨迹生成
**功能**：为任务生成包含深度推理的轨迹。这是项目的 **创新核心**。

**关键逻辑** (三层循环)：

**第1层 - 任务执行循环**：
```python
for step in range(max_steps):
    当前状态 = 观察(智能体位置, 摄像头方向, 手中物体)
    captured_image = 保存截图()
```

**第2层 - VLM调用触发 (思维生成)**：
```python
# 关键决策点：何时需要思考？
if step_type in ["decision_point", "search", "reasoning_needed"]:
    # 调用VLM生成思维
    思维内容 = VLM.request(
        system_prompt="你是一个具身推理智能体...",
        user_prompt=f"当前观察: {description}. 下一步应该怎么做?",
        images=[当前图像, 前一帧, 初始状态]
    )
    
    # 解析VLM输出：提取行动指令
    next_action = parse_vlm_output(思维内容)
```

**第3层 - 环境执行循环**：
```python
执行行动(next_action)
  ↓
获取环境反馈(成功/失败)
  ↓
if 失败:
  生成反思思维 (为什么失败?)
  重新规划行动
else:
  保存: {观察+思维+行动} 到轨迹
  继续下一步
```

**多样化思维模式** (o1风格思维分类)：
- **分析型** (ANALYSIS): "这个容器可能在哪里?"
- **空间推理型** (SPATIAL): "从当前位置看，厨房在北方..."
- **反思型** (REFLECTION): "我遗漏了什么吗？让我再检查一遍..."
- **规划型** (PLANNING): "我需要先打开冰箱，然后搜索..."
- **验证型** (VERIFICATION): "这个确实是我要找的物体吗？"

#### 1.3 **RocAgent.py** (数据生成版) - 虚拟智能体
**功能**：在 AI2THOR 环境中执行行动、观察、交互。

**关键能力**：
```python
类 RocAgent:
  - navigate(item): 移动到物体前方
  - observe_once(方向): 观察周围环境
  - interact(item, type): 打开/关闭/操纵物体
  - compute_position_8(item): 计算8个最佳位置
  - adjust_view(item): 调整视角看清物体
```

**核心导航逻辑** (Teleport + 微调)：
```
1. 获取与物体交互的可达位置集合
2. 选择最佳位置 (优先级: 正面 > 侧面)
3. 使用 teleport 直接跳到该位置
4. 调整视角和身体高度
5. 执行交互 (pickup, open等)
```

#### 1.4 **baseAction.py** - 行动原语库
**功能**：定义所有可执行的基础行动。

**行动分类**：
```python
移动行动:
  - move_ahead, move_back, move_left, move_right
  - rotate_left, rotate_right
  - look_up, look_down
  - teleport(位置, 旋转)

操纵行动:
  - pick_up(物体)
  - release() / drop_out(物体)
  - put_in(容器)
  - throw_out(物体)

交互行动:
  - open/close(容器)
  - toggle_on/toggle_off(开关)
  - cook/slice_(烹饪类)
  - fill/empty(容器)
```

---

### 2️⃣ **evaluate/** - 评估框架

**目的**：在真实数据集上评估模型性能，模拟真实推理循环。

#### 架构流程：
```
evaluate.py (主评估脚本)
    ↓
[加载测试集 809个任务]
    ↓
RocAgent (评估版，在evaluate/ai2thor_engine/)
    ↓
[VLM推理循环]
    └─→ state_machine (观察→规划→思考→决策→验证→结束)
    └─→ 每步调用模型
    └─→  保存轨迹和结果
    ↓
[性能评估]
├─ 成功率 (Success Rate)
├─ 搜索效率 (Search Efficiency)
└─ 任务完成度 (Task Completeness)
```

#### 2.1 **evaluate.py** - 主评估脚本
**关键函数**：

**load_data()**：
```python
- 加载 809 个测试样本
- 支持分布式评估 (--total_count, --cur_count)
- 缓存已评估任务，避免重复
```

**get_trajectory()**：
```python
对每个任务:
  1. 初始化 RocAgent(场景、视距、网格大小、视野)
  2. 设置 max_steps (根据task_type自动计算)
  3. 循环执行:
     step = 1 to max_steps:
       当前状态 = agent.observe()
       思维 = model.reason(状态, 任务目标)
       行动 = model.plan(思维)
       agent.execute(行动)
       保存: {图像, 思维, 行动}
  4. 评估: 是否完成任务?
```

#### 2.2 **RocAgent.py** (评估版) - 状态机智能体

**状态机设计** (核心推理循环)：
```
STATE_OBSERVATION (观察)
    ↓ [捕获图像]
STATE_PLANNING (规划)
    ↓ [分析当前场景]
STATE_THINKING (思维)
    ↓ [调用VLM生成思维]
STATE_REFLECTION (反思)
    ↓ [检查进度，必要时重新规划]
STATE_DECISION_MAKING (决策)
    ↓ [根据思维生成行动]
STATE_VERIFICATION (验证)
    ↓ [验证行动是否成功]
STATE_END (结束)
    ↓ [评估任务完成情况]
```

**关键数据结构**：
```python
self.action_space = {
    "navigate to": 移动到位置,
    "pickup": 拿起物体,
    "put": 放入容器,
    "toggle": 切换开关,
    "open": 打开,
    "close": 关闭,
    "observe": 观察,
    "move_forward": 向前移动,
    "end": 结束
}

self.target_item_type2obj_id = {
    "Apple": ["Apple|位置1", "Apple|位置2"],
    "Plate": ["Plate|位置1"],
    ...
}
```

#### 2.3 **prompt.py** - VLM提示词库
**目的**：为VLM构建高质量的上下文提示。

**提示词类型**：
```python
- system_prompt: 定义智能体角色和行为规范
- observation_prompt: 描述当前观察
- planning_prompt: 请求规划下一步
- reflection_prompt: 请求自我反思
- verification_prompt: 请求验证完成情况
```

**示例**：
```
System: "你是一个在虚拟家庭中执行任务的智能体。你有以下能力：
  - 移动和导航
  - 观察周围环境
  - 拿起和放置物体
  - 打开和关闭容器
你需要通过深度思考和推理来完成任务。"

User: "当前场景：厨房。任务：找到苹果。
      我已经检查了柜台和冰箱，但没有找到。
      我还应该检查哪里？请进行推理。"
```

#### 2.4 **web_ui/** - 实时监控仪表板
**功能**：Web界面实时显示评估进度和交互过程。

```python
server.py:
  - Flask服务器
  - 路由: /task_progress, /interaction_log, /results
  
monitor.py:
  - 监控每个任务的执行状态
  - 记录思维和行动
  - 实时更新UI
```

---

### 3️⃣ **inference/** - 推理服务

**目的**：部署模型并提供 HTTP API 服务。

#### 架构：
```
local_deploy.py (服务启动)
    ↓
├─ HfServer (HuggingFace Transformers)
├─ VllmServer (vLLM - 高性能推理)
└─ EmbeddingServer (嵌入向量服务)
    ↓
Flask HTTP API
    ├─ POST /generate (文本生成)
    ├─ POST /chat (对话推理)
    └─ POST /match (对象匹配)
```

#### 关键代码逻辑：
```python
# 启动推理服务
http_server(args):
  model_server = VllmServer(model_type, model_name)
  
  @app.route("/chat", methods=["POST"])
  def chat():
    inputs = request.json['inputs']  # 对话历史
    generation_params = request.json['generation_parms']
    
    output, output_len = model_server.chat(inputs, generation_params)
    return {"output_text": output, "output_len": output_len}
```

#### 支持的推理框架：
- **HF Transformers**: 标准HuggingFace推理
- **vLLM**: 高吞吐量、低延迟 (使用PagedAttention优化)
- **Embedding**: 使用 Sentence-Transformers 计算向量相似度

---

### 4️⃣ **finetune/** - 微调管道

**目的**：基于合成数据对模型进行三阶段迭代微调。

#### 三阶段训练管道：
```
Stage 1: 模仿学习 (Imitation Learning)
  输入: 合成轨迹中的 {观察, 思维, 行动}
  目标: 学习从观察预测思维和行动
  损失: 匹配VLM生成的思维和行动
  
  ↓
  
Stage 2: 自探索微调 (Self-Exploration Tuning)
  输入: 模型生成的轨迹
  目标: 改进模型在真实环境中的探索能力
  损失: 任务完成度反馈
  
  ↓
  
Stage 3: 自改进微调 (Self-Correction Tuning)
  输入: 失败案例和反思
  目标: 学习从错误中改进
  特殊token: <|feedback|> 标记反思部分
  损失: 仅在反思token上计算梯度
```

#### 配置文件：
```yaml
# template.yaml
model_name_or_path: Qwen/Qwen2.5-VL-3B-Instruct
output_dir: ./output
per_device_train_batch_size: 4
learning_rate: 2e-4
num_train_epochs: 3
# 使用LoRA低秩适配
lora_target: ["q_proj", "v_proj"]
lora_rank: 8
lora_alpha: 16
```

#### 关键特性：
- **LoRA微调**：参数高效，使用低秩矩阵
- **选择性训练**：Stage 3 仅训练反思token
- **DeepSpeed集成**：支持ZeRO优化
- **多卡分布式**：Accelerate + DeepSpeed

---

### 5️⃣ **scripts/** - 执行脚本

#### train.sh - 训练脚本
```bash
# 使用LLaMA-Factory进行训练
python LLaMA-Factory/src/train.py \
  --config_file finetune/full/template.yaml
```

#### eval.sh - 评估脚本
```bash
# 在809个测试任务上评估
python evaluate/evaluate.py \
  --model_name Qwen/Qwen2.5-VL-3B-Instruct \
  --input_path data/test_809.json \
  --cur_count 1 --total_count 1
```

#### run_specific_tasks.sh - 特定任务评估
```bash
python evaluate/evaluate.py \
  --task_ids 0,5,10,20
```

---

## 🔄 完整数据流程

### 训练数据流程：
```
1. TaskGenerate.py
   ↓ [生成9.3k个任务模板]
   
2. o1StyleGenerate.py + VLM API
   ↓ [调用GPT-4o生成深度思维]
   ↓ [捕获环境状态的64K张图像]
   ↓ [生成8M个思维token]
   
3. 数据集格式化
   {
     "instruction": "找到苹果",
     "images": [...],        # 图像序列
     "thought": "首先观察...", # VLM生成的思维
     "action": "move to",     # 行动
     "text": "<image>观察...分析...决策..."  # 多模态序列
   }
   
4. LLaMA-Factory 微调
   ↓ [三阶段训练]
   ↓ [输出微调后的模型]
   
5. 模型输出
   Qwen2.5-VL-3B/7B-Instruct (embodied版本)
```

### 评估流程：
```
1. 加载测试集 (809个任务)
   
2. 对每个任务:
   a) 初始化环境 (AI2THOR场景)
   b) 循环执行 (max_step=10-20步):
      - 观察当前状态
      - 调用微调模型生成思维
      - 解析行动指令
      - 执行行动
      - 保存轨迹
   
3. 评估指标:
   ✅ 成功率: 任务是否完成?
   ✅ 搜索效率: 平均每个任务的步数
   ✅ 完成度: 预测行动中关键行动的比例
   
4. 输出结果
   - 每个任务的详细轨迹
   - 整体性能统计
   - 可视化仪表板
```

---

## 🎨 关键创新点

### 1. **Observation-Thought-Action 三元组**
```
不同于传统的 Observation-Action 二元组，
Embodied-Reasoner 引入了中间的 "Thought" 层，
使模型能够进行显式的中间推理。

[观察图像] 
    ↓
[生成思维] ← VLM (GPT-4o)
    ├─ 分析当前状态
    ├─ 推理物体位置
    ├─ 规划下一步
    └─ 反思进度
    ↓
[执行行动]
```

### 2. **长轨迹 (8M tokens) 中的多样化思维**
```
传统数据: [观察] → [行动] (每步1-2 token)
Embodied: [观察] → [思维+分析+规划+反思] (每步 50-200 token)

优势:
- 模型学习显式的推理过程
- 支持长期规划能力
- 提高复杂任务的成功率
```

### 3. **10种子任务类型的多样化**
```
不是简单的单一任务，而是系统性地设计了多种难度和类型:
- 单搜索 (single_search)
- 双搜索 (two_search)
- 拿取放置 (pickup_and_put)
- 有序转移 (ordered_pickup_and_put_two_object)
- 容器搜索 (search_in_container)
- ... 等共10种

每种任务对应不同的推理模式。
```

### 4. **AI2THOR 环境的充分利用**
```
- 120个高质量室内场景
- 2100个可交互物体
- 2600个容器
- 真实的物理模拟
- 第一人称视角 (模拟真实环境)
```

---

## 💡 工程难点和解决方案

| 难点 | 解决方案 |
|------|----------|
| **VLM调用成本高** | 批量生成，缓存中间结果 |
| **思维生成易失败** | 3次重试机制，失败反思 |
| **行动执行碰撞** | Teleport + 8方向微调 |
| **长轨迹数据不稳定** | 三阶段迭代训练 + 自改进 |
| **评估时间长** | 分布式评估 (多进程) |
| **模型计算量大** | vLLM 推理优化 + LoRA微调 |

---

## 📊 性能指标解释

### 成功率 (Success Rate)
```
= 完成任务的样本数 / 总样本数
范围: 0-100%
```

### 搜索效率 (Search Efficiency)
```
= (max_steps - 实际步数) / max_steps * 100%
高效率 = 用更少步数完成任务
```

### 任务完成度 (Task Completeness)
```
= 关键行动命中数 / 总预测行动数 * 100%
例: 如果任务要求 [navigate, open, pickup, end]
    模型输出 [navigate, rotate, open, pickup, observe, end]
    完成度 = 4/6 ≈ 67%
```

---

## 🚀 如何运行？

### 快速开始 (评估)
```bash
# 1. 安装依赖
conda create -n embodied-reasoner python=3.9
conda activate embodied-reasoner
pip install -r requirements.txt

# 2. 启动推理服务
python inference/local_deploy.py \
  --frame vllm \
  --model_name Qwen/Qwen2.5-VL-3B-Instruct \
  --port 10000

# 3. 运行评估
bash scripts/eval.sh
```

### 数据生成 (生成训练数据)
```bash
# 1. 生成任务
python data_engine/TaskGenerate.py

# 2. 生成思维轨迹
python data_engine/o1StyleGenerate.py
python data_engine/o1StyleGenerate_ordered.py
```

### 微调模型
```bash
# 需要 LLaMA-Factory
git clone -b embodied-reasoner https://github.com/iGangao/LLaMA-Factory.git
cd LLaMA-Factory
bash ../scripts/train.sh
```

---

## 总结

**Embodied-Reasoner** 的工程设计采用了 **模块化、分层、迭代** 的架构：

1. **数据层** (data_engine): 自动合成高质量的多模态轨迹
2. **模型层** (finetune): 三阶段迭代微调
3. **推理层** (inference): 高性能服务化部署
4. **评估层** (evaluate): 完整的性能评估体系

核心创新在于：
- ✨ 长轨迹中的显式思维推理
- ✨ 多样化的思维模式
- ✨ 自动化的数据生成管道
- ✨ 完整的评估框架

这使得 Embodied-Reasoner 能够在具身交互任务中表现出类似 o1 的深度推理能力。
