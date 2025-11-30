# 数据生成管道详细解析

## 问题核心理解

你的三个关键问题：
1. **taskgenerate** 包含什么？场景定义还是动作定义？
2. **TaskGenerate.py** 做什么？需要VLM吗？
3. **VLM配置**在哪里？如何修改？

答案：**都不一样！** 让我一层层剥开。

---

## 📍 Layer 1: `taskgenerate/` - 场景元数据库

### 文件结构
```
taskgenerate/
├── kitchens/
│   ├── FloorPlan1/
│   │   ├── metadata.json        ← AI2THOR场景的完整元数据
│   │   └── originPos.json       ← 代理初始位置
│   ├── FloorPlan2/
│   ...
├── living_rooms/
├── bedrooms/
├── bathrooms/
└── pick_up_and_put.json         ← 物体兼容性映射表
```

### metadata.json 内容 - **只包含场景静态信息**

```json
{
  "agent": {
    "position": {"x": 1.5, "y": 0.901, "z": 0.5},
    "rotation": {"x": 0, "y": 0, "z": 0},
    "cameraHorizon": 0
  },
  "objects": [
    {
      "objectId": "CounterTop|00.08|01.15|00.00",
      "objectType": "CounterTop",
      "parentReceptacles": [],           ← 父容器(为空=在地板上)
      "pickupable": false,
      "receptacle": true,               ← 是否是容器
      "openable": false,
      "isOpen": false,
      "toggleable": false,
      "isToggled": false,
      "visible": true,
      "axisAlignedBoundingBox": {...}
    },
    {
      "objectId": "Apple|00.47|01.15|00.48",
      "objectType": "Apple",
      "parentReceptacles": ["CounterTop|00.08|01.15|00.00"],  ← 在计数器上
      "pickupable": true,               ← 可拿起
      "receptacle": false,
      "openable": false,
      "isOpen": false,
      "toggleable": false,
      "isToggled": false,
      "visible": true
    },
    {
      "objectId": "Fridge|00.20|00.00|01.50",
      "objectType": "Fridge",
      "parentReceptacles": [],
      "pickupable": false,
      "receptacle": true,
      "openable": true,                 ← 可打开
      "isOpen": false,                  ← 初始状态关闭
      "toggleable": false
    },
    {
      "objectId": "Egg|00.00|01.00|00.30",
      "objectType": "Egg",
      "parentReceptacles": ["Fridge|00.20|00.00|01.50"],  ← 在冰箱内
      "pickupable": true,
      "receptacle": false,
      "openable": false
    }
  ]
}
```

### pick_up_and_put.json 内容 - **物体兼容性规则**

```json
[
  {
    "Apple": ["Pot", "Pan", "Bowl", "Microwave", "Fridge", "Plate", "SinkBasin", "CounterTop", "GarbageCan"]
  },
  {
    "Egg": ["Pot", "Pan", "Bowl", "Microwave", "Fridge", "Plate", "SinkBasin", "CounterTop", "GarbageCan"]
  },
  {
    "Bread": ["Pan", "Microwave", "Fridge", "Plate", "CounterTop", "GarbageCan"]
  }
]
```

**作用**：定义"苹果可以放在什么容器里"的规则。

---

## 🔧 Layer 2: `TaskGenerate.py` - 任务模板生成器

### 本质
**完全不需要VLM！** 这是纯粹的：
- 输入：场景元数据 (JSON)
- 输出：任务元数据 (JSON) + 关键行动序列
- 处理：逻辑规则 + 随机采样

### 核心逻辑

#### 第1步：加载场景元数据
```python
# 实际代码流程
metadata = load_json('taskgenerate/kitchens/FloorPlan1/metadata.json')
scene_objects = metadata['objects']  # 这个场景有哪些物体
```

#### 第2步：根据 `task_type` 执行对应方法

```python
class TaskGenerate:
    def single_search(self, num=1):
        """
        任务类型：单目标搜索
        规则：找一个可拿起的物体
        """
        generate_task = []
        
        # 遍历所有物体
        for obj in scene_objects:
            # 过滤条件1：物体可拿起
            if not self.is_pickupable(obj):
                continue
            
            # 过滤条件2：物体不在地板上（在某个容器内）
            if self.is_parent_floor_or_null(obj):
                continue
            
            # 过滤条件3：物体的直接容器不需要打开（以求简单）
            if self.is_parent_receptacle_openable(obj):
                continue
            
            # 过滤条件4：物体的容器的容器是地板（二级深度限制）
            if not self.is_grandparent_floor_or_null(obj):
                continue
            
            # ✅ 条件都满足 → 生成一个任务
            obj_type = obj['objectType']
            obj_id = obj['objectId']
            obj_parent_id = obj['parentReceptacles'][-1]
            obj_parent_type = obj_parent_id.split('|')[0]
            
            # 随机选择表达方式
            expressions = [
                f"Find the {obj_type} in the room.",
                f"Locate the {obj_type} in the room.",
                ...
            ]
            task_name = random.choice(expressions)
            
            # 构造任务 JSON
            task = {
                "taskname": task_name,                    # "Find the Apple in the room."
                "tasktype": "single_search",
                "metadatapath": "taskgenerate/kitchens/FloorPlan1/metadata.json",
                "actions": [
                    {
                        "action": "navigate to",
                        "objectId": "CounterTop|00.08|01.15|00.00",
                        "objectType": "CounterTop",
                        "reward": 1,
                        "relatedObject": ["CounterTop|00.08|01.15|00.00", "Apple|00.47|01.15|00.48"]
                    },
                    {
                        "action": "end",
                        "reward": 1,
                        "relatedObject": [...]
                    }
                ],
                "totalreward": 2
            }
            generate_task.append(task)
```

### 10种任务类型详解

| 任务类型 | 规则逻辑 | 关键行动序列 | 难度 |
|----------|---------|-------------|------|
| **single_search** | 找可拿起的物体，在容器上 | navigate → end | ⭐ |
| **single_search_from_closerep** | 找可拿起的物体，在**可打开的容器内** | navigate → open → end | ⭐⭐ |
| **single_pickup** | 拿起一个物体 | navigate → pickup → end | ⭐ |
| **single_pickup_from_closerep** | 拿起容器内的物体 | navigate → open → pickup → close → end | ⭐⭐⭐ |
| **single_toggle** | 切换开关（灯等） | navigate → toggle → end | ⭐ |
| **pickup_and_put** | 拿起物体放到另一个容器 | navigate → pickup → navigate → put → end | ⭐⭐ |
| **pickup_from_closerep_and_put** | 从容器拿出→放到另一容器 | navigate → open → pickup → close → navigate → put → end | ⭐⭐⭐ |
| **pickup_and_put_in_closerep** | 拿起→放入可打开容器 | navigate → pickup → navigate → open → put → end | ⭐⭐⭐ |
| **pickup_from_closerep_and_put_in_closerep** | 复杂操作 | navigate → open → pickup → close → navigate → open → put → end | ⭐⭐⭐⭐ |
| **ordered_pickup_two_object_and_put** | 有序的两对象转移 | 最复杂的组合 | ⭐⭐⭐⭐⭐ |

### 输出文件位置
```
{task_type}_task_metadata/
├── FloorPlan1.json    ← 该场景的所有{task_type}任务
├── FloorPlan2.json
└── ...
```

**关键点**：`TaskGenerate.py` **完全不需要VLM**！它只是逻辑过滤 + JSON生成。

---

## 🎬 Layer 3: `o1StyleGenerate.py` - 思维轨迹生成器

### 本质
**必须需要VLM！** 这是：
- 输入：任务元数据 (来自TaskGenerate) + 虚拟环境
- 输出：Observation-Thought-Action 轨迹 + 图像
- 处理：模型推理 + 环境执行

### 核心流程

```
1️⃣ 加载任务
   task = load_json("single_search_task_metadata/FloorPlan1.json")
   task["taskname"] = "Find the Apple in the room."
   task["actions"] = [{"action":"navigate to", "objectId":"CounterTop|..."}, {"action":"end"}]

2️⃣ 初始化环境
   controller = Controller(scene="FloorPlan1", ...)
   rocAgent = RocAgent(controller)  # 虚拟智能体

3️⃣ 循环执行（关键！）
   for step in range(max_steps):
       
       a) 观察
          image = capture_screenshot()
          observation_text = f"I'm in a kitchen. I can see: {visible_objects}"
       
       b) [FIRST TIME ONLY] 自我观察
          selfobs = VLM.generate(
              prompt="Describe the objects in front of you",
              image=image
          )
          # 返回: "<Observation> I see a kitchen with a counter..."
          trajectory.append(selfobs)
       
       c) 思维生成 [需要VLM]
          thinking = VLM.generate(
              system="You are a reasoning agent...",
              prompt=f"Task: {task['taskname']}. Current observation: {selfobs}. Next step?",
              images=[current_image, last_frame, initial_image]
          )
          # 返回: "<Thought> The Apple is likely on the CounterTop...</Thought>"
          trajectory.append(thinking)
       
       d) 行动决策 [需要VLM]
          decision = VLM.generate(
              prompt="Based on your thought, what action should you take?",
              images=[...]
          )
          # 返回: "navigate to CounterTop"
          action = parse_action(decision)
       
       e) 执行行动
          rocAgent.execute(action)
          feedback = check_success()
       
       f) 验证/反思 [需要VLM]
          if not feedback['success']:
              reflection = VLM.generate(
                  prompt="Why did the action fail? What to do next?"
              )
              # 返回: "<Reflection> The navigation failed..."
              trajectory.append(reflection)
          
       g) 保存
          trajectory.append({
              "observation": image,
              "action": action,
              "reward": feedback['reward']
          })

4️⃣ 输出轨迹
   {
       "scene": "FloorPlan1",
       "tasktype": "single_search",
       "taskname": "Find the Apple in the room.",
       "trajectory": [
           "<Observation> I see a kitchen...",
           "<Thought> The apple is likely on...",
           "<Decision> I should navigate...",
           "..."
       ],
       "images": [
           "path/to/image_0.png",
           "path/to/image_1.png",
           ...
       ]
   }
```

---

## 🔑 VLM 配置在哪里？

### 1️⃣ API Key 配置

**文件位置**：
```
z:\Code_Windows\embodied_reasoner\data_engine\VLMCallapi_keys.py
```

**当前内容** (为空)：
```python
api_keys=[  
    # please add your api keys here
]
```

### 2️⃣ 如何修改？

#### 方案A：直接添加key
```python
# VLMCallapi_keys.py
api_keys = [
    "sk-proj-xxxxxxxxxxxxx",  # ChatGPT API key
    "sk-proj-yyyyyyyyyyyyy",  # 备用key
]
```

#### 方案B：使用环境变量（推荐）
```python
# VLMCallapi_keys.py
import os

api_keys = [
    os.getenv("OPENAI_API_KEY"),
    os.getenv("OPENAI_API_KEY_BACKUP"),
]
```

然后在终端设置：
```bash
export OPENAI_API_KEY="sk-proj-xxxxx"
```

### 3️⃣ VLM API 调用代码

**文件位置**：
```
z:\Code_Windows\embodied_reasoner\data_engine\vlmCall.py
```

**核心代码**：
```python
class VLMAPI:
    def __init__(self, model):
        self.model = model  # "gpt-4o-2024-11-20"
    
    def vlm_request(self, system_text, user_text, image_path1=None, max_tokens=1500):
        """
        调用VLM模型生成思维
        """
        # 1. 编码图像为base64
        if image_path1:
            base64_image = self.encode_image(image_path1)
        
        # 2. 构造消息体
        messages = [
            {
                "role": "system",
                "content": system_text
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low"
                        }
                    }
                ]
            }
        ]
        
        # 3. 构造请求
        payload = json.dumps({
            "model": self.model,
            "stream": False,
            "messages": messages,
            "temperature": 0.9,
            "max_tokens": max_tokens
        })
        
        # 4. 发送请求到OpenAI兼容的API
        conn = http.client.HTTPSConnection("us.ifopen.ai")  # ← 注意：这不是官方OpenAI！
        headers = {
            'Accept': 'application/json',
            'Authorization': 'Bearer ' + api_key,
            'Content-Type': 'application/json'
        }
        
        conn.request("POST", "/v1/chat/completions", payload, headers)
        res = conn.getresponse()
        data = json.loads(res.read().decode("utf-8"))
        
        # 5. 提取响应
        content = data["choices"][0]["message"]["content"]
        return content
```

### 4️⃣ 如何修改API端点？

**重要发现**：当前代码使用 **非官方的 OpenAI 兼容 API**！

```python
# 当前代码
conn = http.client.HTTPSConnection("us.ifopen.ai")  # 第三方服务

# 改为官方OpenAI
conn = http.client.HTTPSConnection("api.openai.com")

# 或改为本地部署
conn = http.client.HTTPSConnection("localhost:8000")
```

### 5️⃣ o1StyleGenerate.py 中的使用

```python
class O1StyleGenerate:
    def __init__(self, controller, scene, ..., model="gpt-4o-2024-11-20"):
        self.model = model  # ← 模型名称
        
    def generate_selfObs(self, image_path):
        """生成自我观察"""
        llmapi = VLMAPI(self.model)
        selfobservation = llmapi.vlm_request(
            systext="You are a mobile robot...",
            usertext="Describe visible objects...",
            image_path1=image_path
        )
        return selfobservation
```

**在主程序中更改模型**：
```python
# o1StyleGenerate.py 主程序
if __name__ == "__main__":
    model = "gpt-4o-2024-11-20"  # ← 改这里
    # 或改为
    model = "gpt-4-turbo"
    model = "gpt-4"
    model = "claude-3-vision"  # 如果支持的话
```

---

## 完整执行流程图

```
TaskGenerate.py (不需要VLM)
    ↓
    输入: AI2THOR场景元数据 + pick_up_and_put.json
    ↓
    └─→ single_search() 遍历对象 → 过滤条件 → 生成任务JSON
    └─→ single_pickup() 类似逻辑
    └─→ ordered_pickup_two_object_and_put() 复杂组合
    ↓
    输出: {task_type}_task_metadata/{scene}.json
         [
             {"taskname": "Find Apple", "actions": [...]},
             {"taskname": "Pick up Plate", "actions": [...]},
             ...
         ]
    ↓
    ↓
o1StyleGenerate.py (需要VLM)
    ↓
    输入: 任务元数据 JSON + AI2THOR 场景
    ↓
    for each task:
        ├─ 初始化虚拟环境
        ├─ for each step:
        │   ├─ 捕获图像
        │   ├─ 调用VLM生成思维 [需要API key]
        │   ├─ 执行行动
        │   ├─ 保存轨迹 + 图像
        │   └─ 检查成功/失败
        └─ 输出轨迹JSON
    ↓
    输出: {scene}/{task_type}/trajectory_0.json
         {
             "scene": "FloorPlan1",
             "trajectory": ["<Observation>...", "<Thought>...", ...],
             "images": ["image_0.png", "image_1.png", ...],
             "reward": 10
         }
```

---

## 关键对比表

| 组件 | 需要VLM? | 输入 | 输出 | 文件 |
|------|---------|------|------|------|
| **taskgenerate/** | ❌ | (无) | 场景元数据 + 物体兼容性 | metadata.json, pick_up_and_put.json |
| **TaskGenerate.py** | ❌ | 场景元数据 | 任务模板 + 关键行动 | *_task_metadata/*.json |
| **o1StyleGenerate.py** | ✅ VLM必需 | 任务元数据 + 虚拟环境 | 完整轨迹 + 思维 + 图像 | trajectory_*.json + images/ |
| **VLMCallapi_keys.py** | ✅ API key | (无) | OpenAI API keys | api_keys = [...] |
| **vlmCall.py** | ✅ HTTP客户端 | 提示词 + 图像 | VLM响应 | VLMAPI 类 |

---

## 设置完整指南

### Step 1: 获取API Key
```bash
# 从 https://platform.openai.com/api-keys 获取
# 或使用其他VLM供应商的key
```

### Step 2: 配置API Key
```python
# VLMCallapi_keys.py
api_keys = [
    "sk-proj-your-key-here"
]
```

### Step 3: (可选) 修改API端点
```python
# vlmCall.py 中搜索 "us.ifopen.ai"
# 改为你的API端点 (官方OpenAI 或本地部署)
```

### Step 4: 运行数据生成
```bash
# 1. 生成任务
cd data_engine
python TaskGenerate.py

# 2. 生成思维轨迹 (需要网络和API key)
python o1StyleGenerate.py
```

---

## 常见问题

### Q: TaskGenerate.py 为什么不需要VLM?
**A**: 它只做**逻辑过滤**，不需要"思考"。就像一个数据库查询：
- 条件：可拿起? ✓ 在容器里? ✓ → 生成任务

### Q: o1StyleGenerate.py 为什么需要VLM?
**A**: 因为它需要生成**真实推理过程**：
- "为什么苹果可能在某处?" → VLM思考
- "下一步应该做什么?" → VLM决策
- 这不能靠规则完成

### Q: taskgenerate/ 中的metadata.json 是怎么生成的?
**A**: 来自AI2THOR模拟器的 `controller.last_event.metadata`
```python
# utils.py
def get_scene_metadata(scene):
    controller = Controller(..., scene=scene)
    metadata = controller.last_event.metadata
    save_data_to_json(metadata, f"taskgenerate/{room}/FloorPlan/{scene}/metadata.json")
```

### Q: 如果没有API key 可以运行吗?
**A**: 
- ✅ TaskGenerate.py 可以
- ❌ o1StyleGenerate.py 不能
- 可以使用本地模型如 Llama 作为替代

### Q: 能改成使用国内模型（如Qwen）吗?
**A**: 可以！改 vlmCall.py:
```python
class VLMAPI:
    def vlm_request(self, ...):
        if self.model.startswith("qwen"):
            conn = http.client.HTTPSConnection("api.alibabacloud.com")  # 阿里云端点
            # 改API调用方式
```
