# 📚 文档导航地图

为了让你完全理解 Embodied-Reasoner 的数据生成管道，我为你创建了 **5份递进式文档**。

## 📖 文档清单

### 1. **QUICK_REFERENCE.md** ⚡ (5分钟快速上手)
**适合人群**: 想立即开始的人
**内容**:
- 三层架构一张图
- 快速配置5分钟指南
- VLM模型选择表
- 成本计算器
- 常见问题速查

📍 **何时读**: 第一时间！

---

### 2. **DATA_GENERATION_DETAILED.md** 🔧 (详细技术解析)
**适合人群**: 想深入理解的工程师
**内容**:
- taskgenerate/ 目录完整解析
- TaskGenerate.py 十种任务类型详解
- o1StyleGenerate.py 完整流程图
- VLM API 配置位置和修改方法
- 三个关键对比表

**关键回答你的问题**:
```
Q: taskgenerate包含场景和动作定义吗?
A: ❌ 只有场景定义（对象、位置、状态）
   📁 taskgenerate/ = 数据库
   📝 pick_up_and_put.json = 兼容性规则

Q: TaskGenerate.py 需要VLM吗?
A: ❌ 不需要！纯逻辑规则引擎
   ✓ 输入：场景元数据
   ✓ 输出：9.3K 任务模板
   ✓ 时间：~1小时
   ✓ 成本：$0

Q: o1StyleGenerate.py 才涉及VLM吗?
A: ✅ 是的！完全依赖VLM
   ✓ 每条轨迹 15-25 次 VLM 调用
   ✓ 成本：$0.01-0.40 per 轨迹
   ✓ 时间：2-5分钟 per 轨迹
```

📍 **何时读**: 配置后，运行前

---

### 3. **DATA_PIPELINE_VISUAL.md** 📊 (可视化对比分析)
**适合人群**: 喜欢看图的人
**内容**:
- 三层架构盒图展示
- 执行时间估算
- VLM调用频率分析
- 完整数据流向图
- 文件修改检查表

**特色**: 大量 ASCII 流程图和表格

📍 **何时读**: 想看全图时

---

### 4. **VLM_CONFIG_GUIDE.md** 🔑 (API配置完全手册)
**适合人群**: 需要配置VLM的人
**内容**:
- 三句话总结区分
- 获取API Key的4种方法
- VLMCallapi_keys.py 配置详解
- API 端点修改指南
- 测试配置的Python脚本
- 成本估算详细表
- 错误排查和调试技巧

**核心配置代码**:
```python
# data_engine/VLMCallapi_keys.py
api_keys = [
    "sk-proj-your-key-here"
]
```

📍 **何时读**: 准备配置VLM时

---

### 5. **PROJECT_ARCHITECTURE.md** 🏗️ (完整项目架构)
**适合人群**: 想了解全貌的人
**内容**:
- 5大模块详细解读
- 完整数据流程
- 性能指标解释
- 工程难点和解决方案
- 如何运行的完整指南

📍 **何时读**: 理解整个项目时

---

## 🎯 快速导航路径

### 场景A: 我想立即开始
```
1. 读 QUICK_REFERENCE.md (5分钟)
   ↓
2. 读 VLM_CONFIG_GUIDE.md 中的"配置步骤" (10分钟)
   ↓
3. 运行 data_engine/TaskGenerate.py (1小时)
   ↓
4. 运行 data_engine/o1StyleGenerate.py (几小时/几天)
```

### 场景B: 我想完全理解原理
```
1. 读 QUICK_REFERENCE.md (5分钟) - 全景
   ↓
2. 读 DATA_GENERATION_DETAILED.md (30分钟) - 细节
   ↓
3. 读 DATA_PIPELINE_VISUAL.md (20分钟) - 可视化
   ↓
4. 读 PROJECT_ARCHITECTURE.md (30分钟) - 全局架构
```

### 场景C: 我想调整和优化
```
1. 读 VLM_CONFIG_GUIDE.md 中的"成本估算" (10分钟)
   ↓
2. 修改 data_engine/o1StyleGenerate.py 中的参数
   ↓
3. 读 DATA_GENERATION_DETAILED.md 中的"10种任务类型" (20分钟)
   ↓
4. 运行新配置的脚本测试
```

---

## 📋 核心知识点速记

### 三层架构对比

| 层级 | 组件 | VLM需求 | 成本 | 时间 | 输出 |
|------|------|--------|------|------|------|
| 1️⃣ | taskgenerate/ | ❌ | $0 | - | 场景元数据 |
| 2️⃣ | TaskGenerate.py | ❌ | $0 | ~1h | 9.3K任务 |
| 3️⃣ | o1StyleGenerate.py | ✅ | $61-3,348 | ~500h | 64K图像+轨迹 |

### VLM 配置三步

```
步骤1: 获取Key
      → https://platform.openai.com/api-keys

步骤2: 添加Key
      → data_engine/VLMCallapi_keys.py

步骤3: 运行脚本
      → python o1StyleGenerate.py
```

### 十种任务类型快速判断

**简单任务** (⭐-⭐⭐):
- single_search: 搜索
- single_pickup: 拿起
- single_toggle: 开关

**中等任务** (⭐⭐-⭐⭐⭐):
- single_search_from_closerep: 打开容器搜索
- pickup_and_put: 转移物体

**复杂任务** (⭐⭐⭐-⭐⭐⭐⭐⭐):
- pickup_from_closerep_and_put_in_closerep: 复杂转移
- ordered_pickup_two_object_and_put: 有序双物体

---

## ❓ 你的三个原始问题 - 最终答案

### Q1: taskgenerate包含场景和动作的定义吗？

**完整答案**:
- ✅ **包含场景定义**: metadata.json 有所有对象的完整信息
- ❌ **不包含动作定义**: 动作在 TaskGenerate.py 中通过规则生成
- 📝 **包含兼容性规则**: pick_up_and_put.json 定义了"什么物体可以放在什么容器里"

### Q2: TaskGenerate.py 只需要模拟器和JSON数据，不需要VLM模型吗？

**完整答案**:
- ✅ **完全正确**！TaskGenerate.py **不需要VLM**
- ⚙️ 它只做：遍历对象 → 应用规则 → 生成任务JSON
- 🔄 纯逻辑处理，无需AI推理
- 💾 输入：taskgenerate/ 中的 metadata.json
- 📤 输出：{task_type}_task_metadata/ 中的任务JSON

### Q3: o1StyleGenerate.py 才涉及VLM？OpenAI配置在哪里？

**完整答案**:
- ✅ **完全正确**！o1StyleGenerate.py **必须有VLM**
- 🔑 **API Key配置**: `data_engine/VLMCallapi_keys.py`
  ```python
  api_keys = ["sk-proj-your-key-here"]
  ```
- 🌐 **VLM调用代码**: `data_engine/vlmCall.py`
- 🔧 **修改API端点**: vlmCall.py 第114行
  ```python
  # 当前
  conn = http.client.HTTPSConnection("us.ifopen.ai")
  # 改为官方
  conn = http.client.HTTPSConnection("api.openai.com")
  ```

---

## 🚀 现在就开始

### 一键快速开始

```bash
# 1. 打开快速参考卡
# → QUICK_REFERENCE.md

# 2. 获取API Key并配置
# → VLM_CONFIG_GUIDE.md

# 3. 运行生成流程
cd data_engine

# 3a. 生成任务模板 (无需VLM)
python TaskGenerate.py

# 3b. 生成轨迹数据 (需VLM)
python o1StyleGenerate.py
```

### 推荐阅读顺序

1. **QUICK_REFERENCE.md** (今天)
2. **DATA_GENERATION_DETAILED.md** (明天)
3. **VLM_CONFIG_GUIDE.md** (配置前)
4. **DATA_PIPELINE_VISUAL.md** (想看图时)
5. **PROJECT_ARCHITECTURE.md** (深入学习)

---

## 📞 有问题的下一步

| 问题类型 | 参考文档 | 位置 |
|---------|--------|------|
| 什么是三层架构? | QUICK_REFERENCE.md | 第1节 |
| 如何配置VLM? | VLM_CONFIG_GUIDE.md | "配置步骤" |
| TaskGenerate.py如何工作? | DATA_GENERATION_DETAILED.md | "Layer 2" |
| 怎样修改API端点? | VLM_CONFIG_GUIDE.md | "修改API端点" |
| 成本是多少? | VLM_CONFIG_GUIDE.md | "成本估算" |
| 怎样用本地模型? | VLM_CONFIG_GUIDE.md | "生产环境建议" |
| 完整架构是什么? | PROJECT_ARCHITECTURE.md | 整个文件 |

---

## ✨ 现在你知道了

✅ **taskgenerate/** = 数据库 (AI2THOR场景清单)
✅ **TaskGenerate.py** = 规则引擎 (无需VLM)
✅ **o1StyleGenerate.py** = 推理引擎 (需要VLM)
✅ **VLM配置** = VLMCallapi_keys.py (添加API Key)
✅ **API修改** = vlmCall.py (改端点)

Happy coding! 🎉
