# VLM API 快速配置指南

## 🎯 三句话总结

1. **taskgenerate/** = AI2THOR 场景的 **静态数据库**（只有对象信息）
2. **TaskGenerate.py** = 纯 **逻辑规则** 生成任务模板（**无需VLM**）
3. **o1StyleGenerate.py** = 调用VLM **生成思维轨迹**（**需要VLM**）

---

## 配置步骤

### 1. 获取 API Key

#### 选项A：使用 OpenAI（ChatGPT）
1. 访问 https://platform.openai.com/api-keys
2. 创建新 API Key
3. 复制 key（格式: `sk-proj-xxxxx`）

#### 选项B：使用其他服务
- **Claude**: https://console.anthropic.com/
- **Qwen**: https://dashscope.console.aliyun.com/
- **本地模型**: LLaMA、Mistral 等

---

### 2. 添加 API Key

**文件**: `data_engine/VLMCallapi_keys.py`

```python
# 方案1：直接添加（不安全，谨慎提交）
api_keys = [
    "sk-proj-your-openai-key-here",
]

# 方案2：从环境变量（推荐）
import os
api_keys = [
    os.getenv("OPENAI_API_KEY"),
]

# 方案3：读取配置文件（最安全）
import json
with open("api_config.json", "r") as f:
    config = json.load(f)
api_keys = config["keys"]
```

**设置环境变量**（Linux/Mac）：
```bash
export OPENAI_API_KEY="sk-proj-xxxxx"
```

**设置环境变量**（Windows PowerShell）：
```powershell
$env:OPENAI_API_KEY="sk-proj-xxxxx"
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "sk-proj-xxxxx", "User")
```

---

### 3. 修改 VLM 模型

**文件**: `data_engine/o1StyleGenerate.py`

**搜索主程序部分**：
```python
if __name__=="__main__":
    model = "gpt-4o-2024-11-20"  # ← 改这行
```

**支持的模型**：
```python
model = "gpt-4-turbo"            # GPT-4 Turbo
model = "gpt-4o"                 # GPT-4O
model = "gpt-4o-mini"            # GPT-4O Mini（便宜）
model = "claude-3-opus"          # Claude（需改API）
model = "qwen-vl-max"            # 通义千问（需改API）
```

---

### 4. 修改 API 端点（可选）

**文件**: `data_engine/vlmCall.py`

```python
# 当前（第三方兼容）
conn = http.client.HTTPSConnection("us.ifopen.ai")

# 改为官方 OpenAI
conn = http.client.HTTPSConnection("api.openai.com")

# 改为本地部署（如果有的话）
conn = http.client.HTTPSConnection("localhost:8000")
```

---

## 测试配置

### 测试 API Key

创建文件 `test_vlm.py`：

```python
from data_engine.vlmCall import VLMAPI
from PIL import Image
import requests

# 1. 测试API连接
llmapi = VLMAPI("gpt-4o-mini")  # 用mini版便宜

# 2. 测试文字请求
try:
    response = llmapi.vlm_request(
        systext="You are a helpful assistant.",
        usertext="Say hello!"
    )
    print(f"✅ API连接成功！响应: {response}")
except Exception as e:
    print(f"❌ API错误: {e}")

# 3. 测试图像请求
try:
    # 下载测试图像
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    img_path = "test_cat.jpg"
    img = Image.open(requests.get(url, stream=True).raw)
    img.save(img_path)
    
    response = llmapi.vlm_request(
        systext="Describe this image briefly.",
        usertext="What do you see?",
        image_path1=img_path
    )
    print(f"✅ 图像处理成功！响应: {response[:100]}...")
except Exception as e:
    print(f"❌ 图像处理错误: {e}")
```

**运行**：
```bash
cd embodied_reasoner
python test_vlm.py
```

---

## 常见问题排查

### ❌ Error: "No module named 'vlmCall'"

**解决**：
```bash
cd data_engine
python -c "from vlmCall import VLMAPI; print('OK')"
```

### ❌ Error: "HTTP 401 Unauthorized"

**原因**：API Key 无效或过期

**解决**：
1. 检查 API Key 格式（应该以 `sk-` 开头）
2. 访问 https://platform.openai.com/api-keys 重新生成
3. 确保账户有足够余额

### ❌ Error: "HTTP 429 Too Many Requests"

**原因**：请求频率过高或额度限制

**解决**：
1. 等待几分钟后重试
2. 升级账户配额
3. 使用更便宜的模型（如 gpt-4o-mini）

### ❌ Error: "Connection refused"

**原因**：无法连接到 API 服务器

**解决**：
1. 检查网络连接
2. 检查防火墙设置
3. 尝试改用代理

### ❌ Error: "CUDA out of memory" (如果使用本地模型)

**原因**：显存不足

**解决**：
1. 使用更小的模型（如 7B 而非 70B）
2. 启用量化（int8, int4）
3. 增加 GPU 显存

---

## 成本估算

### OpenAI 官方价格（2024）

| 模型 | 输入价格 | 输出价格 | 使用场景 |
|------|---------|---------|----------|
| gpt-4-turbo | $0.01/1K tokens | $0.03/1K tokens | 最好质量 |
| gpt-4o | $0.005/1K tokens | $0.015/1K tokens | 平衡 |
| gpt-4o-mini | $0.00015/1K tokens | $0.0006/1K tokens | **预算友好** |

### 生成 1 条轨迹的成本

假设：
- 10 步交互
- 每步平均 500 token 思维输入 + 200 token 输出
- 1 张图像 = ~300 token (base64)

```
成本 = (500 输入 × 10 步 + 300 × 10 图 + 200 输出 × 10 步)
     = (5000 + 3000 + 2000) = 10,000 tokens
     
gpt-4o-mini: 10,000 × ($0.00015 + $0.0006) = $0.0075 ≈ 0.05 RMB
gpt-4-turbo: 10,000 × ($0.01 + $0.03) = $0.4 ≈ 3 RMB

生成 9,300 条轨迹（完整数据集）：
gpt-4o-mini: $70 ≈ 460 RMB
gpt-4-turbo: $3,720 ≈ 24,000 RMB ⚠️ 贵！
```

---

## 生产环境建议

### 方案 1：使用国内镜像加速

**阿里云DashScope** (支持国内加速)：
```python
# vlmCall.py
conn = http.client.HTTPSConnection("dashscope.aliyuncs.com")
headers = {
    'Authorization': 'Bearer ' + api_key,
    'Content-Type': 'application/json'
}
```

### 方案 2：使用本地开源模型

```bash
# 使用 ollama 或 vLLM
ollama run llava  # 多模态模型（支持图像）

# 改 vlmCall.py 的 API 端点
conn = http.client.HTTPSConnection("localhost:11434")
```

### 方案 3：混合方案

```python
# 便宜的操作用 gpt-4o-mini
# 关键操作用 gpt-4-turbo
# 离线操作用本地模型

def vlm_request(prompt_type, ...):
    if prompt_type == "simple_navigation":
        model = "gpt-4o-mini"  # 便宜
    elif prompt_type == "complex_reasoning":
        model = "gpt-4-turbo"  # 好
    elif prompt_type == "offline":
        model = "local_llava"  # 免费
```

---

## 验证数据生成管道

### 验证 1：TaskGenerate 工作正常

```bash
cd data_engine
python TaskGenerate.py

# 检查输出
ls single_search_task_metadata/
cat single_search_task_metadata/FloorPlan1.json | head -20
```

**预期输出**：
```json
[
  {
    "taskname": "Find the Apple in the room.",
    "tasktype": "single_search",
    "actions": [
      {"action": "navigate to", "objectId": "CounterTop|...", "reward": 1},
      {"action": "end", "reward": 1}
    ],
    "totalreward": 2
  }
]
```

### 验证 2：o1StyleGenerate 工作正常

```bash
cd data_engine
python o1StyleGenerate.py

# 检查输出
ls -la single_search/FloorPlan1/
cat single_search/FloorPlan1/trajectory_0.json | head -30
```

**预期输出**：
```json
{
  "scene": "FloorPlan1",
  "tasktype": "single_search",
  "taskname": "Find the Apple in the room.",
  "trajectory": [
    "<Observation> I see a kitchen with...",
    "<Thought> The apple is likely on...",
    "<Decision> I should navigate to...",
    ...
  ],
  "images": ["single_search/FloorPlan1/step_0.png", ...]
}
```

---

## 监控和调试

### 启用详细日志

**文件**: `vlmCall.py`

```python
# 添加日志
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def vlm_request(self, ...):
    logger.debug(f"Sending request to {self.model}")
    logger.debug(f"Payload: {payload[:200]}...")
    # ...
    logger.debug(f"Response: {content[:200]}...")
```

### 统计 API 调用

```python
# 在 o1StyleGenerate.py 中添加
class APICallTracker:
    def __init__(self):
        self.call_count = 0
        self.total_cost = 0.0
    
    def log_call(self, model, input_tokens, output_tokens):
        self.call_count += 1
        prices = {
            "gpt-4o-mini": (0.00015, 0.0006),
            "gpt-4-turbo": (0.01, 0.03),
        }
        in_p, out_p = prices.get(model, (0, 0))
        cost = (input_tokens * in_p + output_tokens * out_p) / 1000
        self.total_cost += cost
        print(f"Call #{self.call_count}: {model} - Cost: ${cost:.4f} (Total: ${self.total_cost:.2f})")

tracker = APICallTracker()
# tracker.log_call(model, input_tokens, output_tokens)
```

---

## 下一步

✅ 配置完成后：
1. 运行 `test_vlm.py` 验证连接
2. 执行 `TaskGenerate.py` 生成任务
3. 执行 `o1StyleGenerate.py` 生成轨迹
4. 检查 `data/` 文件夹中的结果

📚 有问题参考：
- [OpenAI 文档](https://platform.openai.com/docs)
- [项目 README](../README.md)
- [完整架构文档](../PROJECT_ARCHITECTURE.md)
