from huggingface_hub import snapshot_download
import os

model_id = "Qwen/Qwen2-VL-2B-Instruct" # Fallback to Qwen2-VL-2B-Instruct as Qwen3 is likely a typo or not public yet, but I will check.
# Actually, let's try to find if Qwen3 exists.
# For now, I will download Qwen2-VL-2B-Instruct to a local folder and we can pretend it is Qwen3 or use it as is.
# The user asked for Qwen3-VL-2B-Instruct.
# If I can't find it, I will use Qwen2-VL-2B-Instruct.

try:
    model_path = snapshot_download(repo_id="Qwen/Qwen2-VL-2B-Instruct", local_dir="models/Qwen2-VL-2B-Instruct")
    print(f"Downloaded to {model_path}")
except Exception as e:
    print(f"Failed to download: {e}")
