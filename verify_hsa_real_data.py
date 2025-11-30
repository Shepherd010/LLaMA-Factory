import torch
import os
import sys
import json
import re
from datetime import datetime
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer

# Add src to path to import llamafactory modules
sys.path.append(os.path.abspath("src"))

from llamafactory.model.hsa_patch import apply_hsa_patch, HSAContext
from llamafactory.data.template import get_template_and_fix_tokenizer

from dataclasses import dataclass
from typing import Optional

@dataclass
class DataArguments:
    template: Optional[str] = None
    train_on_prompt: bool = False
    tool_format: Optional[str] = None
    default_system: Optional[str] = None
    enable_thinking: bool = True

def verify_hsa_real_data():
    report_lines = []
    report_lines.append("# HSA Verification Report (Real Data)")
    report_lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Model:** Qwen2-VL-2B-Instruct")
    report_lines.append(f"**Dataset:** data/hsa_test.json")
    report_lines.append("")

    print("Loading model...")
    model_path = "models/Qwen2-VL-2B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        device_map="cuda", 
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Get Template
    print("Loading Template...")
    data_args = DataArguments(template="qwen2_vl")
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # Load Data
    print("Loading Dataset...")
    with open("data/hsa_test.json", "r") as f:
        data = json.load(f)
    
    # Use the first example
    example = data[0]
    messages = example["messages"]
    
    # Preprocess messages: Remove <image> tags and extract system message
    system_content = None
    processed_messages = []
    
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        # Remove <image> tag
        content = content.replace("<image>", "")
        semantic_type = msg.get("semantic_type", "PROCESS_NOISE")
        
        if role == "system":
            system_content = content
        else:
            processed_messages.append({
                "role": role,
                "content": content,
                "semantic_type": semantic_type
            })

    # Encode
    print("Encoding messages...")
    # Set USE_HSA env var temporarily to ensure template encodes semantic ids
    os.environ["USE_HSA"] = "true"
    encoded_pairs = template.encode_multiturn_with_semantic(tokenizer, processed_messages, system=system_content)
    
    # Flatten to single sequence
    full_input_ids = []
    full_semantic_ids = []
    
    for source_ids, target_ids, source_sem_ids, target_sem_ids in encoded_pairs:
        full_input_ids.extend(source_ids + target_ids)
        full_semantic_ids.extend(source_sem_ids + target_sem_ids)
        
    # Add EOS if efficient_eos is enabled (usually handled in collator/processor, but let's check template)
    # For verification, the raw sequence is fine.
    
    input_ids_tensor = torch.tensor([full_input_ids], device="cuda")
    semantic_ids_tensor = torch.tensor([full_semantic_ids], device="cuda")
    attention_mask_tensor = torch.ones_like(input_ids_tensor)
    
    print(f"Input Sequence Length: {input_ids_tensor.shape[1]}")
    
    # Analyze Semantic IDs distribution
    sem_counts = {
        0: (semantic_ids_tensor == 0).sum().item(),
        1: (semantic_ids_tensor == 1).sum().item(),
        2: (semantic_ids_tensor == 2).sum().item()
    }
    print(f"Semantic ID Counts: Global(0)={sem_counts[0]}, Landmark(1)={sem_counts[1]}, Noise(2)={sem_counts[2]}")
    
    report_lines.append("## Data Statistics")
    report_lines.append(f"- **Total Tokens:** {input_ids_tensor.shape[1]}")
    report_lines.append(f"- **Global Tokens (0):** {sem_counts[0]}")
    report_lines.append(f"- **Landmark Tokens (1):** {sem_counts[1]}")
    report_lines.append(f"- **Noise Tokens (2):** {sem_counts[2]}")
    report_lines.append("")

    print("\n--- Running Baseline (No HSA) ---")
    HSAContext.clear()
    with torch.no_grad():
        outputs_baseline = model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
        logits_baseline = outputs_baseline.logits

    print("\n--- Applying HSA Patch ---")
    apply_hsa_patch(model)

    print("\n--- Running HSA (With Semantic IDs) ---")
    with torch.no_grad():
        outputs_hsa = model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor, semantic_ids=semantic_ids_tensor)
        logits_hsa = outputs_hsa.logits

    print("\n--- Comparison ---")
    diff = (logits_baseline - logits_hsa).abs().max().item()
    print(f"Max difference in logits: {diff}")

    report_lines.append("## Verification Result")
    report_lines.append(f"- **Max Logit Difference:** `{diff:.6f}`")
    
    if diff > 1e-3:
        print("SUCCESS: HSA is modifying the model behavior on real data!")
        report_lines.append("- **Status:** ✅ PASS")
        report_lines.append("- **Conclusion:** The HSA mechanism is active and significantly alters the attention patterns for the provided real-world dataset example.")
    else:
        print("FAILURE: HSA did not modify the model behavior significantly.")
        report_lines.append("- **Status:** ❌ FAIL")
        report_lines.append("- **Conclusion:** No significant difference observed. Check if semantic_ids are correctly passed or if the sequence is too short to trigger noise masking.")

    # Write report
    with open("HSA_REAL_DATA_REPORT.md", "w") as f:
        f.write("\n".join(report_lines))
    print("\nReport saved to HSA_REAL_DATA_REPORT.md")

if __name__ == "__main__":
    verify_hsa_real_data()
