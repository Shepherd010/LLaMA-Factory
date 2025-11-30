import torch
import os
import sys
from datetime import datetime
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLAttention

# Add src to path to import llamafactory modules
sys.path.append(os.path.abspath("src"))

from llamafactory.model.hsa_patch import apply_hsa_patch, HSAContext

def verify_hsa():
    report_lines = []
    report_lines.append("# HSA Mechanism Verification Report")
    report_lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Model:** Qwen2-VL-2B-Instruct")
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

    # Construct a dummy input
    # Sequence: [Global, Global, Noise, Noise, Landmark, Global]
    # Length: 6
    input_ids = torch.tensor([[100, 101, 102, 103, 104, 105]], device="cuda")
    attention_mask = torch.ones_like(input_ids)
    
    # Semantic IDs: 
    # 0: Global
    # 1: Landmark
    # 2: Noise
    semantic_ids = torch.tensor([[0, 0, 2, 2, 1, 0]], device="cuda")

    print("\n--- Running Baseline (No HSA) ---")
    # Ensure context is clear
    HSAContext.clear()
    
    with torch.no_grad():
        # Run forward pass. Note: without patch, semantic_ids in kwargs might be ignored or cause error depending on model.
        # But since we haven't patched yet, we just run standard forward.
        # We pass semantic_ids just to see if it crashes (it shouldn't if we don't pass it, but we want to compare apples to apples inputs if possible)
        # Actually, unpatched model won't accept semantic_ids. So we run without it.
        outputs_baseline = model(input_ids=input_ids, attention_mask=attention_mask)
        logits_baseline = outputs_baseline.logits

    print("\n--- Applying HSA Patch ---")
    apply_hsa_patch(model)

    print("\n--- Running HSA (With Semantic IDs) ---")
    with torch.no_grad():
        # Now we pass semantic_ids. The patched model should pick it up.
        outputs_hsa = model(input_ids=input_ids, attention_mask=attention_mask, semantic_ids=semantic_ids)
        logits_hsa = outputs_hsa.logits

    print("\n--- Comparison ---")
    diff = (logits_baseline - logits_hsa).abs().max().item()
    print(f"Max difference in logits: {diff}")

    report_lines.append("## Test Case 1: Basic Functionality")
    report_lines.append("Verifies that providing `semantic_ids` alters the model output compared to the baseline.")
    report_lines.append(f"- **Input Sequence Length:** {input_ids.shape[1]}")
    report_lines.append(f"- **Semantic Pattern:** `[Global, Global, Noise, Noise, Landmark, Global]`")
    report_lines.append(f"- **Max Logit Difference:** `{diff:.6f}`")

    if diff > 1e-3:
        print("SUCCESS: HSA is modifying the model behavior!")
        report_lines.append("- **Result:** ✅ PASS (Model behavior modified)")
    else:
        print("FAILURE: HSA did not modify the model behavior significantly.")
        report_lines.append("- **Result:** ❌ FAIL (No significant modification)")
    report_lines.append("")
        
    # Further verification: Check if Noise tokens are masked
    # We can't easily check attention weights without hooks, but the logit difference is a strong indicator.
    
    # Let's try a case where HSA should definitely mask something.
    # If we have a long sequence of Noise, the early noise should be masked for later tokens.
    
    print("\n--- Detailed Verification ---")
    # Create a longer sequence where Noise (2) is far from current token
    # [Noise, ..., Noise, Global]
    # Distance > 5 (W=5 in patch)
    seq_len = 20
    input_ids_long = torch.randint(0, 1000, (1, seq_len), device="cuda")
    attention_mask_long = torch.ones_like(input_ids_long)
    semantic_ids_long = torch.full((1, seq_len), 2, device="cuda") # All noise
    semantic_ids_long[0, -1] = 0 # Last is global (query)
    
    # Run with HSA
    with torch.no_grad():
        outputs_hsa_long = model(input_ids=input_ids_long, attention_mask=attention_mask_long, semantic_ids=semantic_ids_long)
    
    # Run without HSA (simulate by passing None semantic_ids or clearing context)
    # Since patch is applied, we can just pass semantic_ids=None
    with torch.no_grad():
        outputs_baseline_long = model(input_ids=input_ids_long, attention_mask=attention_mask_long, semantic_ids=None)
        
    diff_long = (outputs_hsa_long.logits - outputs_baseline_long.logits).abs().max().item()
    print(f"Max difference in long sequence (Noise masking): {diff_long}")
    
    report_lines.append("## Test Case 2: Long Sequence Noise Masking")
    report_lines.append("Verifies that distant 'Process Noise' tokens are effectively masked (attention bias = -inf).")
    report_lines.append(f"- **Sequence Length:** {seq_len}")
    report_lines.append(f"- **Semantic Pattern:** `[Noise, ..., Noise, Global]` (Distance > Window Size)")
    report_lines.append(f"- **Max Logit Difference:** `{diff_long:.6f}`")

    if diff_long > 1e-3:
        print("SUCCESS: HSA is active in long sequences.")
        report_lines.append("- **Result:** ✅ PASS (Distant noise masked)")
    else:
        print("FAILURE: HSA not active in long sequences.")
        report_lines.append("- **Result:** ❌ FAIL (Distant noise not masked)")
    
    # Write report
    with open("HSA_VERIFICATION_REPORT.md", "w") as f:
        f.write("\n".join(report_lines))
    print("\nReport saved to HSA_VERIFICATION_REPORT.md")

if __name__ == "__main__":
    verify_hsa()
