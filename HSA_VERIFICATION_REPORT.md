# HSA Mechanism Verification Report
**Date:** 2025-11-26 16:57:09
**Model:** Qwen2-VL-2B-Instruct

## Test Case 1: Basic Functionality
Verifies that providing `semantic_ids` alters the model output compared to the baseline.
- **Input Sequence Length:** 6
- **Semantic Pattern:** `[Global, Global, Noise, Noise, Landmark, Global]`
- **Max Logit Difference:** `16.656250`
- **Result:** ✅ PASS (Model behavior modified)

## Test Case 2: Long Sequence Noise Masking
Verifies that distant 'Process Noise' tokens are effectively masked (attention bias = -inf).
- **Sequence Length:** 20
- **Semantic Pattern:** `[Noise, ..., Noise, Global]` (Distance > Window Size)
- **Max Logit Difference:** `27.031250`
- **Result:** ✅ PASS (Distant noise masked)