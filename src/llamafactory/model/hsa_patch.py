import torch
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLAttention, Qwen2VLModel

# Context to store semantic_ids globally for the current forward pass
class HSAContext:
    _semantic_ids = None

    @classmethod
    def set(cls, semantic_ids):
        cls._semantic_ids = semantic_ids

    @classmethod
    def get(cls):
        return cls._semantic_ids

    @classmethod
    def clear(cls):
        cls._semantic_ids = None

def apply_hsa_patch(model):
    # 1. Patch Qwen2VLModel.forward to capture semantic_ids
    original_model_forward = Qwen2VLModel.forward

    def hsa_model_forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Capture semantic_ids from kwargs
        semantic_ids = kwargs.pop("semantic_ids", None)
        # Always update context (set to None if not provided to clear previous state)
        HSAContext.set(semantic_ids)
        
        # Call original forward
        return original_model_forward(self, input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    # 2. Patch Qwen2VLAttention.forward to use semantic_ids
    original_attn_forward = Qwen2VLAttention.forward
    
    def hsa_forward(self, hidden_states, attention_mask=None, **kwargs):
        # Try to get from kwargs first, then from context
        semantic_ids = kwargs.get("semantic_ids")
        if semantic_ids is None:
            semantic_ids = HSAContext.get()
        
        if semantic_ids is not None:
            # 1. 计算 Bias Matrix (参考 HSA_ARCHITECTURE.md 中的公式)
            batch_size, seq_len = semantic_ids.shape
            device = semantic_ids.device
            dtype = hidden_states.dtype
            
            # Indices for distance calculation
            indices = torch.arange(seq_len, device=device)
            # |i - j|: [seq_len, seq_len]
            dist = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))
            
            # Semantic IDs for Key (j): [batch_size, 1, 1, seq_len]
            semantic_ids_j = semantic_ids.unsqueeze(1).unsqueeze(2)
            
            # Constants
            LAMBDA_1 = 0.001
            LAMBDA_2 = 0.5
            W = 5
            
            # Initialize bias matrix
            bias_matrix = torch.zeros((batch_size, 1, seq_len, seq_len), device=device, dtype=dtype)
            
            # Case 1: Global (0) -> 0 (Already set)
            
            # Case 2: Landmark (1) -> -lambda1 * |i - j|
            mask_landmark = (semantic_ids_j == 1)
            bias_matrix = torch.where(mask_landmark, -LAMBDA_1 * dist, bias_matrix)
            
            # Case 3: Noise (2)
            mask_noise = (semantic_ids_j == 2)
            
            # Noise Near: |i - j| <= W -> -lambda2 * |i - j|
            mask_noise_near = mask_noise & (dist <= W)
            bias_matrix = torch.where(mask_noise_near, -LAMBDA_2 * dist, bias_matrix)
            
            # Noise Far: |i - j| > W -> -inf
            mask_noise_far = mask_noise & (dist > W)
            min_dtype = torch.finfo(dtype).min
            bias_matrix = torch.where(mask_noise_far, torch.tensor(min_dtype, dtype=dtype), bias_matrix)
            
            # 2. 融合 Mask
            if attention_mask is None:
                attention_mask = bias_matrix
            else:
                # Ensure dimensions match for broadcasting if necessary
                attention_mask = attention_mask + bias_matrix
            
            # Ensure attention_mask dtype matches hidden_states (query) dtype for SDPA
            if attention_mask.dtype != hidden_states.dtype:
                attention_mask = attention_mask.to(hidden_states.dtype)
                
        return original_attn_forward(self, hidden_states, attention_mask=attention_mask, **kwargs)

    # 应用 Patch
    Qwen2VLModel.forward = hsa_model_forward
    Qwen2VLAttention.forward = hsa_forward
    print("HSA Patch Applied with Context Propagation!")
