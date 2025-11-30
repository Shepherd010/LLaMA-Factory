# Copyright 2025 OpenAccess AI Collective and the LlamaFactory team.
#
# This code is inspired by the OpenAccess AI Collective's axolotl library.
# https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/monkeypatch/utils.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import DataCollatorForSeq2Seq

from ..extras.constants import AUDIO_PLACEHOLDER, IGNORE_INDEX, IMAGE_PLACEHOLDER
from ..extras.packages import is_pillow_available


if is_pillow_available():
    from PIL import Image


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from .template import Template


def prepare_4d_attention_mask(attention_mask_with_indices: "torch.Tensor", dtype: "torch.dtype") -> "torch.Tensor":
    r"""Expand 2d attention mask to 4d attention mask.

    Expand the attention mask with indices from (batch_size, seq_len) to (batch_size, 1, seq_len, seq_len),
    handle packed sequences and transforms the mask to lower triangular form to prevent future peeking.

    e.g.
    ```python
    # input
    [[1, 1, 2, 2, 2, 0]]
    # output
    [
        [
            [
                [o, x, x, x, x, x],
                [o, o, x, x, x, x],
                [x, x, o, x, x, x],
                [x, x, o, o, x, x],
                [x, x, o, o, o, x],
                [x, x, x, x, x, x],
            ]
        ]
    ]
    ```
    where `o` equals to `0.0`, `x` equals to `min_dtype`.
    """
    _, seq_len = attention_mask_with_indices.size()
    min_dtype = torch.finfo(dtype).min
    zero_tensor = torch.tensor(0, dtype=dtype)

    # Create a non-padding mask.
    non_padding_mask = (attention_mask_with_indices != 0).unsqueeze(1).unsqueeze(2)
    # Create indices for comparison.
    indices = attention_mask_with_indices.unsqueeze(1).unsqueeze(2)  # [bsz, 1, 1, seq_len]
    indices_t = attention_mask_with_indices.unsqueeze(1).unsqueeze(3)  # [bsz, 1, seq_len, 1]
    # Create a lower triangular mask.
    tril_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
    attention_mask_4d = (indices == indices_t) & non_padding_mask & tril_mask
    # Invert the attention mask.
    attention_mask_4d = torch.where(attention_mask_4d, zero_tensor, min_dtype)
    return attention_mask_4d


@dataclass
class MultiModalDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    r"""Data collator that supports VLMs.

    Features should contain input_ids, attention_mask, labels, and optionally contain images, videos and audios.
    """

    template: Optional["Template"] = None
    processor: Optional["ProcessorMixin"] = None

    def __post_init__(self):
        if self.template is None:
            raise ValueError("Template is required for MultiModalDataCollator.")

        if isinstance(self.model, PeftModel):
            self.model = self.model.base_model.model

        if self.model is not None and hasattr(self.model, "get_rope_index"):  # for qwen2vl mrope
            self.get_rope_func = self.model.get_rope_index  # transformers < 4.52.0 or qwen2.5 omni
        elif self.model is not None and hasattr(self.model, "model") and hasattr(self.model.model, "get_rope_index"):
            self.get_rope_func = self.model.model.get_rope_index  # transformers >= 4.52.0
        else:
            self.get_rope_func = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        semantic_ids = None
        if "semantic_ids" in features[0]:
            semantic_ids = [f.pop("semantic_ids") for f in features]

        batch_images, batch_videos, batch_audios = [], [], []
        batch_imglens, batch_vidlens, batch_audlens, batch_input_ids = [], [], [], []
        for feature in features:
            images = feature.pop("images", None) or []
            videos = feature.pop("videos", None) or []
            audios = feature.pop("audios", None) or []
            batch_images.extend(images)
            batch_videos.extend(videos)
            batch_audios.extend(audios)
            batch_imglens.append(len(images))
            batch_vidlens.append(len(videos))
            batch_audlens.append(len(audios))
            batch_input_ids.append(feature["input_ids"])

        fake_input_ids = []
        if (
            self.template.mm_plugin.image_token is not None and sum(batch_imglens) == 0 and sum(batch_vidlens) == 0
        ):  # avoid process hanging in zero3/fsdp case
            fake_messages = [{"role": "user", "content": IMAGE_PLACEHOLDER}]
            fake_images = [Image.new("RGB", (64, 64), (255, 255, 255))]
            fake_messages = self.template.mm_plugin.process_messages(
                fake_messages, fake_images, [], [], self.processor
            )
            _fake_input_ids = self.tokenizer.encode(fake_messages[0]["content"], add_special_tokens=False)
            _fake_input_ids, _ = self.template.mm_plugin.process_token_ids(
                _fake_input_ids, None, fake_images, [], [], self.tokenizer, self.processor
            )
            fake_input_ids.extend(_fake_input_ids)
            batch_images = fake_images
            batch_imglens[0] = 1

        if (
            self.template.mm_plugin.audio_token is not None and sum(batch_audlens) == 0
        ):  # avoid process hanging in zero3/fsdp case
            fake_messages = [{"role": "user", "content": AUDIO_PLACEHOLDER}]
            fake_audios = [np.zeros(1600)]
            fake_messages = self.template.mm_plugin.process_messages(
                fake_messages, [], [], fake_audios, self.processor
            )
            _fake_input_ids = self.tokenizer.encode(fake_messages[0]["content"], add_special_tokens=False)
            _fake_input_ids, _ = self.template.mm_plugin.process_token_ids(
                _fake_input_ids, None, [], [], fake_audios, self.tokenizer, self.processor
            )
            fake_input_ids.extend(_fake_input_ids)
            batch_audios = fake_audios
            batch_audlens[0] = 1

        if len(fake_input_ids) != 0:
            if self.tokenizer.padding_side == "right":
                features[0]["input_ids"] = features[0]["input_ids"] + fake_input_ids
                features[0]["attention_mask"] = features[0]["attention_mask"] + [0] * len(fake_input_ids)
                features[0]["labels"] = features[0]["labels"] + [IGNORE_INDEX] * len(fake_input_ids)
            else:
                features[0]["input_ids"] = fake_input_ids + features[0]["input_ids"]
                features[0]["attention_mask"] = [0] * len(fake_input_ids) + features[0]["attention_mask"]
                features[0]["labels"] = [IGNORE_INDEX] * len(fake_input_ids) + features[0]["labels"]

            batch_input_ids[0] = features[0]["input_ids"]

        mm_inputs = self.template.mm_plugin.get_mm_inputs(
            batch_images,
            batch_videos,
            batch_audios,
            batch_imglens,
            batch_vidlens,
            batch_audlens,
            batch_input_ids,
            self.processor,
        )
        if "token_type_ids" in mm_inputs:
            token_type_ids = mm_inputs.pop("token_type_ids")
            for i, feature in enumerate(features):
                feature["token_type_ids"] = token_type_ids[i]

        features: dict[str, torch.Tensor] = super().__call__(features)

        if self.get_rope_func is not None:
            # Validate and adjust image_grid_thw/video_grid_thw to match actual tokens in input_ids
            image_grid_thw = mm_inputs.get("image_grid_thw")
            video_grid_thw = mm_inputs.get("video_grid_thw")
            pixel_values = mm_inputs.get("pixel_values")
            
            if image_grid_thw is not None or video_grid_thw is not None:
                image_token_id = getattr(self.model.config, "image_token_id", None)
                video_token_id = getattr(self.model.config, "video_token_id", None)
                vision_start_token_id = getattr(self.model.config, "vision_start_token_id", None)
                spatial_merge_size = getattr(self.model.config, "vision_config", {})
                if hasattr(spatial_merge_size, "spatial_merge_size"):
                    spatial_merge_size = spatial_merge_size.spatial_merge_size
                else:
                    spatial_merge_size = 2  # default value for qwen vl models
                
                if vision_start_token_id is not None:
                    batch_input_ids = features["input_ids"]
                    batch_attention_mask = features["attention_mask"]
                    batch_labels = features.get("labels")
                    valid_mask = batch_attention_mask >= 1
                    
                    # Calculate how many image tokens the grid_thw can support
                    if image_grid_thw is not None and len(image_grid_thw) > 0:
                        max_supported_image_tokens = 0
                        total_patches = 0
                        for grid in image_grid_thw:
                            t, h, w = grid[0].item(), grid[1].item(), grid[2].item()
                            patches = t * h * w
                            tokens = patches // (spatial_merge_size ** 2)
                            max_supported_image_tokens += tokens
                            total_patches += patches
                    else:
                        max_supported_image_tokens = 0
                        total_patches = 0
                    
                    if video_grid_thw is not None and len(video_grid_thw) > 0:
                        max_supported_video_tokens = 0
                        for grid in video_grid_thw:
                            t, h, w = grid[0].item(), grid[1].item(), grid[2].item()
                            tokens = (t * h * w) // (spatial_merge_size ** 2)
                            max_supported_video_tokens += tokens
                    else:
                        max_supported_video_tokens = 0
                    
                    # Count actual tokens in input_ids
                    if image_token_id is not None:
                        actual_image_tokens = ((batch_input_ids == image_token_id) & valid_mask).sum().item()
                    else:
                        actual_image_tokens = 0
                    
                    if video_token_id is not None:
                        actual_video_tokens = ((batch_input_ids == video_token_id) & valid_mask).sum().item()
                    else:
                        actual_video_tokens = 0
                    
                    # Case 1: input_ids has MORE image tokens than grid_thw can support
                    # -> Need to truncate input_ids by replacing excess image tokens with pad
                    if actual_image_tokens > max_supported_image_tokens and image_token_id is not None:
                        excess_tokens = actual_image_tokens - max_supported_image_tokens
                        # Clone tensors to avoid in-place modification issues
                        batch_input_ids = batch_input_ids.clone()
                        batch_attention_mask = batch_attention_mask.clone()
                        if batch_labels is not None:
                            batch_labels = batch_labels.clone()
                        
                        # Replace excess image tokens from the END with pad_token_id
                        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                        for batch_idx in range(batch_input_ids.shape[0]):
                            # Find all image token positions in this sample
                            image_positions = (batch_input_ids[batch_idx] == image_token_id).nonzero(as_tuple=True)[0]
                            if len(image_positions) > 0:
                                # Calculate how many to remove from this sample
                                # Proportionally distribute based on sample's share of total image tokens
                                sample_image_count = len(image_positions)
                                tokens_to_remove = min(excess_tokens, sample_image_count - (max_supported_image_tokens // max(1, batch_input_ids.shape[0])))
                                tokens_to_remove = max(0, tokens_to_remove)
                                
                                if tokens_to_remove > 0:
                                    # Remove from the end
                                    positions_to_remove = image_positions[-tokens_to_remove:]
                                    batch_input_ids[batch_idx, positions_to_remove] = pad_token_id
                                    batch_attention_mask[batch_idx, positions_to_remove] = 0
                                    if batch_labels is not None:
                                        batch_labels[batch_idx, positions_to_remove] = -100
                                    excess_tokens -= tokens_to_remove
                            
                            if excess_tokens <= 0:
                                break
                        
                        features["input_ids"] = batch_input_ids
                        features["attention_mask"] = batch_attention_mask
                        if batch_labels is not None:
                            features["labels"] = batch_labels
                    
                    # Case 2: input_ids has FEWER image tokens than grid_thw
                    # -> Need to truncate grid_thw and pixel_values
                    elif actual_image_tokens < max_supported_image_tokens and image_grid_thw is not None:
                        expected_tokens = 0
                        valid_grids = []
                        patches_to_keep = 0
                        for grid in image_grid_thw:
                            t, h, w = grid[0].item(), grid[1].item(), grid[2].item()
                            patches = t * h * w
                            tokens = patches // (spatial_merge_size ** 2)
                            if expected_tokens + tokens <= actual_image_tokens:
                                expected_tokens += tokens
                                patches_to_keep += patches
                                valid_grids.append(grid)
                            else:
                                break
                        
                        if len(valid_grids) > 0:
                            image_grid_thw = torch.stack(valid_grids)
                            if pixel_values is not None and patches_to_keep < pixel_values.shape[0]:
                                mm_inputs["pixel_values"] = pixel_values[:patches_to_keep]
                        else:
                            image_grid_thw = None
                            if "pixel_values" in mm_inputs:
                                del mm_inputs["pixel_values"]
                    
                    # Similar handling for video (Case 2 only - truncate grid if needed)
                    if actual_video_tokens < max_supported_video_tokens and video_grid_thw is not None:
                        expected_tokens = 0
                        valid_grids = []
                        for grid in video_grid_thw:
                            t, h, w = grid[0].item(), grid[1].item(), grid[2].item()
                            tokens = (t * h * w) // (spatial_merge_size ** 2)
                            if expected_tokens + tokens <= actual_video_tokens:
                                expected_tokens += tokens
                                valid_grids.append(grid)
                            else:
                                break
                        if len(valid_grids) > 0:
                            video_grid_thw = torch.stack(valid_grids)
                        else:
                            video_grid_thw = None
                    
                    # Update mm_inputs
                    if image_grid_thw is not None:
                        mm_inputs["image_grid_thw"] = image_grid_thw
                    elif "image_grid_thw" in mm_inputs:
                        del mm_inputs["image_grid_thw"]
                    
                    if video_grid_thw is not None:
                        mm_inputs["video_grid_thw"] = video_grid_thw
                    elif "video_grid_thw" in mm_inputs:
                        del mm_inputs["video_grid_thw"]
                    elif "video_grid_thw" in mm_inputs:
                        del mm_inputs["video_grid_thw"]
            
            rope_index_kwargs = {
                "input_ids": features["input_ids"],
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "attention_mask": (features["attention_mask"] >= 1).float(),
            }
            if "second_per_grid_ts" in mm_inputs:  # for qwen2vl
                rope_index_kwargs["second_per_grid_ts"] = mm_inputs.get("second_per_grid_ts")
            elif "video_second_per_grid" in mm_inputs:  # for qwen2.5 omni
                rope_index_kwargs["second_per_grids"] = mm_inputs.get("video_second_per_grid")

            if getattr(self.model.config, "model_type", None) in ["qwen2_5_omni_thinker", "qwen3_omni_moe_thinker"]:
                rope_index_kwargs["use_audio_in_video"] = getattr(self.processor, "use_audio_in_video", False)
                feature_attention_mask = mm_inputs.get("feature_attention_mask", None)
                if feature_attention_mask is not None:  # FIXME: need to get video image lengths
                    audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
                    rope_index_kwargs["audio_seqlens"] = audio_feature_lengths  # prepare for input

                features["position_ids"], rope_deltas = self.get_rope_func(**rope_index_kwargs)
                features["rope_deltas"] = rope_deltas - (1 - rope_index_kwargs["attention_mask"]).sum(
                    dim=-1
                ).unsqueeze(-1)
            else:  # for qwen vl
                try:
                    features["position_ids"], features["rope_deltas"] = self.get_rope_func(**rope_index_kwargs)
                except (RuntimeError, IndexError) as e:
                    if "shape mismatch" in str(e) or "index" in str(e).lower() or "out of bounds" in str(e).lower():
                        # Fallback: if grid_thw mismatch persists, set them to None to use default position ids
                        rope_index_kwargs["image_grid_thw"] = None
                        rope_index_kwargs["video_grid_thw"] = None
                        # Also need to clear pixel_values to avoid feature mismatch in forward pass
                        if "pixel_values" in mm_inputs:
                            del mm_inputs["pixel_values"]
                        if "image_grid_thw" in mm_inputs:
                            del mm_inputs["image_grid_thw"]
                        if "video_grid_thw" in mm_inputs:
                            del mm_inputs["video_grid_thw"]
                        features["position_ids"], features["rope_deltas"] = self.get_rope_func(**rope_index_kwargs)
                    else:
                        raise

        if (
            self.model is not None
            and getattr(self.model.config, "model_type", None)
            in [
                "glm4v",
                "Keye",
                "qwen2_vl",
                "qwen2_5_vl",
                "qwen2_5_omni_thinker",
                "qwen3_omni_moe_thinker",
                "qwen3_vl",
                "qwen3_vl_moe",
            ]
            and ("position_ids" not in features or features["position_ids"].dim() != 3)
        ):
            raise ValueError(f"{self.model.config.model_type} requires 3D position ids for mrope.")

        if "cross_attention_mask" in mm_inputs:  # for mllama inputs when pad_to_multiple_of is enabled
            cross_attention_mask = mm_inputs.pop("cross_attention_mask")
            seq_len = features["input_ids"].size(1)
            orig_len = cross_attention_mask.size(1)
            mm_inputs["cross_attention_mask"] = F.pad(cross_attention_mask, (0, 0, 0, 0, 0, seq_len - orig_len))

        features.update(mm_inputs)

        if semantic_ids is not None:
            sem_ids = [torch.tensor(s, dtype=torch.long) for s in semantic_ids]
            batch_input_ids = features["input_ids"]
            batch_size, seq_len = batch_input_ids.shape
            padded_sem_ids = []
            for s in sem_ids:
                pad_len = seq_len - len(s)
                if self.tokenizer.padding_side == "left":
                    padded_s = torch.cat([torch.full((pad_len,), 2, dtype=torch.long), s])
                else:
                    padded_s = torch.cat([s, torch.full((pad_len,), 2, dtype=torch.long)])
                padded_sem_ids.append(padded_s)
            features["semantic_ids"] = torch.stack(padded_sem_ids)

        if "image_bound" in features:  # for minicpmv inputs
            bsz, seq_length = features["input_ids"].shape
            features["position_ids"] = torch.arange(seq_length).long().repeat(bsz, 1)
            return {"data": features, "input_ids": features["input_ids"], "labels": features["labels"]}

        return features


@dataclass
class SFTDataCollatorWith4DAttentionMask(MultiModalDataCollatorForSeq2Seq):
    r"""Data collator for 4d attention mask."""

    block_diag_attn: bool = False
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager"
    compute_dtype: "torch.dtype" = torch.float32

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        features = super().__call__(features)
        if self.block_diag_attn and self.attn_implementation != "flash_attention_2":
            features["attention_mask"] = prepare_4d_attention_mask(features["attention_mask"], self.compute_dtype)

        for key, value in features.items():  # cast data dtype for paligemma
            if torch.is_tensor(value) and torch.is_floating_point(value):
                features[key] = value.to(self.compute_dtype)

        return features


@dataclass
class PairwiseDataCollatorWithPadding(MultiModalDataCollatorForSeq2Seq):
    r"""Data collator for pairwise data."""

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        r"""Pad batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        concatenated_features = []
        for key in ("chosen", "rejected"):
            for feature in features:
                target_feature = {
                    "input_ids": feature[f"{key}_input_ids"],
                    "attention_mask": feature[f"{key}_attention_mask"],
                    "labels": feature[f"{key}_labels"],
                    "images": feature["images"],
                    "videos": feature["videos"],
                    "audios": feature["audios"],
                }
                concatenated_features.append(target_feature)

        return super().__call__(concatenated_features)


@dataclass
class KTODataCollatorWithPadding(MultiModalDataCollatorForSeq2Seq):
    r"""Data collator for KTO data."""

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        target_features = []
        kl_features = []
        kto_tags = []
        for feature in features:
            target_feature = {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
                "labels": feature["labels"],
                "images": feature["images"],
                "videos": feature["videos"],
                "audios": feature["audios"],
            }
            kl_feature = {
                "input_ids": feature["kl_input_ids"],
                "attention_mask": feature["kl_attention_mask"],
                "labels": feature["kl_labels"],
                "images": feature["images"],
                "videos": feature["videos"],
                "audios": feature["audios"],
            }
            target_features.append(target_feature)
            kl_features.append(kl_feature)
            kto_tags.append(feature["kto_tags"])

        batch = super().__call__(target_features)
        kl_batch = super().__call__(kl_features)
        batch["kl_input_ids"] = kl_batch["input_ids"]
        batch["kl_attention_mask"] = kl_batch["attention_mask"]
        batch["kl_labels"] = kl_batch["labels"]
        if "cross_attention_mask" in kl_batch:  # for mllama inputs
            batch["kl_cross_attention_mask"] = kl_batch["cross_attention_mask"]

        if "token_type_ids" in kl_batch:
            batch["kl_token_type_ids"] = kl_batch["token_type_ids"]

        batch["kto_tags"] = torch.tensor(kto_tags)
        return batch
