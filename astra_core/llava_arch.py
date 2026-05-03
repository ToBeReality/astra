import math
import os
import random
import re

import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import lru_cache
from typing import Optional

from llava.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
from llava.mm_utils import get_anyres_image_grid_shape
from llava.model.llava_arch import LlavaMetaForCausalLM, unpad_image
from llava.utils import rank0_print

from .configuration_astra import AstraConfig
from .utils import astra_compression


@lru_cache(maxsize=4)
def _load_siglip_text_components(model_id: str):
    """Lazy-load SigLIP full model and processor for text embeddings."""
    from transformers import AutoProcessor, SiglipModel

    model = SiglipModel.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor


def LlavaMetaForCausalLM_encode_images(self: LlavaMetaForCausalLM, images: torch.Tensor):
    # Keep raw SigLIP vision features (pre-projector) for query-aligned budgeting.
    raw_features, cls_attentions = self.get_model().get_vision_tower()(images)
    setattr(self, "_astra_last_raw_vision_features", raw_features)
    image_features = self.get_model().mm_projector(raw_features)
    return image_features, cls_attentions


def LlavaMetaForCausalLM_prepare_inputs_labels_for_multimodal(
    self: LlavaMetaForCausalLM,
    input_ids,
    position_ids,
    attention_mask,
    past_key_values,
    labels,
    images,
    modalities=["image"],
    image_sizes=None,
):
    vision_tower = self.get_vision_tower()
    # rank_print(modalities)
    if vision_tower is None or images is None or input_ids.shape[1] == 1:
        return input_ids, position_ids, attention_mask, past_key_values, None, labels

    if isinstance(modalities, str):
        modalities = [modalities]

    # import pdb; pdb.set_trace()
    if type(images) is list or images.ndim == 5:
        if type(images) is list:
            images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

        video_idx_in_batch = []
        for _ in range(len(modalities)):
            if modalities[_] == "video":
                video_idx_in_batch.append(_)

        images_list = []
        for image in images:
            if image.ndim == 4:
                images_list.append(image)
            else:
                images_list.append(image.unsqueeze(0))

        concat_images = torch.cat([image for image in images_list], dim=0)
        split_sizes = [image.shape[0] for image in images_list]
        encoded_image_features, cls_attentions = self.encode_images(concat_images)
        # image_features,all_faster_video_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)

        # This is a list, each element is [num_images, patch * patch, dim]
        # rank_print(f"Concat images : {concat_images.shape}")
        encoded_image_features = torch.split(encoded_image_features, split_sizes)
        image_features = []
        astra_config: AstraConfig = getattr(self, "astra_config")
        assert len(encoded_image_features) == 1, "Only support single video in a batch for now."
        for idx, image_feat in enumerate(encoded_image_features):
            if idx in video_idx_in_batch:
                # * Apply Astra here.
                # NOTE: astra_config.visual_token_* must be set for inner-LLM pruning (fastv_prune).
                # Use the *unpadded* token positions (attention_mask) to match the later insertion logic.
                ids0 = input_ids[0]
                if attention_mask is None:
                    ids0_nopad = ids0
                    mask0_nopad = torch.ones_like(ids0_nopad, dtype=torch.bool)
                else:
                    am0 = attention_mask[0].to(dtype=torch.bool)
                    ids0_nopad = ids0[am0]
                    mask0_nopad = torch.ones_like(ids0_nopad, dtype=torch.bool)

                pos = torch.where(ids0_nopad == IMAGE_TOKEN_INDEX)[0]
                astra_config.visual_token_start_index = int(pos[0].item()) if pos.numel() > 0 else None

                # Query-conditioned paths: SigLIP joint-space embeddings (stv_budget_* on astra_config).
                # No LLM embed_tokens fallback: post-merge semantic recycle uses siglip_pooled_token_relevance
                # when enabled, or attention-only selection when not.
                try:
                    astra_config.stv_budget_mu_d_raw = None
                    qtext = str(getattr(astra_config, "query_text", "") or "").strip()
                    raw_v = getattr(self, "_astra_last_raw_vision_features", None)
                    if qtext and raw_v is not None and raw_v.dim() == 3:
                        # raw_v: (T, P, Dv) for the video frames after vision tower (pre mm_projector).
                        T, P, _ = raw_v.shape
                        # Per-frame "visual CLS": mean over unpooled patch tokens.
                        v_frame = raw_v.float().mean(dim=1)  # (T, Dv)
                        # 帧间变化率（不经 SigLIP head / visual_projection）：与 analysis 脚本里「vision patch 均值」一致的定义。
                        try:
                            from astra.frame_selection import adjacent_change_mean

                            _, mu_d_raw_t = adjacent_change_mean(v_frame)
                            astra_config.stv_budget_mu_d_raw = float(
                                mu_d_raw_t.detach().float().cpu().item()
                            )
                        except Exception:
                            astra_config.stv_budget_mu_d_raw = None

                        # Load SigLIP full model to reuse its projection heads + text encoder.
                        # Prefer the actual vision tower identifier when possible.
                        vt = self.get_model().get_vision_tower()
                        siglip_id = getattr(vt, "vision_tower_name", None) or getattr(vt, "vision_tower", None) or "google/siglip-so400m-patch14-384"
                        siglip_id = str(siglip_id)
                        siglip_model, siglip_proc = _load_siglip_text_components(siglip_id)
                        device = v_frame.device
                        siglip_model = siglip_model.to(device)

                        # Build per-frame visual embeddings in SigLIP joint space (pre-mm_projector path).
                        #
                        # IMPORTANT (efficiency): calling `siglip_model.get_image_features(pixel_values=...)`
                        # would run the SigLIP vision tower *again*, which can inflate TTFT by ~1s.
                        # By default, reuse the already-computed `raw_v` (vision tower output) and only
                        # apply SigLIP heads/projection. Keep the official API as an optional diagnostic.
                        v_cls_proj = None
                        try:
                            v_cls = siglip_model.vision_model.head(raw_v.float())  # (T, Dv)
                            if getattr(siglip_model, "visual_projection", None) is not None:
                                v_cls = siglip_model.visual_projection(v_cls)
                            v_cls_proj = F.normalize(v_cls, dim=-1)  # (T, De)
                        except Exception:
                            v_cls_proj = None

                        # Default: use head-based embedding (closest to "official" while reusing raw_v).
                        v_proj = v_cls_proj

                        # Fallback: raw token mean + visual projection.
                        if v_proj is None:
                            if getattr(siglip_model, "visual_projection", None) is not None:
                                v_proj = siglip_model.visual_projection(v_frame)  # (T, De)
                            else:
                                v_proj = v_frame
                            v_proj = F.normalize(v_proj, dim=-1)

                        # Optional (but default for "formal runs"): official SigLIP image embedding via
                        # `get_image_features(pixel_values=...)`.
                        #
                        # NOTE (efficiency): this runs the SigLIP vision tower again and can inflate TTFT.
                        # Control with env:
                        # - ASTRA_SIGLIP_OFFICIAL_IMAGE_FEATS=1/0  (default: 1)
                        # - ASTRA_FAST_TTFT=1 will force-disable it (for micro-benchmarks).
                        v_off = None
                        try:
                            force_fast = str(os.getenv("ASTRA_FAST_TTFT", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}
                            enable_official = str(os.getenv("ASTRA_SIGLIP_OFFICIAL_IMAGE_FEATS", "1")).strip().lower() in {"1", "true", "yes", "y", "on"}
                            enable_official = enable_official and (not force_fast)
                            if enable_official and isinstance(images_list, list) and idx < len(images_list):
                                pv = images_list[idx].to(device)
                                if pv is not None and pv.dim() == 4 and pv.shape[0] == T:
                                    v_off = siglip_model.get_image_features(pixel_values=pv)  # (T, De)
                                    v_off = F.normalize(v_off.float(), dim=-1)
                        except Exception:
                            v_off = None

                        # Encode text and project.
                        # SigLIP text model has a small max length (often 64); truncate to avoid indexing errors.
                        proc = siglip_proc(
                            text=[qtext],
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=64,
                        )
                        proc = {k: v.to(device) for k, v in proc.items()}
                        # Prefer official text embedding API; fallback to manual text tower + projection.
                        text_cls = None
                        try:
                            text_cls = siglip_model.get_text_features(
                                input_ids=proc["input_ids"],
                                attention_mask=proc.get("attention_mask", None),
                            )  # (1, De)
                        except Exception:
                            t_out = siglip_model.text_model(
                                input_ids=proc["input_ids"],
                                attention_mask=proc.get("attention_mask", None),
                                return_dict=True,
                            )
                            text_cls = getattr(t_out, "pooler_output", None)
                            if text_cls is None:
                                t_last = t_out.last_hidden_state
                                att = proc.get("attention_mask", None)
                                if att is None:
                                    text_cls = t_last.mean(dim=1)
                                else:
                                    att_f = att.to(dtype=t_last.dtype).unsqueeze(-1)
                                    text_cls = (t_last * att_f).sum(dim=1) / att_f.sum(dim=1).clamp_min(1.0)
                            if getattr(siglip_model, "text_projection", None) is not None:
                                text_cls = siglip_model.text_projection(text_cls)
                        text_cls = F.normalize(text_cls, dim=-1).squeeze(0)  # (De,)

                        astra_config.stv_budget_text_cls_embed = text_cls.detach()
                        astra_config.stv_budget_frame_features = v_proj.to(dtype=raw_v.dtype).detach()
                        astra_config.stv_budget_visual_cls_embed = v_proj.to(dtype=raw_v.dtype).detach()

                        if v_cls_proj is not None:
                            astra_config.stv_budget_visual_cls_embed_head = v_cls_proj.to(dtype=raw_v.dtype).detach()
                        if v_off is not None:
                            astra_config.stv_budget_visual_cls_embed_official = v_off.to(dtype=raw_v.dtype).detach()

                        # Optional: SigLIP-joint-space pooled-token relevance for post-merge semantic recycle.
                        # Keep overhead minimal:
                        # - only compute when post_merge_selection_method actually uses semantic recycle
                        # - reuse the already-loaded SigLIP model + text embedding above
                        pm = str(getattr(astra_config, "post_merge_selection_method", "") or "").strip().lower()
                        need_sem_recycle = ("sem" in pm) or (pm == "semantic_recycle_visual_guided_pruning") or (pm == "di_spm_tsr")
                        if need_sem_recycle:
                            try:
                                # Match LlavaMetaForCausalLM.get_2dPool(image_feat) behavior (default stride=2).
                                stride = 2
                                H = int(getattr(vt, "num_patches_per_side", 0) or 0)
                                if H > 0 and H * H == int(P):
                                    W = H
                                else:
                                    # Fallback: factorize P as a square if possible; otherwise skip.
                                    s = int(math.isqrt(int(P)))
                                    H = W = s if s * s == int(P) else 0
                                if H > 0:
                                    # raw_v -> (T, Dv, H, W)
                                    v_sp = raw_v.float().view(T, H, W, -1).permute(0, 3, 1, 2).contiguous()
                                    mode = str(getattr(self.config, "mm_spatial_pool_mode", "average")).strip().lower()
                                    if mode == "average":
                                        v_sp = nn.functional.avg_pool2d(v_sp, stride)
                                    elif mode == "max":
                                        v_sp = nn.functional.max_pool2d(v_sp, stride)
                                    elif mode == "bilinear":
                                        hh, ww = v_sp.shape[2:]
                                        scaled_shape = [math.ceil(hh / stride), math.ceil(ww / stride)]
                                        v_sp = nn.functional.interpolate(v_sp, size=scaled_shape, mode="bilinear")
                                    else:
                                        v_sp = None
                                    if v_sp is not None:
                                        # (T, Dv, H', W') -> (T, Np', Dv)
                                        v_pool = v_sp.permute(0, 2, 3, 1).contiguous().view(T, -1, v_sp.shape[1])
                                        # Project pooled tokens into SigLIP joint space.
                                        if getattr(siglip_model, "visual_projection", None) is not None:
                                            vj = siglip_model.visual_projection(v_pool)  # (T, Np', De)
                                        else:
                                            vj = v_pool
                                        vj = F.normalize(vj, dim=-1)
                                        # Cosine relevance per pooled token: (T, Np')
                                        rel = (vj * text_cls.view(1, 1, -1)).sum(dim=-1)
                                        astra_config.siglip_pooled_token_relevance = rel.reshape(-1).to(dtype=raw_v.dtype).detach()
                            except Exception:
                                pass
                    else:
                        raise RuntimeError("missing query_text or raw vision features for SigLIP conditioning")
                except Exception:
                    pass

                pooled_image_feature = self.get_2dPool(image_feat)
                pooled_cls_attentions = self.get_2dPool(cls_attentions.unsqueeze(-1)).squeeze(-1)
                compressed_visual_tokens, keep_visual_indices = astra_compression(
                    video_features=pooled_image_feature,
                    cls_attention=pooled_cls_attentions,
                    astra_config=astra_config,
                )
                # Default visual token length for downstream inner-LLM pruning.
                # Some branches (e.g. mm_newline_position="frame") will override it later after appending newline tokens.
                astra_config.visual_token_length = int(compressed_visual_tokens.shape[0])
                image_features.append(compressed_visual_tokens)
            else:
                image_features.append(image_feat)
        # image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
        # rank_print(f"Encoded image feats : {[x.shape for x in image_features]}")
        # image_features = torch.split(image_features, split_sizes, dim=0)
        mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
        image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
        mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")

        if mm_patch_merge_type == "flat":
            # image/video features can be either (N, P, D) or already flattened (L, D).
            # Only flatten the 3D case; flattening 2D (L, D) would corrupt it into 1D.
            image_features = [x.flatten(0, 1) if x.dim() == 3 else x for x in image_features]

        elif mm_patch_merge_type.startswith("spatial"):
            new_image_features = []
            for image_idx, image_feature in enumerate(image_features):
                # FIXME: now assume the image is square, and split to 2x2 patches
                # num_patches = h * w, where h = w = sqrt(num_patches)
                # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                # we want to first unflatten it to (2, 2, h, w, hidden_size)
                # rank0_print("At least we are reaching here")
                # import pdb; pdb.set_trace()
                if image_idx in video_idx_in_batch:  # video operations
                    # rank0_print("Video")
                    if mm_newline_position == "grid":
                        # Grid-wise
                        image_feature = self.add_token_per_grid(image_feature)
                        if getattr(self.config, "add_faster_video", False):
                            faster_video_feature = self.add_token_per_grid(all_faster_video_features[image_idx])
                            # Add a token for each frame
                            concat_slow_fater_token = []
                            # import pdb; pdb.set_trace()
                            for _ in range(image_feature.shape[0]):
                                if _ % self.config.faster_token_stride == 0:
                                    concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                else:
                                    concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                            # import pdb; pdb.set_trace()
                            image_feature = torch.cat(concat_slow_fater_token)

                            # print("!!!!!!!!!!!!")

                        new_image_features.append(image_feature)
                    elif mm_newline_position == "frame":
                        # Frame-wise
                        # image_feature = self.add_token_per_frame(image_feature)
                        # * Append mm_newline_token to each frame
                        compressed_visual_token_list = []
                        num_frames, num_visual_tokens = pooled_image_feature.shape[:2] # (64, 169)
                        for frame_idx in range(num_frames):
                            start_idx = frame_idx * num_visual_tokens
                            end_idx = start_idx + num_visual_tokens
                            ind = torch.where((keep_visual_indices >= start_idx) & (keep_visual_indices < end_idx))[0]
                            frame_visual_tokens = compressed_visual_tokens[ind]
                            frame_visual_tokens = torch.cat((frame_visual_tokens, self.model.image_newline[None].to(image_feature.device)), dim=0)
                            compressed_visual_token_list.append(frame_visual_tokens)
                        image_feature = torch.cat(compressed_visual_token_list, dim=0)
                        astra_config.visual_token_length = image_feature.shape[0] # * Update the visual token length in the config
                        # print(f"visual token length: {astra_config.visual_token_length}") # ? Debugging line
                        # new_image_features.append(image_feature.flatten(0, 1))
                        new_image_features.append(image_feature)

                    elif mm_newline_position == "one_token":
                        # one-token
                        # image_feature = image_feature.flatten(0, 1)
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None].to(image_feature.device)), dim=0)
                        # Ensure visual_token_length is available even when not using frame-wise newline.
                        if image_idx in video_idx_in_batch:
                            astra_config.visual_token_length = int(image_feature.shape[0])
                        new_image_features.append(image_feature)
                    elif mm_newline_position == "no_token":
                        if image_idx in video_idx_in_batch:
                            # Will be flattened below; precompute the final token length here.
                            astra_config.visual_token_length = int(image_feature.flatten(0, 1).shape[0])
                        new_image_features.append(image_feature.flatten(0, 1))
                    else:
                        raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                    # rank0_print("Single-images")
                    base_image_feature = image_feature[0]
                    image_feature = image_feature[1:]
                    height = width = self.get_vision_tower().num_patches_per_side
                    assert height * width == base_image_feature.shape[0]

                    if "anyres_max" in image_aspect_ratio:
                        matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                        if matched_anyres_max_num_patches:
                            max_num_patches = int(matched_anyres_max_num_patches.group(1))

                    if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                        if hasattr(self.get_vision_tower(), "image_size"):
                            vision_tower_image_size = self.get_vision_tower().image_size
                        else:
                            raise ValueError("vision_tower_image_size is not found in the vision tower.")
                        try:
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                        except Exception as e:
                            rank0_print(f"Error: {e}")
                            num_patch_width, num_patch_height = 2, 2
                        image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                    else:
                        image_feature = image_feature.view(2, 2, height, width, -1)

                    if "maxpool2x2" in mm_patch_merge_type:
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = nn.functional.max_pool2d(image_feature, 2)
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                        unit = image_feature.shape[2]
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = unpad_image(image_feature, image_sizes[image_idx])
                        c, h, w = image_feature.shape
                        times = math.sqrt(h * w / (max_num_patches * unit**2))
                        if times > 1.1:
                            image_feature = image_feature[None]
                            image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                        image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    elif "unpad" in mm_patch_merge_type:
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = unpad_image(image_feature, image_sizes[image_idx])
                        image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    else:
                        image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                        image_feature = image_feature.flatten(0, 3)
                    if "nobase" in mm_patch_merge_type:
                        pass
                    else:
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    new_image_features.append(image_feature)
                else:  # single image operations
                    image_feature = image_feature[0]
                    if "unpad" in mm_patch_merge_type:
                        image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                    new_image_features.append(image_feature)
            image_features = new_image_features
        else:
            raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
    else:
        image_features = self.encode_images(images)

    # TODO: image start / end is not implemented here to support pretraining.
    if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
        raise NotImplementedError
    # rank_print(f"Total images : {len(image_features)}")

    # Let's just add dummy tensors if they do not exist,
    # it is a headache to deal with None all the time.
    # But it is not ideal, and if you have a better idea,
    # please open an issue / submit a PR, thanks.
    _labels = labels
    _position_ids = position_ids
    _attention_mask = attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if position_ids is None:
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    # remove the padding using attention_mask -- FIXME
    _input_ids = input_ids
    input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
    labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

    # We'll (re-)derive inner-LLM visual token boundaries from the final embed assembly below.
    # This is more robust than relying on placeholder token indices, and works for different token_strategy settings.
    astra_config: AstraConfig = getattr(self, "astra_config", None)
    video_vis_start_unpadded: int | None = None
    video_vis_len: int = 0

    new_input_embeds = []
    new_labels = []
    cur_image_idx = 0
    # rank_print("Inserting Images embedding")
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        # rank0_print(num_images)
        if num_images == 0:
            cur_image_features = image_features[cur_image_idx]
            cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
            cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
            new_input_embeds.append(cur_input_embeds)
            new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            continue

        image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
        split_sizes = [x.shape[0] for x in cur_labels_noim]
        cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
        cur_new_input_embeds = []
        cur_new_labels = []
        cur_len_so_far = 0

        for i in range(num_images + 1):
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            cur_new_labels.append(cur_labels_noim[i])
            cur_len_so_far += int(cur_input_embeds_no_im[i].shape[0])
            if i < num_images:
                try:
                    cur_image_features = image_features[cur_image_idx]
                except IndexError:
                    cur_image_features = image_features[cur_image_idx - 1]
                cur_image_idx += 1

                # Record visual token boundaries for inner-LLM pruning (LLaVA video path).
                # We assume a single video per batch (already asserted earlier) and store the contiguous inserted block.
                if astra_config is not None and batch_idx == 0:
                    if isinstance(modalities, list) and len(modalities) > batch_idx and modalities[batch_idx] == "video":
                        if video_vis_start_unpadded is None:
                            video_vis_start_unpadded = cur_len_so_far
                        video_vis_len += int(cur_image_features.shape[0])

                cur_new_input_embeds.append(cur_image_features)
                cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                cur_len_so_far += int(cur_image_features.shape[0])

        cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

        # import pdb; pdb.set_trace()
        cur_new_input_embeds = torch.cat(cur_new_input_embeds)
        cur_new_labels = torch.cat(cur_new_labels)

        new_input_embeds.append(cur_new_input_embeds)
        new_labels.append(cur_new_labels)

    # Truncate sequences to max length as image embeddings can make the sequence longer
    tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
    # rank_print("Finishing Inserting")

    new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
    new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
    # TODO: Hard code for control loss spike
    # if tokenizer_model_max_length is not None:
    #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
    #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

    # Combine them
    max_len = max(x.shape[0] for x in new_input_embeds)
    batch_size = len(new_input_embeds)

    new_input_embeds_padded = []
    new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
    # rank0_print("Prepare pos id")

    for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
        cur_len = cur_new_embed.shape[0]
        if getattr(self.config, "tokenizer_padding_side", "right") == "left":
            # Left pad: shift recorded unpadded indices by the pad offset.
            if astra_config is not None and i == 0 and video_vis_start_unpadded is not None and video_vis_len > 0:
                pad_offset = int(max_len - cur_len)
                astra_config.visual_token_start_index = pad_offset + int(video_vis_start_unpadded)
                astra_config.visual_token_length = int(video_vis_len)
            new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
            if cur_len > 0:
                new_labels_padded[i, -cur_len:] = cur_new_labels
                attention_mask[i, -cur_len:] = True
                position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
        else:
            # Right pad: unpadded indices match the padded layout.
            if astra_config is not None and i == 0 and video_vis_start_unpadded is not None and video_vis_len > 0:
                astra_config.visual_token_start_index = int(video_vis_start_unpadded)
                astra_config.visual_token_length = int(video_vis_len)
            new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

    new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
    # rank0_print("tokenizer padding")

    if _labels is None:
        new_labels = None
    else:
        new_labels = new_labels_padded

    if _attention_mask is None:
        attention_mask = None
    else:
        attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    if _position_ids is None:
        position_ids = None
    if getattr(self.config, "use_pos_skipping", False) and self.training:
        position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
        split_position = random.randint(0, new_input_embeds.size(1))
        left_add = random.randint(0, self.config.pos_skipping_range)
        right_add = random.randint(left_add, self.config.pos_skipping_range)
        position_ids[:, :split_position] += left_add
        position_ids[:, split_position:] += right_add
    # import pdb; pdb.set_trace()
    # rank0_print("Finish preparing")
    return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
