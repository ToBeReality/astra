import copy
import json
import logging
import os
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from packaging import version
from tqdm import tqdm
from transformers import AutoConfig

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
eval_logger = logging.getLogger("lmms-eval")

# Enable TF32 for CUDA
torch.backends.cuda.matmul.allow_tf32 = True

# Import LLaVA modules
try:
    from llava.constants import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IGNORE_INDEX,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import (
        KeywordsStoppingCriteria,
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from llava.model.builder import load_pretrained_model
except ImportError as e:
    eval_logger.debug(f"LLaVA is not installed. Please install LLaVA to use this model.\nError: {e}")


# Determine best attention implementation
if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


@register_model("llava_onevision")
class Llava_OneVision(lmms):
    """
    Llava Model
    """

    def __init__(
        self,
        pretrained: str = "lmms-lab/llava-onevision-qwen2-7b-ov",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name: Optional[str] = None,
        attn_implementation: Optional[str] = best_fit_attn_implementation,
        device_map: Optional[str] = "cuda:0",
        conv_template: Optional[str] = "qwen_1_5",
        use_cache: Optional[bool] = True,
        truncate_context: Optional[bool] = False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        customized_config: Optional[str] = None,  # ends in json
        max_frames_num: Optional[int] = 32,
        mm_spatial_pool_stride: Optional[int] = 2,
        mm_spatial_pool_mode: Optional[str] = "bilinear",
        token_strategy: Optional[str] = "single",  # could be "single" or "multiple", "multiple" denotes adding multiple <image> tokens for each frame
        video_decode_backend: str = "decord",
        # ! Astra parameters.
        enable_astra: bool = False,
        retention_ratio: float = 0.25,
        # DynamicSegment parameters (FIXED)
        do_segment: bool = True,
        segment_threshold: float = 0.9,
        min_segment_num: int = 8,
        complementary_segment: bool = True,
        # VGP and ABTM parameters
        token_selection_method: str = "visual_guided_pruning",
        alpha: float = 0.7,
        temporal_threshold: float = 0.8,
        # Temporal backward-merge guards / stats (new)
        temporal_skip_if_frame_sim_lt: float = 0.8,
        temporal_skip_if_frame_sim_gt: float = 1.0,
        temporal_soft_lower_bound: bool = True,
        temporal_soft_lower_bound_iters: int = 8,
        temporal_debug_stats: bool = False,
        temporal_direct_same_cell_merge: bool = False,
        temporal_merge_same_cell_only: bool = False,
        temporal_merge_prev_frame_global: bool = False,
        # Post-merge selection (new)
        post_merge_selection_method: str = "semantic_recycle_pruning",
        post_merge_protect_fused_anchors: bool = False,
        post_merge_recycle_temporal_leaves: bool = True,
        recycle_mu_d_adapt: bool = False,
        recycle_rel_gate: float = 0.8,
        recycle_mu_d_extra_frames: int = 0,
        recycle_mu_d_map_low: float = -1.0,
        recycle_mu_d_map_full: float = -1.0,
        recycle_stv_coverage_schedule: bool = False,
        recycle_stv_cov_fluct_low: float = 0.05,
        recycle_stv_cov_fluct_high: float = 0.2,
        average_stv_from_raw_visual: bool = False,
        post_merge_triple_dom_parts: int = 7,
        post_merge_triple_visual_cluster_parts: int = 1,
        post_merge_triple_sem_parts: int = 2,
        post_merge_dynamic_visual_cluster_sem: bool = False,
        post_merge_dynamic_fluct_low: float = 0.145,
        post_merge_dynamic_fluct_high: float = 0.185,
        post_merge_dynamic_dom_ratio: float = 0.70,
        post_merge_dynamic_visual_cluster_high_ratio: float = 0.15,
        post_merge_dynamic_sem_low_ratio: float = 0.30,
        post_merge_dynamic_sem_high_ratio: float = 0.15,
        complementary_ratio: float = 0.30,
        # Inner-LLM Pruning parameters (FIXED)
        expansion: float = 1.25,
        pruning_layer: int = 20,
        llm_retention_ratio: float = 0.3,
        # Optional frame-level selection (dpp / stv_frame_gumbel legacy) + stv_guided_dynamic_budget_allocation 动态逐帧预算
        frame_selection_method: str = "none",
        frame_top_p: float = 0.0,
        stv_frame_tau: float = 0.8,
        stv_frame_important_ratio: float = 0.25,
        stv_frame_context_upper_ratio: float = 0.75,
        stv_frame_irrelevant_tokens_per_frame: int = 1,
        important_budget_ratio: float = 0.7,
        aggressive_keyframe_mode: bool = False,
        non_important_tokens_per_frame: int = 1,
        merge_non_important_to_important: bool = False,
        non_important_merge_beta: float = 0.5,
        stv_budget_min_tokens_per_frame: int = 1,
        stv_budget_eps: float = 1e-6,
        stv_budget_temperature: float = 1.0,
        stv_budget_mu_threshold: float = 0.10,
        stv_budget_mu_full: float = 0.60,
        stv_budget_mu_low: float = 0.10,
        stv_budget_uniform_cv_threshold: float = 0.08,
        # ContextualMerge (post-pooling) params
        contextual_merge_retention_ratio: float = 0.10,
        contextual_merge_dom_ctx_ratio_dom: int = 54,
        contextual_merge_dom_ctx_ratio_ctx: int = 10,
        contextual_merge_dominant_tokens_per_frame: int = 0,
        contextual_merge_contextual_tokens_per_frame: int = 0,
        # FastV-only / ablation: bypass vision-side compression
        disable_vision_compression: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        llava_model_args = {
            "multimodal": True,
        }
        if customized_config is not None:
            llava_model_args["customized_config"] = customized_config
        if attn_implementation is not None:
            llava_model_args["attn_implementation"] = attn_implementation
        if "use_flash_attention_2" in kwargs:
            llava_model_args["use_flash_attention_2"] = kwargs["use_flash_attention_2"]
        model_name = model_name if model_name is not None else get_model_name_from_path(pretrained)

        self.pretrained = pretrained
        self.token_strategy = token_strategy
        self.max_frames_num = max_frames_num
        self.mm_spatial_pool_stride = mm_spatial_pool_stride
        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.video_decode_backend = video_decode_backend

        overwrite_config = {}
        overwrite_config["mm_spatial_pool_stride"] = self.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_mode"] = self.mm_spatial_pool_mode
        cfg_pretrained = AutoConfig.from_pretrained(self.pretrained)

        llava_model_args["overwrite_config"] = overwrite_config
        try:
            # Try to load the model with the multimodal argument
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
                pretrained,
                None,
                model_name,
                device_map=self.device_map,
                **llava_model_args,
            )
        except TypeError:
            # for older versions of LLaVA that don't have multimodal argument
            llava_model_args.pop("multimodal", None)
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
                pretrained,
                None,
                model_name,
                device_map=self.device_map,
                **llava_model_args,
            )

        # ! Enable Astra
        if enable_astra:
            from astra_core import astra

            self._model = astra(
                model=self._model,
                retention_ratio=retention_ratio,
                do_segment=do_segment,
                segment_threshold=segment_threshold,
                min_segment_num=min_segment_num,
                complementary_segment=complementary_segment,
                alpha=alpha,
                token_selection_method=token_selection_method,
                temporal_threshold=temporal_threshold,
                temporal_skip_if_frame_sim_lt=temporal_skip_if_frame_sim_lt,
                temporal_skip_if_frame_sim_gt=temporal_skip_if_frame_sim_gt,
                temporal_soft_lower_bound=temporal_soft_lower_bound,
                temporal_soft_lower_bound_iters=temporal_soft_lower_bound_iters,
                temporal_debug_stats=temporal_debug_stats,
                temporal_direct_same_cell_merge=temporal_direct_same_cell_merge,
                temporal_merge_same_cell_only=temporal_merge_same_cell_only,
                temporal_merge_prev_frame_global=temporal_merge_prev_frame_global,
                disable_vision_compression=disable_vision_compression,
                contextual_merge_retention_ratio=contextual_merge_retention_ratio,
                contextual_merge_dom_ctx_ratio_dom=contextual_merge_dom_ctx_ratio_dom,
                contextual_merge_dom_ctx_ratio_ctx=contextual_merge_dom_ctx_ratio_ctx,
                contextual_merge_dominant_tokens_per_frame=contextual_merge_dominant_tokens_per_frame,
                contextual_merge_contextual_tokens_per_frame=contextual_merge_contextual_tokens_per_frame,
                post_merge_selection_method=post_merge_selection_method,
                post_merge_protect_fused_anchors=post_merge_protect_fused_anchors,
                post_merge_recycle_temporal_leaves=post_merge_recycle_temporal_leaves,
                recycle_mu_d_adapt=recycle_mu_d_adapt,
                recycle_rel_gate=recycle_rel_gate,
                recycle_mu_d_extra_frames=recycle_mu_d_extra_frames,
                recycle_mu_d_map_low=recycle_mu_d_map_low,
                recycle_mu_d_map_full=recycle_mu_d_map_full,
                recycle_stv_coverage_schedule=recycle_stv_coverage_schedule,
                recycle_stv_cov_fluct_low=recycle_stv_cov_fluct_low,
                recycle_stv_cov_fluct_high=recycle_stv_cov_fluct_high,
                average_stv_from_raw_visual=average_stv_from_raw_visual,
                post_merge_triple_dom_parts=post_merge_triple_dom_parts,
                post_merge_triple_visual_cluster_parts=post_merge_triple_visual_cluster_parts,
                post_merge_triple_sem_parts=post_merge_triple_sem_parts,
                post_merge_dynamic_visual_cluster_sem=post_merge_dynamic_visual_cluster_sem,
                post_merge_dynamic_fluct_low=post_merge_dynamic_fluct_low,
                post_merge_dynamic_fluct_high=post_merge_dynamic_fluct_high,
                post_merge_dynamic_dom_ratio=post_merge_dynamic_dom_ratio,
                post_merge_dynamic_visual_cluster_high_ratio=post_merge_dynamic_visual_cluster_high_ratio,
                post_merge_dynamic_sem_low_ratio=post_merge_dynamic_sem_low_ratio,
                post_merge_dynamic_sem_high_ratio=post_merge_dynamic_sem_high_ratio,
                complementary_ratio=complementary_ratio,
                expansion=expansion,
                pruning_layer=pruning_layer,
                llm_retention_ratio=llm_retention_ratio,
                frame_selection_method=frame_selection_method,
                frame_top_p=frame_top_p,
                stv_frame_tau=stv_frame_tau,
                stv_frame_important_ratio=stv_frame_important_ratio,
                stv_frame_context_upper_ratio=stv_frame_context_upper_ratio,
                stv_frame_irrelevant_tokens_per_frame=stv_frame_irrelevant_tokens_per_frame,
                important_budget_ratio=important_budget_ratio,
                aggressive_keyframe_mode=aggressive_keyframe_mode,
                non_important_tokens_per_frame=non_important_tokens_per_frame,
                merge_non_important_to_important=merge_non_important_to_important,
                non_important_merge_beta=non_important_merge_beta,
                stv_budget_min_tokens_per_frame=stv_budget_min_tokens_per_frame,
                stv_budget_eps=stv_budget_eps,
                stv_budget_temperature=stv_budget_temperature,
                stv_budget_mu_threshold=stv_budget_mu_threshold,
                stv_budget_mu_full=stv_budget_mu_full,
                stv_budget_mu_low=stv_budget_mu_low,
                stv_budget_uniform_cv_threshold=stv_budget_uniform_cv_threshold,
            )
            # print(f"[INFO] Enable Astra with retention_ratio={retention_ratio}, expansion={expansion}, do_segment={do_segment}, segment_threshold={segment_threshold}, min_segment_num={min_segment_num}, complementary_segment={complementary_segment}, token_selection_method={token_selection_method}, alpha={alpha}, temporal_threshold={temporal_threshold}, pruning_layer={pruning_layer}, llm_retention_ratio={llm_retention_ratio}")

        self._config = self._model.config
        self.model.eval()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._world_size = 1

        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        origin_image_aspect_ratio = getattr(self._config, "image_aspect_ratio", None)

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visual = doc_to_visual(self.task_dict[task][split][doc_id])

            if origin_image_aspect_ratio is not None and self._config.image_aspect_ratio != origin_image_aspect_ratio:
                self._config.image_aspect_ratio = origin_image_aspect_ratio
                eval_logger.info(f"Resetting image aspect ratio to {origin_image_aspect_ratio}")

            if visual is None or visual == []:
                visual = None
                task_type = "text"
                image_tensor = None
            else:
                if len(visual) > 1 or "image_aspect_ratio" not in self._config.__dict__:
                    self._config.image_aspect_ratio = "pad"
                    eval_logger.info(f"In Multi-Image setting, image aspect ratio: {self._config.image_aspect_ratio}")

                if "task_type" in self.metadata and self.metadata["task_type"] == "video" and "sample_frames" in self.metadata:
                    assert type(visual) == list, "sample_frames must be specified for video task"
                    sample_indices = np.linspace(0, len(visual) - 1, self.metadata["sample_frames"], dtype=int)
                    visual = [visual[i] for i in sample_indices]
                    assert len(visual) == self.metadata["sample_frames"]

                    image_tensor = process_images(visual, self._image_processor, self._config)
                    if type(image_tensor) is list:
                        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                    else:
                        image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

                    task_type = "video"

                # elif type(visual[0]) == PIL.Image.Image:
                elif isinstance(visual[0], PIL.Image.Image):
                    image_tensor = process_images(visual, self._image_processor, self._config)
                    if type(image_tensor) is list:
                        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                    else:
                        image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

                    task_type = "image"

                elif type(visual[0]) == str:
                    image_tensor = []
                    try:
                        if self.video_decode_backend == "decord":
                            frames = self.load_video(visual, self.max_frames_num)
                        elif self.video_decode_backend == "pyav":
                            frames = read_video_pyav(visual[0], num_frm=self.max_frames_num)
                        frames = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().to(self._device)
                        image_tensor.append(frames)
                    except Exception as e:
                        eval_logger.error(f"Error {e} in loading video")
                        image_tensor = None

                    task_type = "video"

            if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in contexts:
                placeholder_count = len(visual) if isinstance(visual, list) else 1
                if task_type == "video":
                    placeholder_count = len(frames) if self.token_strategy == "multiple" else 1
                image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                image_tokens = " ".join(image_tokens)
                prompts_input = image_tokens + "\n" + contexts
            else:
                prompts_input = contexts

            if "llama_3" in self.conv_template:
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[self.conv_template].copy()

            conv.append_message(conv.roles[0], prompts_input)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])

            conv.messages[-1][1] = continuation
            full_prompt = conv.get_prompt()
            full_input_ids = tokenizer_image_token(full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

            labels = full_input_ids.clone()
            labels[0, : input_ids.shape[1]] = -100

            kwargs = {}
            if task_type == "image":
                kwargs["image_sizes"] = [[v.size[0], v.size[1]] for v in visual] if isinstance(visual, list) else [[visual.size[0], visual.size[1]]]
            elif task_type == "video":
                kwargs["modalities"] = ["video"]
                self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
                self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

            with torch.inference_mode():
                outputs = self.model(
                    input_ids=full_input_ids,
                    labels=labels,
                    images=image_tensor,
                    use_cache=True,
                    **kwargs,
                )

            loss = outputs["loss"]
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = full_input_ids[:, input_ids.shape[1] :]
            greedy_tokens = greedy_tokens[:, input_ids.shape[1] : full_input_ids.shape[1]]
            max_equal = (greedy_tokens == cont_toks).all()

            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)

        pbar.close()
        return res

    def flatten(self, input):
        if not input or any(i is None for i in input):
            return []
        new_list = []
        for i in input:
            if i:
                for j in i:
                    new_list.append(j)
        return new_list

    def load_video(self, video_path, max_frames_num):
        if type(video_path) == str:
            num_threads = int(os.environ.get("DECORD_NUM_THREADS", "1"))
            num_threads = max(1, min(num_threads, 32))
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=num_threads)
        else:
            num_threads = int(os.environ.get("DECORD_NUM_THREADS", "1"))
            num_threads = max(1, min(num_threads, 32))
            vr = VideoReader(video_path[0], ctx=cpu(0), num_threads=num_threads)
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames  # (frames, height, width, channels)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        budget_stats_res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        metadata = requests[0].metadata
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        origin_image_aspect_ratio = getattr(self._config, "image_aspect_ratio", None)

        for chunk in chunks:
            (
                batched_contexts,
                all_gen_kwargs,
                batched_doc_to_visual,
                batched_doc_id,
                batched_task,
                batched_split,
            ) = zip(*chunk)
            task = batched_task[0]
            split = batched_split[0]
            batched_visuals = [batched_doc_to_visual[0](self.task_dict[task][split][ids]) for ids in batched_doc_id]  # [B, N]
            assert len(batched_visuals) == 1

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            question_input = []
            # import ipdb; ipdb.set_trace()
            for visual, context in zip(batched_visuals, batched_contexts):
                if origin_image_aspect_ratio is not None and self._config.image_aspect_ratio != origin_image_aspect_ratio:
                    self._config.image_aspect_ratio = origin_image_aspect_ratio
                    eval_logger.info(f"Resetting image aspect ratio to {origin_image_aspect_ratio}")

                if visual is None or visual == []:  # for text-only tasks.
                    visual = None
                    task_type = "text"
                    placeholder_count = 0
                    image_tensor = None
                else:
                    if len(visual) > 1 or "image_aspect_ratio" not in self._config.__dict__:  # for multi image case, we treat per image aspect ratio as "pad" by default.
                        self._config.image_aspect_ratio = getattr(gen_kwargs, "image_aspect_ratio", "pad")
                        eval_logger.info(f"In Multi-Image setting, image aspect ratio: {self._config.image_aspect_ratio}")

                    if "task_type" in metadata and metadata["task_type"] == "video" and "sample_frames" in metadata:  # overwrite logic for video task with multiple static image frames
                        assert type(visual) == list, "sample_frames must be specified for video task"
                        sample_indices = np.linspace(0, len(visual) - 1, metadata["sample_frames"], dtype=int)
                        visual = [visual[i] for i in sample_indices]
                        assert len(visual) == metadata["sample_frames"]

                        image_tensor = process_images(visual, self._image_processor, self._config)
                        if type(image_tensor) is list:
                            image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                        else:
                            image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

                        task_type = "video"
                        placeholder_count = 1

                    elif type(visual[0]) == PIL.Image.Image:  # For image, multi-image tasks
                        image_tensor = process_images(visual, self._image_processor, self._config)
                        if type(image_tensor) is list:
                            image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                        else:
                            image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

                        task_type = "image"
                        placeholder_count = len(visual) if isinstance(visual, list) else 1

                    elif type(visual[0]) == str:  # For video task
                        image_tensor = []
                        try:
                            if self.video_decode_backend == "decord":
                                frames = self.load_video(visual, self.max_frames_num)
                            elif self.video_decode_backend == "pyav":
                                frames = read_video_pyav(visual[0], num_frm=self.max_frames_num)
                            frames = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().to(self._device)
                            image_tensor.append(frames)
                        except Exception as e:
                            eval_logger.error(f"Error {e} in loading video")
                            image_tensor = None

                        task_type = "video"
                        placeholder_count = len(frames) if self.token_strategy == "multiple" else 1

                if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                    """
                    Three senarios:
                    1. No image, and there for, no image token should be added.
                    2. image token is already specified in the context, so we don't need to add it.
                    3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                    4. For video tasks, we could add a <image> token or multiple <image> tokens for each frame in the context. This depends on the training strategy and should balance in test to decide which is better
                    """
                    # if task_type == "image": # indeed in multi-image case, not the video in frames.
                    #     image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                    # elif task_type == "video":
                    # image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count if self.token_strategy == "multiple" else [DEFAULT_IMAGE_TOKEN]
                    image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                    image_tokens = " ".join(image_tokens)
                    question = image_tokens + "\n" + context
                else:
                    question = context

                # This is much safer for llama3, as we now have some object type in it
                if "llama_3" in self.conv_template:
                    conv = copy.deepcopy(conv_templates[self.conv_template])
                else:
                    conv = conv_templates[self.conv_template].copy()

                if utils.is_json(question):  # conversational question input
                    question = json.loads(question)
                    for idx, item in enumerate(question):
                        role = conv.roles[idx % 2]
                        message = item["value"]
                        conv.append_message(role, message)

                    assert len(conv.messages) % 2 == 1
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)
                else:  # only simple string for question
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)

            input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)

            if task_type == "image":
                gen_kwargs["image_sizes"] = [batched_visuals[0][idx].size for idx in range(len(batched_visuals[0]))]
            elif task_type == "video":
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
                gen_kwargs["modalities"] = ["video"]
                gen_kwargs["stopping_criteria"] = [stopping_criteria]
                self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
                self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

            # These steps are not in LLaVA's original code, but are necessary for generation to work
            # TODO: attention to this major generation step...
            # preconfigure gen_kwargs with defaults
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024

            if "image_aspect_ratio" in gen_kwargs.keys():
                gen_kwargs.pop("image_aspect_ratio")
            # When do_sample=False, remove sampling-related parameters to avoid warnings
            # These might be in gen_kwargs or in the model's generation_config
            if not gen_kwargs.get("do_sample", False):
                gen_kwargs.pop("temperature", None)
                gen_kwargs.pop("top_p", None)
                gen_kwargs.pop("top_k", None)
            try:
                with torch.inference_mode():
                    # Provide raw query text for SigLIP text encoder conditioning (Astra stv_frame).
                    try:
                        fv_cfg = getattr(self.model, "astra_config", None)
                        if fv_cfg is not None:
                            # Use the pre-image-token context (original question) to avoid special tokens.
                            fv_cfg.query_text = str(batched_contexts[0])
                    except Exception:
                        pass
                    cont = self.model.generate(
                        input_ids,
                        attention_mask=attention_masks,
                        pad_token_id=pad_token_ids,
                        images=image_tensor,
                        use_cache=self.use_cache,
                        **gen_kwargs,
                    )
                    # cont = self.model.generate(qwen_input_ids, pad_token_id=pad_token_ids, images=image_tensor, use_cache=self.use_cache, **gen_kwargs)

                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            except Exception as e:
                raise e

            # Capture Astra stv_frame budget stats for this sample (if enabled).
            # Note: batches are effectively size 1 for Llava (see assertion in __init__), but we still
            # align stats with outputs to be safe.
            stats = None
            try:
                fv_cfg = getattr(self.model, "astra_config", None)
                if fv_cfg is not None:
                    stats = getattr(fv_cfg, "last_budget_stats", None)
            except Exception:
                stats = None

            text_outputs = [response.strip() for response in text_outputs]
            res.extend(text_outputs)
            # One stats record per output in the batch.
            for _ in range(len(text_outputs)):
                budget_stats_res.append(stats)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)
        budget_stats_res = re_ords.get_original(budget_stats_res)

        # Attach per-request extra stats for --log_samples (kept out of filtering logic).
        for req, st in zip(requests, budget_stats_res):
            if st is None:
                continue
            if not hasattr(req, "extra") or req.extra is None:
                req.extra = {}
            req.extra["astra_budget_stats"] = st
            # Optional lightweight stdout logging for debugging budget sharpness.
            # Enable via: ASTRA_BUDGET_STDOUT=1 (rank0 only).
            try:
                import os, json

                if os.environ.get("ASTRA_BUDGET_STDOUT", "").strip() in ("1", "true", "True"):
                    if str(os.environ.get("RANK", "0")) == "0":
                        msg = {
                            "task": getattr(req, "task_name", None),
                            "mu_d": st.get("mu_d"),
                            "sem_mix": st.get("sem_mix"),
                            "budget_min": st.get("min"),
                            "budget_max": st.get("max"),
                            "budget_cv": st.get("cv"),
                            "qlogits_std": st.get("qlogits_std"),
                            "vcls_qlogits_std": st.get("vcls_qlogits_std"),
                            "regime": st.get("regime"),
                        }
                        print("[astra_budget_stats] " + json.dumps(msg, ensure_ascii=False))
            except Exception:
                pass

        pbar.close()
        return res

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        metadata = requests[0].metadata
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        origin_image_aspect_ratio = getattr(self._config, "image_aspect_ratio", None)

        for chunk in chunks:
            (
                batched_contexts,
                all_gen_kwargs,
                batched_doc_to_visual,
                batched_doc_to_text,
                batched_doc_id,
                batched_task,
                batched_split,
            ) = zip(*chunk)
            task = batched_task[0]
            split = batched_split[0]
            batched_visuals = [batched_doc_to_visual[0](self.task_dict[task][split][ids]) for ids in batched_doc_id]  # [B, N]
            assert len(batched_visuals) == 1

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            # multi round inference: terminate when receiving signal from the doc_to_text
            round_idx = 0
            batched_round_res = []
            batched_previous_round_info = None
            while True:
                question_input = []

                if round_idx != 0:  # get current round visual and context from doc_to_text function
                    (
                        batched_visuals,
                        batched_contexts,
                        batched_terminal_singal,
                        batched_round_res,
                        batched_previous_round_info,
                    ) = list(
                        zip(
                            *[
                                batched_doc_to_text[0](
                                    self.task_dict[task][split][ids],
                                    previous_output=[round_res[ids_idx] for round_res in batched_round_res],
                                    round_idx=round_idx,
                                    previous_round_info=batched_previous_round_info[ids_idx] if batched_previous_round_info is not None else None,
                                )
                                for ids_idx, ids in enumerate(batched_doc_id)
                            ]
                        )
                    )
                    # import ipdb; ipdb.set_trace()
                    batched_round_res = list(zip(*batched_round_res))  # [(r1_1, r1_2), (r2_1, r2_2), ...]
                    if batched_terminal_singal[0]:  # terminal signal from doc_to_text function
                        break

                for visual, context in zip(batched_visuals, batched_contexts):
                    if origin_image_aspect_ratio is not None and self._config.image_aspect_ratio != origin_image_aspect_ratio:
                        self._config.image_aspect_ratio = origin_image_aspect_ratio
                        eval_logger.info(f"Resetting image aspect ratio to {origin_image_aspect_ratio}")

                    if visual is None or visual == []:  # for text-only tasks.
                        visual = None
                        task_type = "text"
                        placeholder_count = 0
                        image_tensor = None
                    else:
                        if len(visual) > 1 or "image_aspect_ratio" not in self._config.__dict__:  # for multi image case, we treat per image aspect ratio as "pad" by default.
                            self._config.image_aspect_ratio = getattr(gen_kwargs, "image_aspect_ratio", "pad")
                            eval_logger.info(f"In Multi-Image setting, image aspect ratio: {self._config.image_aspect_ratio}")

                        if "task_type" in metadata and metadata["task_type"] == "video" and "sample_frames" in metadata:  # overwrite logic for video task with multiple static image frames
                            assert type(visual) == list, "sample_frames must be specified for video task"
                            sample_indices = np.linspace(0, len(visual) - 1, metadata["sample_frames"], dtype=int)
                            visual = [visual[i] for i in sample_indices]
                            assert len(visual) == metadata["sample_frames"]

                            image_tensor = process_images(visual, self._image_processor, self._config)
                            if type(image_tensor) is list:
                                image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                            else:
                                image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

                            task_type = "video"
                            placeholder_count = 1

                        elif type(visual[0]) == PIL.Image.Image:  # For image, multi-image tasks
                            image_tensor = process_images(visual, self._image_processor, self._config)
                            if type(image_tensor) is list:
                                image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                            else:
                                image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

                            task_type = "image"
                            placeholder_count = len(visual) if isinstance(visual, list) else 1

                        elif type(visual[0]) == str:  # For video task
                            image_tensor = []
                            try:
                                if self.video_decode_backend == "decord":
                                    frames = self.load_video(visual, self.max_frames_num)
                                elif self.video_decode_backend == "pyav":
                                    frames = read_video_pyav(visual[0], num_frm=self.max_frames_num)
                                frames = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().to(self._device)
                                image_tensor.append(frames)
                            except Exception as e:
                                eval_logger.error(f"Error {e} in loading video")
                                image_tensor = None

                            task_type = "video"
                            placeholder_count = len(frames) if self.token_strategy == "multiple" else 1

                    if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                        """
                        Three senarios:
                        1. No image, and there for, no image token should be added.
                        2. image token is already specified in the context, so we don't need to add it.
                        3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                        4. For video tasks, we could add a <image> token or multiple <image> tokens for each frame in the context. This depends on the training strategy and should balance in test to decide which is better
                        """
                        # if task_type == "image": # indeed in multi-image case, not the video in frames.
                        #     image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                        # elif task_type == "video":
                        # image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count if self.token_strategy == "multiple" else [DEFAULT_IMAGE_TOKEN]
                        image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                        image_tokens = " ".join(image_tokens)
                        question = image_tokens + "\n" + context
                    else:
                        question = context

                    # This is much safer for llama3, as we now have some object type in it
                    if "llama_3" in self.conv_template:
                        conv = copy.deepcopy(conv_templates[self.conv_template])
                    else:
                        conv = conv_templates[self.conv_template].copy()

                    if utils.is_json(question):  # conversational question input
                        question = json.loads(question)
                        for idx, item in enumerate(question):
                            role = conv.roles[idx % 2]
                            message = item["value"]
                            conv.append_message(role, message)

                        assert len(conv.messages) % 2 == 1
                        conv.append_message(conv.roles[1], None)
                        prompt_question = conv.get_prompt()
                        question_input.append(prompt_question)
                    else:  # only simple string for question
                        conv.append_message(conv.roles[0], question)
                        conv.append_message(conv.roles[1], None)
                        prompt_question = conv.get_prompt()
                        question_input.append(prompt_question)

                # preconfigure gen_kwargs with defaults
                if "max_new_tokens" not in gen_kwargs:
                    gen_kwargs["max_new_tokens"] = 1024
                if "do_sample" not in gen_kwargs:
                    gen_kwargs["do_sample"] = False
                # Only set temperature and top_p if do_sample is True
                if gen_kwargs.get("do_sample", False):
                    if "temperature" not in gen_kwargs:
                        gen_kwargs["temperature"] = 1.0  # Default temperature for sampling
                    if "top_p" not in gen_kwargs:
                        gen_kwargs["top_p"] = 1.0  # Default top_p for sampling
                if "num_beams" not in gen_kwargs:
                    gen_kwargs["num_beams"] = 1

                input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
                pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
                attention_masks = input_ids.ne(pad_token_ids).to(self.device)

                if task_type == "image":
                    gen_kwargs["image_sizes"] = [batched_visuals[0][idx].size for idx in range(len(batched_visuals[0]))]
                elif task_type == "video":
                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    keywords = [stop_str]
                    stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
                    gen_kwargs["modalities"] = ["video"]
                    gen_kwargs["stopping_criteria"] = [stopping_criteria]
                    self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
                    self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

                # These steps are not in LLaVA's original code, but are necessary for generation to work
                # TODO: attention to this major generation step...
                if "image_aspect_ratio" in gen_kwargs.keys():
                    gen_kwargs.pop("image_aspect_ratio")
                # Remove temperature and top_p when do_sample=False to avoid warnings
                if not gen_kwargs.get("do_sample", False):
                    gen_kwargs.pop("temperature", None)
                    gen_kwargs.pop("top_p", None)
                try:
                    with torch.inference_mode():
                        # Provide raw query text for SigLIP text encoder conditioning (Astra stv_frame).
                        try:
                            fv_cfg = getattr(self.model, "astra_config", None)
                            if fv_cfg is not None:
                                fv_cfg.query_text = str(batched_contexts[0])
                        except Exception:
                            pass
                        cont = self.model.generate(
                            input_ids,
                            attention_mask=attention_masks,
                            pad_token_id=pad_token_ids,
                            images=image_tensor,
                            use_cache=self.use_cache,
                            **gen_kwargs,
                        )
                        # cont = self.model.generate(qwen_input_ids, pad_token_id=pad_token_ids, images=image_tensor, use_cache=self.use_cache, **gen_kwargs)

                    text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
                except Exception as e:
                    raise e

                text_outputs = [response.strip() for response in text_outputs]
                batched_round_res.append(text_outputs)

                round_idx += 1

            res.extend(list(zip(*batched_round_res)))
            self.cache_hook.add_partial("generate_until_multi_round", (context, gen_kwargs), batched_round_res)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res
