from torch import nn
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2Model,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLVisionAttention,
    Qwen3VLVisionBlock,
    Qwen3VLVisionModel,
    Qwen3VLModel,
    Qwen3VLTextAttention,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextModel,
)

from llava.model.llava_arch import LlavaMetaForCausalLM
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
from llava.model.multimodal_encoder.siglip_encoder import (
    SigLipAttention,
    SigLipVisionTower,
)

from .configuration_astra import AstraConfig
from .llava_arch import (
    LlavaMetaForCausalLM_encode_images,
    LlavaMetaForCausalLM_prepare_inputs_labels_for_multimodal,
)
from .modeling_qwen2 import (
    Qwen2Attention_forward,
    Qwen2DecoderLayer_forward,
    Qwen2Model_forward,
)

from .modeling_qwen3_vl import (
    Qwen3VLVisionAttention_forward,
    Qwen3VLVisionBlock_forward,
    Qwen3VLVisionModel_forward,
    Qwen3VLModel_forward,
    Qwen3VLTextAttention_forward,
    Qwen3VLTextDecoderLayer_forward,
    Qwen3VLTextModel_forward,
    Qwen3VLModel_get_image_features,
)

from .siglip_encoder import SigLipAttention_forward, SigLipVisionTower_forward


def _astra_to_float(x, name: str) -> float:
    """Coerce model_args values to float (lmms_eval may leave invalid strings as str)."""
    if isinstance(x, bool):
        raise ValueError(f"Astra: {name} must be numeric, got bool {x}")
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip().replace(",", ".")
        try:
            return float(s)
        except ValueError as e:
            raise ValueError(
                f"Astra: {name} must be a float (e.g. 0.25). Got {x!r}. "
                f"If you passed multiple values, use a single number or fix commas in --model_args."
            ) from e
    raise TypeError(f"Astra: {name} must be float-like, got {type(x).__name__}: {x!r}")


def _astra_to_float_or_default(x, name: str, default: float) -> float:
    """Like _astra_to_float, but treat None / empty str as default (lmms_eval may pass '')."""
    if x is None or (isinstance(x, str) and not x.strip()):
        return float(default)
    return _astra_to_float(x, name)


def _astra_to_int(x, name: str) -> int:
    if isinstance(x, bool):
        raise ValueError(f"Astra: {name} must be int, got bool {x}")
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        try:
            return int(s)
        except ValueError:
            try:
                return int(float(s.replace(",", ".")))
            except ValueError as e:
                raise ValueError(f"Astra: {name} must be int, got {x!r}") from e
    raise TypeError(f"Astra: {name} must be int-like, got {type(x).__name__}: {x!r}")


def _astra_to_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return x != 0
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(x)


def astra(
    model: nn.Module,
    retention_ratio: float = 0.25,
    # 1) DynamicSegment params (FIXED)
    do_segment: bool = True,
    segment_threshold: float = 0.9,
    min_segment_num: int = 8,
    complementary_segment: bool = True,
    # 2) VGP and ABTM params
    token_selection_method: str = "visual_guided_pruning",
    alpha: float = 0.7,
    temporal_threshold: float = 0.8,
    temporal_direct_same_cell_merge: bool = False,
    temporal_skip_if_frame_sim_lt: float = -1.0,
    temporal_skip_if_frame_sim_gt: float = 1.0,
    temporal_soft_lower_bound: bool = False,
    temporal_soft_lower_bound_iters: int = 8,
    temporal_debug_stats: bool = False,
    temporal_merge_same_cell_only: bool = False,
    temporal_merge_prev_frame_global: bool = False,
    # 2.5) Post-merge selection (stv_guided_dynamic_budget_allocation path)
    post_merge_selection_method: str = "visual_guided_pruning",
    post_merge_protect_fused_anchors: bool = False,
    post_merge_recycle_temporal_leaves: bool = True,
    post_merge_recycle_frame_policy: str = "top_quarter",
    post_merge_recycle_frame_mass: float = 0.85,
    post_merge_recycle_frame_temp: float = 0.10,
    post_merge_recycle_num_frames: int = 0,
    post_merge_recycle_frame_tau: float = 0.55,
    post_merge_recycle_frame_min: int = 1,
    post_merge_recycle_frame_max: int = 0,
    recycle_mu_d_adapt: bool = False,
    recycle_rel_gate: float = 0.8,
    recycle_mu_d_extra_frames: int = 0,
    recycle_mu_d_map_low: float = -1.0,
    recycle_mu_d_map_full: float = -1.0,
    recycle_stv_coverage_schedule: bool = False,
    recycle_stv_cov_fluct_low: float = 0.05,
    recycle_stv_cov_fluct_high: float = 0.2,
    average_stv_from_raw_visual: bool = False,
    complementary_ratio: float = 0.30,
    post_merge_zip_lambda_vis: float = 1.0,
    post_merge_zip_lambda_sem: float = 0.3,
    post_merge_zip_inject: float = 0.35,
    post_merge_zip_sem_cross_dom: bool = True,
    post_merge_zip_softmax_temp: float = 1.0,
    post_merge_trace_enabled: bool = False,
    post_merge_trace_path: str = "",
    post_merge_triple_dom_parts: int = 7,
    post_merge_triple_visual_cluster_parts: int = 1,
    post_merge_triple_sem_parts: int = 2,
    post_merge_dynamic_visual_cluster_sem: bool = False,
    post_merge_dynamic_fluct_low: float = 0.10,
    post_merge_dynamic_fluct_high: float = 0.20,
    post_merge_dynamic_dom_ratio: float = 0.70,
    post_merge_dynamic_visual_cluster_high_ratio: float = 0.20,
    post_merge_dynamic_sem_low_ratio: float = 0.30,
    post_merge_dynamic_sem_high_ratio: float = 0.10,
    post_merge_di_parts: int = 8,
    post_merge_tsr_parts: int = 2,
    post_merge_di_assign_temp: float = 1.0,
    post_merge_di_inject: float = 0.35,
    post_merge_tsr_donor_low_ratio: float = 0.60,
    # 3) Inner-LLM Compression params
    expansion: float = 1.25,
    pruning_layer: int = 20,
    llm_retention_ratio: float = 0.3,
    # 4) Optional frame-level selection & budget allocation (disabled by default)
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
    # 5) ContextualMerge-style post-pooling pruning (optional)
    contextual_merge_retention_ratio: float = 0.10,
    contextual_merge_dom_ctx_ratio_dom: int = 54,
    contextual_merge_dom_ctx_ratio_ctx: int = 10,
    contextual_merge_dominant_tokens_per_frame: int = 0,
    contextual_merge_contextual_tokens_per_frame: int = 0,
    disable_vision_compression: bool = False,
) -> nn.Module:
    """Apply Astra to the model.

    Args:
        model (nn.Module): The model to apply Astra to.
        retention_ratio (float, optional): The retention ratio. Defaults to 0.25.
        do_segment (bool, optional): Whether to perform dynamic video segmentation. Defaults to True.
        segment_threshold (float, optional): The threshold for dynamic video segmentation. Defaults to 0.9.
        min_segment_num (int, optional): The minimum number of segments. Defaults to 8.
        complementary_segment (bool, optional): Whether to perform complementary segmentation. Defaults to True.
        token_selection_method (str, optional): The method for token selection. Defaults to "visual_guided_pruning".
        alpha (float, optional): The alpha for token selection. Defaults to 0.7.
        temporal_threshold (float, optional): The temporal threshold for token selection. Defaults to 0.8.
        expansion (float, optional): The expansion ratio for inner-LLM compression. Defaults to 1.25.
        pruning_layer (int, optional): The layer to prune. Defaults to 20.
        llm_retention_ratio (float, optional): The retention ratio for inner-LLM compression. Defaults to 0.3.

    Raises:
        NotImplementedError: If the model is not supported.

    Returns:
        nn.Module: The model with Astra applied.
    """

    # Replace with custom methods.
    if type(model) is LlavaQwenForCausalLM:  ## For LLaVA-OneVision or LLaVA-Video
        LlavaMetaForCausalLM.encode_images = LlavaMetaForCausalLM_encode_images
        LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = LlavaMetaForCausalLM_prepare_inputs_labels_for_multimodal
        SigLipAttention.forward = SigLipAttention_forward
        SigLipVisionTower.forward = SigLipVisionTower_forward
        Qwen2Attention.forward = Qwen2Attention_forward
        Qwen2DecoderLayer.forward = Qwen2DecoderLayer_forward
        Qwen2Model.forward = Qwen2Model_forward
    elif type(model) is Qwen3VLForConditionalGeneration:  ## For Qwen3-VL
        Qwen3VLVisionAttention.forward = Qwen3VLVisionAttention_forward
        Qwen3VLVisionBlock.forward = Qwen3VLVisionBlock_forward
        Qwen3VLVisionModel.forward = Qwen3VLVisionModel_forward
        Qwen3VLModel.forward = Qwen3VLModel_forward
        Qwen3VLTextAttention.forward = Qwen3VLTextAttention_forward
        Qwen3VLTextDecoderLayer.forward = Qwen3VLTextDecoderLayer_forward
        Qwen3VLTextModel.forward = Qwen3VLTextModel_forward
        Qwen3VLModel.get_image_features = Qwen3VLModel_get_image_features
    else:
        raise NotImplementedError(f"Astra is not supported for {type(model)} yet.")

    # Normalize types from CLI / lmms_eval (avoids int * str -> repeated string then * float TypeError).
    retention_ratio = _astra_to_float(retention_ratio, "retention_ratio")
    segment_threshold = _astra_to_float(segment_threshold, "segment_threshold")
    min_segment_num = _astra_to_int(min_segment_num, "min_segment_num")
    alpha = _astra_to_float(alpha, "alpha")
    token_selection_method = str(token_selection_method).strip().lower()
    temporal_threshold = _astra_to_float(temporal_threshold, "temporal_threshold")
    temporal_direct_same_cell_merge = _astra_to_bool(temporal_direct_same_cell_merge)
    temporal_skip_if_frame_sim_lt = _astra_to_float(
        temporal_skip_if_frame_sim_lt, "temporal_skip_if_frame_sim_lt"
    )
    temporal_skip_if_frame_sim_gt = _astra_to_float(
        temporal_skip_if_frame_sim_gt, "temporal_skip_if_frame_sim_gt"
    )
    temporal_soft_lower_bound = _astra_to_bool(temporal_soft_lower_bound)
    temporal_soft_lower_bound_iters = _astra_to_int(
        temporal_soft_lower_bound_iters, "temporal_soft_lower_bound_iters"
    )
    temporal_debug_stats = _astra_to_bool(temporal_debug_stats)
    temporal_merge_same_cell_only = _astra_to_bool(temporal_merge_same_cell_only)
    temporal_merge_prev_frame_global = _astra_to_bool(temporal_merge_prev_frame_global)
    if temporal_merge_prev_frame_global and temporal_merge_same_cell_only:
        temporal_merge_same_cell_only = False
    post_merge_selection_method = str(post_merge_selection_method).strip().lower()
    post_merge_protect_fused_anchors = _astra_to_bool(post_merge_protect_fused_anchors)
    post_merge_recycle_temporal_leaves = _astra_to_bool(post_merge_recycle_temporal_leaves)
    post_merge_recycle_frame_policy = str(post_merge_recycle_frame_policy).strip().lower()
    post_merge_recycle_frame_mass = _astra_to_float(post_merge_recycle_frame_mass, "post_merge_recycle_frame_mass")
    post_merge_recycle_frame_temp = _astra_to_float(post_merge_recycle_frame_temp, "post_merge_recycle_frame_temp")
    post_merge_recycle_num_frames = _astra_to_int(post_merge_recycle_num_frames, "post_merge_recycle_num_frames")
    post_merge_recycle_frame_tau = _astra_to_float(post_merge_recycle_frame_tau, "post_merge_recycle_frame_tau")
    post_merge_recycle_frame_min = _astra_to_int(post_merge_recycle_frame_min, "post_merge_recycle_frame_min")
    post_merge_recycle_frame_max = _astra_to_int(post_merge_recycle_frame_max, "post_merge_recycle_frame_max")
    recycle_mu_d_adapt = _astra_to_bool(recycle_mu_d_adapt)
    recycle_rel_gate = _astra_to_float(recycle_rel_gate, "recycle_rel_gate")
    recycle_mu_d_extra_frames = _astra_to_int(recycle_mu_d_extra_frames, "recycle_mu_d_extra_frames")
    recycle_mu_d_map_low = _astra_to_float(recycle_mu_d_map_low, "recycle_mu_d_map_low")
    recycle_mu_d_map_full = _astra_to_float(recycle_mu_d_map_full, "recycle_mu_d_map_full")
    recycle_stv_coverage_schedule = _astra_to_bool(recycle_stv_coverage_schedule)
    recycle_stv_cov_fluct_low = _astra_to_float(recycle_stv_cov_fluct_low, "recycle_stv_cov_fluct_low")
    recycle_stv_cov_fluct_high = _astra_to_float(recycle_stv_cov_fluct_high, "recycle_stv_cov_fluct_high")
    average_stv_from_raw_visual = _astra_to_bool(average_stv_from_raw_visual)
    complementary_ratio = _astra_to_float(complementary_ratio, "complementary_ratio")
    post_merge_zip_lambda_vis = _astra_to_float(post_merge_zip_lambda_vis, "post_merge_zip_lambda_vis")
    post_merge_zip_lambda_sem = _astra_to_float(post_merge_zip_lambda_sem, "post_merge_zip_lambda_sem")
    post_merge_zip_inject = _astra_to_float(post_merge_zip_inject, "post_merge_zip_inject")
    post_merge_zip_sem_cross_dom = _astra_to_bool(post_merge_zip_sem_cross_dom)
    post_merge_zip_softmax_temp = _astra_to_float(post_merge_zip_softmax_temp, "post_merge_zip_softmax_temp")
    post_merge_trace_enabled = _astra_to_bool(post_merge_trace_enabled)
    post_merge_trace_path = str(post_merge_trace_path).strip()
    post_merge_triple_dom_parts = _astra_to_int(post_merge_triple_dom_parts, "post_merge_triple_dom_parts")
    post_merge_triple_visual_cluster_parts = _astra_to_int(post_merge_triple_visual_cluster_parts, "post_merge_triple_visual_cluster_parts")
    post_merge_triple_sem_parts = _astra_to_int(post_merge_triple_sem_parts, "post_merge_triple_sem_parts")
    post_merge_dynamic_visual_cluster_sem = _astra_to_bool(post_merge_dynamic_visual_cluster_sem)
    post_merge_dynamic_fluct_low = _astra_to_float(post_merge_dynamic_fluct_low, "post_merge_dynamic_fluct_low")
    post_merge_dynamic_fluct_high = _astra_to_float(post_merge_dynamic_fluct_high, "post_merge_dynamic_fluct_high")
    post_merge_dynamic_dom_ratio = _astra_to_float(post_merge_dynamic_dom_ratio, "post_merge_dynamic_dom_ratio")
    post_merge_dynamic_visual_cluster_high_ratio = _astra_to_float(post_merge_dynamic_visual_cluster_high_ratio, "post_merge_dynamic_visual_cluster_high_ratio")
    post_merge_dynamic_sem_low_ratio = _astra_to_float(post_merge_dynamic_sem_low_ratio, "post_merge_dynamic_sem_low_ratio")
    post_merge_dynamic_sem_high_ratio = _astra_to_float(post_merge_dynamic_sem_high_ratio, "post_merge_dynamic_sem_high_ratio")
    post_merge_di_parts = _astra_to_int(post_merge_di_parts, "post_merge_di_parts")
    post_merge_tsr_parts = _astra_to_int(post_merge_tsr_parts, "post_merge_tsr_parts")
    post_merge_di_assign_temp = _astra_to_float(post_merge_di_assign_temp, "post_merge_di_assign_temp")
    post_merge_di_inject = _astra_to_float(post_merge_di_inject, "post_merge_di_inject")
    post_merge_tsr_donor_low_ratio = _astra_to_float(post_merge_tsr_donor_low_ratio, "post_merge_tsr_donor_low_ratio")
    expansion = _astra_to_float(expansion, "expansion")
    pruning_layer = _astra_to_int(pruning_layer, "pruning_layer")
    llm_retention_ratio = _astra_to_float(llm_retention_ratio, "llm_retention_ratio")
    frame_selection_method = str(frame_selection_method).strip().lower()
    frame_top_p = _astra_to_float(frame_top_p, "frame_top_p")
    stv_frame_tau = _astra_to_float(stv_frame_tau, "stv_frame_tau")
    stv_frame_important_ratio = _astra_to_float(stv_frame_important_ratio, "stv_frame_important_ratio")
    stv_frame_context_upper_ratio = _astra_to_float(stv_frame_context_upper_ratio, "stv_frame_context_upper_ratio")
    stv_frame_irrelevant_tokens_per_frame = _astra_to_int(
        stv_frame_irrelevant_tokens_per_frame, "stv_frame_irrelevant_tokens_per_frame"
    )
    important_budget_ratio = _astra_to_float(important_budget_ratio, "important_budget_ratio")
    non_important_tokens_per_frame = _astra_to_int(non_important_tokens_per_frame, "non_important_tokens_per_frame")
    non_important_merge_beta = _astra_to_float(non_important_merge_beta, "non_important_merge_beta")
    stv_budget_min_tokens_per_frame = _astra_to_int(
        stv_budget_min_tokens_per_frame, "stv_budget_min_tokens_per_frame"
    )
    stv_budget_eps = _astra_to_float(stv_budget_eps, "stv_budget_eps")
    stv_budget_temperature = _astra_to_float(
        stv_budget_temperature, "stv_budget_temperature"
    )
    stv_budget_mu_threshold = _astra_to_float(stv_budget_mu_threshold, "stv_budget_mu_threshold")
    stv_budget_mu_full = _astra_to_float(stv_budget_mu_full, "stv_budget_mu_full")
    stv_budget_mu_low = _astra_to_float(stv_budget_mu_low, "stv_budget_mu_low")
    stv_budget_uniform_cv_threshold = _astra_to_float_or_default(
        stv_budget_uniform_cv_threshold, "stv_budget_uniform_cv_threshold", 0.08
    )
    contextual_merge_retention_ratio = _astra_to_float_or_default(contextual_merge_retention_ratio, "contextual_merge_retention_ratio", 0.10)
    contextual_merge_dom_ctx_ratio_dom = _astra_to_int(contextual_merge_dom_ctx_ratio_dom, "contextual_merge_dom_ctx_ratio_dom")
    contextual_merge_dom_ctx_ratio_ctx = _astra_to_int(contextual_merge_dom_ctx_ratio_ctx, "contextual_merge_dom_ctx_ratio_ctx")
    contextual_merge_dominant_tokens_per_frame = _astra_to_int(contextual_merge_dominant_tokens_per_frame, "contextual_merge_dominant_tokens_per_frame")
    contextual_merge_contextual_tokens_per_frame = _astra_to_int(contextual_merge_contextual_tokens_per_frame, "contextual_merge_contextual_tokens_per_frame")
    disable_vision_compression = _astra_to_bool(disable_vision_compression)
    do_segment = _astra_to_bool(do_segment)
    complementary_segment = _astra_to_bool(complementary_segment)
    aggressive_keyframe_mode = _astra_to_bool(aggressive_keyframe_mode)
    merge_non_important_to_important = _astra_to_bool(merge_non_important_to_important)

    # Create Astra config.
    astra_config = AstraConfig(
        retention_ratio=retention_ratio,
        do_segment=do_segment,
        segment_threshold=segment_threshold,
        min_segment_num=min_segment_num,
        complementary_segment=complementary_segment,
        alpha=alpha,
        token_selection_method=token_selection_method,
        temporal_threshold=temporal_threshold,
        temporal_direct_same_cell_merge=temporal_direct_same_cell_merge,
        temporal_skip_if_frame_sim_lt=temporal_skip_if_frame_sim_lt,
        temporal_skip_if_frame_sim_gt=temporal_skip_if_frame_sim_gt,
        temporal_soft_lower_bound=temporal_soft_lower_bound,
        temporal_soft_lower_bound_iters=temporal_soft_lower_bound_iters,
        temporal_debug_stats=temporal_debug_stats,
        temporal_merge_same_cell_only=temporal_merge_same_cell_only,
        temporal_merge_prev_frame_global=temporal_merge_prev_frame_global,
        post_merge_selection_method=post_merge_selection_method,
        post_merge_protect_fused_anchors=post_merge_protect_fused_anchors,
        post_merge_recycle_temporal_leaves=post_merge_recycle_temporal_leaves,
        post_merge_recycle_frame_policy=post_merge_recycle_frame_policy,
        post_merge_recycle_frame_mass=post_merge_recycle_frame_mass,
        post_merge_recycle_frame_temp=post_merge_recycle_frame_temp,
        post_merge_recycle_num_frames=post_merge_recycle_num_frames,
        post_merge_recycle_frame_tau=post_merge_recycle_frame_tau,
        post_merge_recycle_frame_min=post_merge_recycle_frame_min,
        post_merge_recycle_frame_max=post_merge_recycle_frame_max,
        recycle_mu_d_adapt=recycle_mu_d_adapt,
        recycle_rel_gate=recycle_rel_gate,
        recycle_mu_d_extra_frames=recycle_mu_d_extra_frames,
        recycle_mu_d_map_low=recycle_mu_d_map_low,
        recycle_mu_d_map_full=recycle_mu_d_map_full,
        recycle_stv_coverage_schedule=recycle_stv_coverage_schedule,
        recycle_stv_cov_fluct_low=recycle_stv_cov_fluct_low,
        recycle_stv_cov_fluct_high=recycle_stv_cov_fluct_high,
        average_stv_from_raw_visual=average_stv_from_raw_visual,
        complementary_ratio=complementary_ratio,
        post_merge_zip_lambda_vis=post_merge_zip_lambda_vis,
        post_merge_zip_lambda_sem=post_merge_zip_lambda_sem,
        post_merge_zip_inject=post_merge_zip_inject,
        post_merge_zip_sem_cross_dom=post_merge_zip_sem_cross_dom,
        post_merge_zip_softmax_temp=post_merge_zip_softmax_temp,
        post_merge_trace_enabled=post_merge_trace_enabled,
        post_merge_trace_path=post_merge_trace_path,
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
        post_merge_di_parts=post_merge_di_parts,
        post_merge_tsr_parts=post_merge_tsr_parts,
        post_merge_di_assign_temp=post_merge_di_assign_temp,
        post_merge_di_inject=post_merge_di_inject,
        post_merge_tsr_donor_low_ratio=post_merge_tsr_donor_low_ratio,
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
        contextual_merge_retention_ratio=contextual_merge_retention_ratio,
        contextual_merge_dom_ctx_ratio_dom=contextual_merge_dom_ctx_ratio_dom,
        contextual_merge_dom_ctx_ratio_ctx=contextual_merge_dom_ctx_ratio_ctx,
        contextual_merge_dominant_tokens_per_frame=contextual_merge_dominant_tokens_per_frame,
        contextual_merge_contextual_tokens_per_frame=contextual_merge_contextual_tokens_per_frame,
        disable_vision_compression=disable_vision_compression,
    )

    # Store Astra Config in the model.
    setattr(model, "astra_config", astra_config)
    setattr(model.model, "astra_config", astra_config)
    if type(model) in (Qwen3VLForConditionalGeneration,):
        setattr(model.model.language_model, "astra_config", astra_config)

    return model
