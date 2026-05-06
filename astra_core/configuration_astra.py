from typing import Optional
from dataclasses import dataclass, field
import torch


@dataclass
class AstraConfig:
    # Average retention ratio.
    retention_ratio: float = field(default=0.25)

    # 1) Token Selection Method. Defaults to VGP.
    alpha: float = field(default=0.7) # Ratio of VGP tokens.
    token_selection_method: str = field(default="visual_guided_pruning")

    # Post-merge selection method (used with stv_guided_dynamic_budget_allocation dynamic per-frame budgets).
    # Keep separate from token_selection_method (which is used by baseline Astra segment_compression).
    # - "visual_guided_pruning": merged tokens -> visual_guided_pruning keep B tokens
    # - "semantic_recycle_visual_guided_pruning": dominant visual_guided_pruning + recycle on top ceil(T/4) frames (by sem_scores), token-text top-k;
    #   recycle count k_rec ≈ complementary_ratio * target_k
    # - "visual_guided_cluster_pruning": dominant visual_guided_pruning + Astra-style VisualCluster-kNN spatial pooling on remainder (same k_rec budget
    #   as semantic_recycle, from complementary_ratio); no text / sem_scores frame policy.
    # - "visual_guided_contextual_pruning": visual_guided_pruning selects target_k dominants, then contextual tokens merge into them (post_merge_zip_*).
    # - "semantic_recycle_pruning": three-way split on target_k (default parts 7:1:2): visual_guided_pruning dominants, then VisualCluster-kNN on
    #   remainder (+ optional leaves), then semantic_recycle-style semantic recycle on remainder excluding VisualCluster centers when VisualCluster
    #   actually compresses (so semantic is last and avoids duplicating cluster centers).
    # - "di_spm_tsr": unify diversity+importance as one spatial prune-merge (DI-SPM, ContextualMerge-like soft merge into
    #   anchors), then text-guided semantic recall (TSR) from "weakly assigned" donors.
    post_merge_selection_method: str = field(default="visual_guided_pruning")
    # If True, tokens that received at least one temporal merge (anchor counts>1) are never dropped by
    # post-merge pruning (unless fused count > target_k, then subselect among fused by visual_guided_pruning).
    post_merge_protect_fused_anchors: bool = field(default=False)
    # Temporal-merge leaves (merged-away tokens): excluded from post-merge attn/dominant pools, but may join
    # semantic_recycle_visual_guided_pruning recycle by token-text cosine when True.
    post_merge_recycle_temporal_leaves: bool = field(default=True)
    # semantic_recycle recycle: how to pick which local frames get a share of k_rec (default = fixed top ceil(T/4) by sem_scores).
    # - top_quarter: legacy fixed count = ceil(T/4), frames by frame-level sem_scores vs text CLS.
    # - query_mass: softmax(sem/temp) over frames, take fewest frames until cumulative mass >= post_merge_recycle_frame_mass.
    # - query_topk: top post_merge_recycle_num_frames frames by sem_scores (if num_frames<=0, same count as top_quarter).
    # - query_threshold: all frames with sem >= tau * max(sem), clamped by min/max; sorted by sem.
    post_merge_recycle_frame_policy: str = field(default="top_quarter")
    post_merge_recycle_frame_mass: float = field(default=0.85)
    post_merge_recycle_frame_temp: float = field(default=0.10)
    post_merge_recycle_num_frames: int = field(default=0)
    post_merge_recycle_frame_tau: float = field(default=0.55)
    post_merge_recycle_frame_min: int = field(default=1)
    post_merge_recycle_frame_max: int = field(default=0)  # 0 = no cap (use seg_T)
    # Semantic recycle (semantic_recycle discrete pick): optional average_stv-driven tradeoff between #recycle-frames and per-frame picks.
    # Total k_pick unchanged. Only for discrete top-k frame policies (not query_mass / query_threshold).
    # Gates (recycle_rel_gate, default 0.8 on min-max normalized scores):
    # - Fewer frames (high μ_d): block if baseline per-frame top picks already have min normalized token rel < gate.
    # - More frames (low μ_d): block if baseline recycle frames' min normalized frame CLS rel < gate.
    recycle_mu_d_adapt: bool = field(default=False)
    recycle_rel_gate: float = field(default=0.8)
    recycle_mu_d_extra_frames: int = field(default=0)  # 0 => max(1, n0//4) symmetric span around baseline n0
    # Map average_stv to recycle frame count; values < 0 fall back to stv_budget_mu_low / mu_full.
    recycle_mu_d_map_low: float = field(default=-1.0)
    recycle_mu_d_map_full: float = field(default=-1.0)
    # Linear schedule on average_stv (independent of recycle_mu_d_adapt gate logic):
    # fluct <= recycle_stv_cov_fluct_low -> use all seg_T frames for semantic recycle / budget story;
    # fluct >= recycle_stv_cov_fluct_high -> use only baseline policy frame count (e.g. top_quarter -> ~T/4).
    recycle_stv_coverage_schedule: bool = field(default=False)
    recycle_stv_cov_fluct_low: float = field(default=0.05)
    recycle_stv_cov_fluct_high: float = field(default=0.2)
    # True: average_stv (driving recycle_stv_coverage_schedule / recycle_mu_d mapping, etc.) uses adjacent 1-cos of raw_v patch means (consistent with stv_budget_mu_d_raw).
    # False: Uses adjacent_change_mean on the joint representation (after head+visual_projection).
    average_stv_from_raw_visual: bool = field(default=False)
    complementary_ratio: float = field(default=0.30)
    # semantic_recycle_pruning: integer parts for largest-remainder split of target_k (sum need not be 10; ratio 7:1:2 -> 7,1,2).
    post_merge_triple_dom_parts: int = field(default=7)
    post_merge_triple_visual_cluster_parts: int = field(default=1)
    post_merge_triple_sem_parts: int = field(default=2)
    # Optional dynamic split for semantic_recycle_pruning based on frame fluctuation:
    # fluct <= low : 70% dominant, 0% VisualCluster, 30% semantic recycle
    # fluct >= high: 70% dominant, 20% VisualCluster, 10% semantic recycle
    # low~high: linear interpolation.
    post_merge_dynamic_visual_cluster_sem: bool = field(default=False)
    post_merge_dynamic_fluct_low: float = field(default=0.10)
    post_merge_dynamic_fluct_high: float = field(default=0.20)
    post_merge_dynamic_dom_ratio: float = field(default=0.70)
    post_merge_dynamic_visual_cluster_high_ratio: float = field(default=0.20)
    post_merge_dynamic_sem_low_ratio: float = field(default=0.30)
    post_merge_dynamic_sem_high_ratio: float = field(default=0.10)
    # di_spm_tsr: DI-SPM token budget parts vs TSR recall parts (default 8:2).
    post_merge_di_parts: int = field(default=8)
    post_merge_tsr_parts: int = field(default=2)
    # DI-SPM assignment temperature and merge injection (ContextualMerge-like).
    post_merge_di_assign_temp: float = field(default=1.0)
    post_merge_di_inject: float = field(default=0.35)
    # TSR donor pool: take lowest-ratio donors by max assignment weight (0..1). 0 disables filtering.
    post_merge_tsr_donor_low_ratio: float = field(default=0.60)
    # post_merge visual_guided_contextual_pruning: ContextualMerge-style merge weights (visual cosine + optional query term).
    post_merge_zip_lambda_vis: float = field(default=1.0)
    post_merge_zip_lambda_sem: float = field(default=0.3)
    post_merge_zip_inject: float = field(default=0.35)
    post_merge_zip_sem_cross_dom: bool = field(default=True)
    post_merge_zip_softmax_temp: float = field(default=1.0)
    # Optional tracing for post-merge token provenance (debug only).
    # When enabled, semantic_recycle_pruning writes one json record per sample/segment
    # containing VisualCluster center source stats (from temporal leaves vs non-leaves).
    post_merge_trace_enabled: bool = field(default=False)
    post_merge_trace_path: str = field(default="")

    # 2) Tree-based Spatio-Temporal Token Merging.
    temporal_threshold: float = field(default=0.8)
    # Speed-only option: directly average-merge same-position tokens across frames
    # (skips temporal similarity matching / backward-merge + soft-LB). For benchmarking only.
    temporal_direct_same_cell_merge: bool = field(default=False)
    # Skip temporal merge for adjacent frame pair based on their frame-level cosine similarity (in [-1, 1]).
    # - If s < temporal_skip_if_frame_sim_lt: skip this adjacent frame pair (recommended: 0.8)
    # - If s > temporal_skip_if_frame_sim_gt: skip this adjacent frame pair (rarely needed; default disables)
    temporal_skip_if_frame_sim_lt: float = field(default=-1.0)
    temporal_skip_if_frame_sim_gt: float = field(default=1.0)
    # Soft lower-bound for temporal merging (Astra-style safeguard):
    # When enabled in stv_guided_dynamic_budget_allocation path, we adaptively relax temporal merging
    # (by increasing the threshold) to ensure the number of tokens remaining after
    # temporal merge is >= target_k (sum of per-frame budgets).
    temporal_soft_lower_bound: bool = field(default=False)
    temporal_soft_lower_bound_iters: int = field(default=8)
    temporal_debug_stats: bool = field(default=False)
    # If True, temporal backward merge only matches each token to the same HxW grid cell on the previous frame.
    temporal_merge_same_cell_only: bool = field(default=False)
    # If True (grid layout), merge candidates are the full previous frame (global search); ignores dynamic window.
    temporal_merge_prev_frame_global: bool = field(default=False)
    # Spatial layout of merged vision tokens per frame (H*W == num_visual_tokens). Set from
    # video_grid_thw when available; 0 means infer by factorizing N in temporal_backward_merge.
    vision_spatial_h: int = field(default=0)
    vision_spatial_w: int = field(default=0)

    # Dynamic Video Segmentation (DynamicSegment).
    do_segment: bool = field(default=True)
    segment_threshold: float = field(default=0.9)
    min_segment_num: int = field(default=8)
    complementary_segment: bool = field(default=True)

    # Optional: Frame-level selection & budget allocation (disabled by default).
    # When enabled, we first mark "important" frames (e.g., via DPP) and then allocate
    # more per-frame token budget to important vs non-important frames, before running
    # the existing per-segment token selection + STTM pipeline.
    frame_selection_method: str = field(default="none")  # "none" | "dpp" | "stv_frame_gumbel" | "stv_guided_dynamic_budget_allocation"
    frame_top_p: float = field(default=0.0)  # e.g. 0.9; <=0 disables
    # Q-Frame QFS temperature (pi = softmax(I / tau)).
    stv_frame_tau: float = field(default=0.8)
    # Q-Frame 3-way split by rank.
    stv_frame_important_ratio: float = field(default=0.25)
    stv_frame_context_upper_ratio: float = field(default=0.75)
    stv_frame_irrelevant_tokens_per_frame: int = field(default=1)
    important_budget_ratio: float = field(default=0.7)  # important : non-important = 0.7 : 0.3
    # Aggressive mode:
    # - non-important frames keep only a single CLS-proxy token
    # - important frames receive all remaining budget and use divprune in-frame selection
    # - disable temporal/spatial merging
    aggressive_keyframe_mode: bool = field(default=False)
    non_important_tokens_per_frame: int = field(default=1)
    # Optional aggressive variant:
    # merge all non-important-frame tokens into important-frame retained tokens
    # instead of keeping explicit non-important-frame tokens.
    merge_non_important_to_important: bool = field(default=False)
    non_important_merge_beta: float = field(default=0.5)
    # stv_guided_dynamic_budget_allocation: Dynamic per-frame budget = uniform integer base + zero-sum reallocation consistent with semantic recycle frames / k_sem (see _apply_uniform_recycle_story_per_frame_budget).
    # The following parameters are used for stv_budget_weights logging and recycle_stv mapping fallback, etc.
    stv_budget_min_tokens_per_frame: int = field(default=1)
    stv_budget_eps: float = field(default=1e-6)
    stv_budget_temperature: float = field(default=1.0)
    stv_budget_mu_threshold: float = field(default=0.10)
    stv_budget_mu_full: float = field(default=0.60)
    stv_budget_mu_low: float = field(default=0.10)
    stv_budget_uniform_cv_threshold: float = field(default=0.08)

    # ContextualMerge-style (post-pooling) per-frame token pruning (text-agnostic).
    # When frame_selection_method="contextual_merge", we prune vision tokens *after pooling*
    # and keep a per-frame budget split into dominant/contextual parts.
    contextual_merge_retention_ratio: float = field(default=0.10)  # per-frame keep ratio after pooling
    contextual_merge_dom_ctx_ratio_dom: int = field(default=54)  # dominant:contextual = 54:10 (official)
    contextual_merge_dom_ctx_ratio_ctx: int = field(default=10)
    # Optional absolute token counts per frame; if >0, overrides ratio-based budgeting.
    contextual_merge_dominant_tokens_per_frame: int = field(default=0)
    contextual_merge_contextual_tokens_per_frame: int = field(default=0)

    # If True, bypass vision-side compression entirely (keep all pooled vision tokens).
    # Useful for FastV-only / inner-LLM pruning ablations.
    disable_vision_compression: bool = field(default=False)

    # Runtime-only: optional text CLS embedding for conditional frame selection.
    text_cls_embed: Optional[torch.Tensor] = field(default=None, repr=False)
    last_important_frame_count: Optional[int] = field(default=None)
    last_budget_stats: Optional[dict] = field(default=None, repr=False)
    # Average Spatio-Temporal Volatility (average STV): Adjacent frame feature cosine similarity across the entire video -> take (1-cos) then average adjacent pairs. See adjacent_change_mean.
    # Written in stv_guided_dynamic_budget_allocation path; independent of stv_budget_mu_threshold/mu_full (sem_mix).
    average_stv: Optional[float] = field(default=None, repr=False)
    last_recycle_adapt_stats: Optional[dict] = field(default=None, repr=False)

    # Vision-Side Compression params.
    num_visual_guided_pruning_tokens: Optional[int] = field(default=None)
    num_sttm_tokens: Optional[int] = field(default=None)

    # Inner-LLM Compression params.
    visual_token_start_index: Optional[int] = field(default=None)
    visual_token_length: Optional[int] = field(default=None)
    expansion: float = field(default=1.25)
    pruning_layer: int = field(default=20)
    llm_retention_ratio: float = field(default=0.3)
