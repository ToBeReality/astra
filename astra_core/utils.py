from typing import Optional, Tuple, List

import json
import math
import os
import torch
from torch.nn import functional as F
from .configuration_astra import AstraConfig
from .token_selection import (
    attn_based_token_selection,
    visual_guided_pruning_based_token_selection,
    div_based_token_selection,
    dpp_based_token_selection,
    TokenSelectionMethod,
)
from .frame_selection import (
    adjacent_change_mean,
    build_frame_features_and_quality,
    dpp_select_frames_top_p,
    stv_frame_gumbel_select_frames,
    stv_budget_weights,
    stv_frame_rank_to_three_classes,
    text_relevance_logits,
    text_relevance_minmax,
)
from .token_selection import ALL_TOKEN_SELECTION_METHOD as TOKEN_SELECTION_BY_NAME

def _is_stv_guided_dynamic_budget_allocation(method: Optional[str]) -> bool:
    """动态逐帧预算与语义回收对齐（整段单 segment）；兼容旧名 stv_frame_soft_budget。"""
    if method is None:
        return False
    m = str(method).strip().lower()
    return m in ("stv_guided_dynamic_budget_allocation", "stv_frame_soft_budget")


ALL_TOKEN_SELECTION_METHOD = {
    TokenSelectionMethod.ATTN: attn_based_token_selection,
    TokenSelectionMethod.VGP: visual_guided_pruning_based_token_selection,
    TokenSelectionMethod.DIV: div_based_token_selection,
    TokenSelectionMethod.DIVPRUNE: div_based_token_selection,
    TokenSelectionMethod.DPP: dpp_based_token_selection,
}


def _post_merge_visual_guided_pruning_select(
    *,
    astra_config: AstraConfig,
    features: torch.Tensor,
    cls_attention: torch.Tensor,
    num_retained_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Post-merge dominant selection using the single VGP implementation."""
    return visual_guided_pruning_based_token_selection(
        features=features,
        cls_attention=cls_attention,
        num_retained_tokens=num_retained_tokens,
    )


def _append_post_merge_trace(
    astra_config: AstraConfig,
    payload: dict,
) -> None:
    """Best-effort JSONL trace writer for post-merge debugging."""
    if not bool(getattr(astra_config, "post_merge_trace_enabled", False)):
        return
    path = str(getattr(astra_config, "post_merge_trace_path", "")).strip()
    if not path:
        path = str(os.environ.get("ASTRA_POST_MERGE_TRACE_PATH", "")).strip()
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        return


def _trace_semantic_recycle_pruning_stats(
    *,
    astra_config: AstraConfig,
    branch: str,
    offset: int,
    seg_T: int,
    target_k: int,
    frame_fluctuation: float,
    k_dom: int,
    k_visual_cluster: int,
    k_sem: int,
    rem_kept_gidx: torch.Tensor,
    leaf_gidx: torch.Tensor,
    use_leaf: bool,
    visual_cluster_centers: torch.Tensor,
    visual_cluster_gidx: torch.Tensor,
    sem_gidx: torch.Tensor,
) -> None:
    if not bool(getattr(astra_config, "post_merge_trace_enabled", False)):
        return
    rem_kept_n = int(rem_kept_gidx.shape[0])
    leaf_n = int(leaf_gidx.shape[0]) if use_leaf else 0
    visual_cluster_center_n = int(visual_cluster_centers.shape[0])
    visual_cluster_center_from_leaf = 0
    if use_leaf and leaf_n > 0 and visual_cluster_center_n > 0:
        visual_cluster_center_from_leaf = int((visual_cluster_centers >= rem_kept_n).sum().item())
    visual_cluster_center_from_rem = max(0, visual_cluster_center_n - visual_cluster_center_from_leaf)

    def _overlap_count(a: torch.Tensor, b: torch.Tensor) -> int:
        if int(a.numel()) == 0 or int(b.numel()) == 0:
            return 0
        try:
            return int(torch.isin(a, b).sum().item())
        except Exception:
            sb = set(b.detach().cpu().tolist())
            return sum(int(x in sb) for x in a.detach().cpu().tolist())

    visual_cluster_pick_n = int(visual_cluster_gidx.shape[0])
    sem_pick_n = int(sem_gidx.shape[0])
    visual_cluster_pick_from_leaf = _overlap_count(visual_cluster_gidx, leaf_gidx) if use_leaf and leaf_n > 0 else 0
    sem_pick_from_leaf = _overlap_count(sem_gidx, leaf_gidx) if use_leaf and leaf_n > 0 else 0
    payload = {
        "branch": branch,
        "offset": int(offset),
        "seg_T": int(seg_T),
        "target_k": int(target_k),
        "frame_fluctuation": float(frame_fluctuation),
        "k_dom": int(k_dom),
        "k_visual_cluster": int(k_visual_cluster),
        "k_sem": int(k_sem),
        "rem_kept_n": int(rem_kept_n),
        "leaf_pool_n": int(leaf_n),
        "visual_cluster_center_n": int(visual_cluster_center_n),
        "visual_cluster_center_from_leaf": int(visual_cluster_center_from_leaf),
        "visual_cluster_center_from_rem": int(visual_cluster_center_from_rem),
        "visual_cluster_pick_n": int(visual_cluster_pick_n),
        "visual_cluster_pick_from_leaf": int(visual_cluster_pick_from_leaf),
        "sem_pick_n": int(sem_pick_n),
        "sem_pick_from_leaf": int(sem_pick_from_leaf),
        "post_merge_selection_method": str(getattr(astra_config, "post_merge_selection_method", "")),
    }
    _append_post_merge_trace(astra_config, payload)


def _factor_spatial_hw(num_tokens: int) -> Tuple[int, int]:
    """Choose H,W with H*W=num_tokens, preferring factors near sqrt (handles non-square N)."""
    nt = int(num_tokens)
    if nt <= 0:
        return 1, 1
    if nt == 1:
        return 1, 1
    r = int(math.isqrt(nt))
    for h in range(r, 0, -1):
        if nt % h == 0:
            return h, nt // h
    return 1, nt


def _spatial_window_prev_cand(
    s_pair: float,
    ih: int,
    iw: int,
    H: int,
    W: int,
    N: int,
    device: torch.device,
    pivot: float,
    prev_cand_global: torch.Tensor,
) -> torch.Tensor:
    """
    Previous-frame candidate indices for token at grid (ih, iw).
    Frame similarity s_pair in [-1, 1]: higher -> smaller window (same-cell only at ~1);
    at pivot -> use full grid; below pivot caller should use global (or skip pair separately).
    """
    eps = 1e-6
    if s_pair >= 1.0 - eps:
        return torch.tensor([ih * W + iw], device=device, dtype=torch.long)
    if s_pair <= pivot + eps:
        return prev_cand_global
    span = max(1.0 - pivot, eps)
    u = (float(s_pair) - pivot) / span
    u = min(max(u, 0.0), 1.0)
    r_max = max(ih, H - 1 - ih, iw, W - 1 - iw)
    r = int(round((1.0 - u) * float(r_max)))
    r = max(0, min(r, r_max))
    h0 = max(0, ih - r)
    h1 = min(H, ih + r + 1)
    w0 = max(0, iw - r)
    w1 = min(W, iw + r + 1)
    rr = torch.arange(h0, h1, device=device, dtype=torch.long)
    cc = torch.arange(w0, w1, device=device, dtype=torch.long)
    return (rr.unsqueeze(1) * W + cc.unsqueeze(0)).reshape(-1)


@torch.no_grad()
def temporal_backward_merge(
    video_features: torch.Tensor,
    token_mask: torch.BoolTensor,
    sim_threshold: float = 0.8,
    skip_if_frame_sim_lt: float = -1.0,
    skip_if_frame_sim_gt: float = 1.0,
    stats: Optional[dict] = None,
    grid_h: int = 0,
    grid_w: int = 0,
    same_cell_only: bool = False,
    full_prev_frame: bool = False,
) -> Tuple[torch.BoolTensor, torch.BoolTensor]:
    """
    Backward temporal token merging (from last frame to first).

    Only tokens where token_mask=True participate. Tokens that are merged are removed
    from the keep set; merged-into tokens keep their original position indices.

    Spatial search on the previous frame uses a real HxW grid when grid_h*grid_w==N (else factor N).
    Per adjacent pair, frame-level cosine s: if skip_if_frame_sim_lt>0 and s<it, skip merging this pair.
    If full_prev_frame is True (and grid), each token searches over all positions on the previous frame.
    If same_cell_only is True, only the aligned grid cell on the previous frame is considered (ablation).
    Otherwise, search window grows as s decreases: s≈1 -> same grid cell only; s at pivot -> full grid;
    between pivot and 1 -> square window radius interpolated (larger radius when s is lower).
    """
    T, N, _ = video_features.shape
    device = video_features.device
    if T <= 1 or N == 0:
        out = token_mask.clone()
        return out, torch.zeros((T, N), dtype=torch.bool, device=out.device)

    if grid_h > 0 and grid_w > 0 and int(grid_h) * int(grid_w) == int(N):
        H, W = int(grid_h), int(grid_w)
    else:
        H, W = _factor_spatial_hw(int(N))
    has_grid = H * W == int(N)

    feats = video_features.float()
    feats = feats / feats.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)

    sum_feats = video_features.float().clone()
    counts = torch.ones((T, N), device=device, dtype=torch.float32)
    keep_mask = token_mask.clone()

    # Pivot for window growth: at s==pivot use full-grid search; if skip_if_frame_sim_lt>0, use it as pivot.
    pivot = float(skip_if_frame_sim_lt) if float(skip_if_frame_sim_lt) > 0.0 else 0.8

    def _frame_sim(t: int) -> float:
        a_mask = token_mask[t]
        b_mask = token_mask[t - 1]
        a = feats[t, a_mask].mean(dim=0) if bool(a_mask.any().item()) else feats[t].mean(dim=0)
        b = feats[t - 1, b_mask].mean(dim=0) if bool(b_mask.any().item()) else feats[t - 1].mean(dim=0)
        a = a / a.norm(p=2, dim=-1, keepdim=False).clamp_min(1e-12)
        b = b / b.norm(p=2, dim=-1, keepdim=False).clamp_min(1e-12)
        return float(torch.dot(a, b).item())

    skipped_pairs = 0
    prev_cand_global = torch.arange(N, device=device, dtype=torch.long)
    for t in range(T - 1, 0, -1):
        s = _frame_sim(t)
        skip_lo = float(skip_if_frame_sim_lt)
        skip_hi = float(skip_if_frame_sim_gt)
        if skip_lo > 0.0 and float(s) < skip_lo:
            skipped_pairs += 1
            continue
        if float(s) > skip_hi:
            skipped_pairs += 1
            continue

        cur_mask = token_mask[t]
        if not bool(cur_mask.any().item()):
            continue

        cur_idx = torch.where(cur_mask)[0]
        for i in cur_idx.tolist():
            if not bool(keep_mask[t, i].item()):
                continue

            cur = feats[t, i]
            if has_grid:
                ih = i // W
                iw = i % W
                if full_prev_frame:
                    prev_cand = prev_cand_global
                elif same_cell_only:
                    prev_cand = torch.tensor([ih * W + iw], device=device, dtype=torch.long)
                else:
                    prev_cand = _spatial_window_prev_cand(
                        float(s), ih, iw, H, W, N, device, pivot, prev_cand_global
                    )
            else:
                prev_cand = prev_cand_global

            sims = torch.matmul(feats[t - 1, prev_cand], cur)
            best_sim, best_k = torch.max(sims, dim=0)
            if float(best_sim.item()) < float(sim_threshold):
                continue

            j = int(prev_cand[best_k].item())
            sum_feats[t - 1, j] += video_features[t, i].float()
            counts[t - 1, j] += 1.0
            keep_mask[t, i] = False

        denom = counts[t - 1].clamp_min(1.0).unsqueeze(-1)
        video_features[t - 1] = (sum_feats[t - 1] / denom).to(video_features.dtype)
        feats[t - 1] = video_features[t - 1].float()
        feats[t - 1] = feats[t - 1] / feats[t - 1].norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)

    out = keep_mask & token_mask
    anchor_fused = (counts > 1.0) & out
    if stats is not None:
        kept = int(out.sum().item())
        total = int(token_mask.sum().item())
        stats.update(
            {
                "T": int(T),
                "N": int(N),
                "same_cell_only": bool(same_cell_only),
                "full_prev_frame": bool(full_prev_frame),
                "sim_threshold": float(sim_threshold),
                "skip_if_frame_sim_lt": float(skip_if_frame_sim_lt),
                "skip_if_frame_sim_gt": float(skip_if_frame_sim_gt),
                "skipped_pairs": int(skipped_pairs),
                "kept_tokens": int(kept),
                "merged_tokens": int(total - kept),
                "fused_anchor_tokens": int(anchor_fused.sum().item()),
            }
        )
    return out, anchor_fused


def _fill_tmerge_debug_stats(
    stats: dict,
    T: int,
    N: int,
    token_mask: torch.BoolTensor,
    keep_out: torch.BoolTensor,
    anchor_fused: torch.BoolTensor,
    sim_threshold: float,
    skip_if_frame_sim_lt: float,
    skip_if_frame_sim_gt: float,
    *,
    skipped_pairs: Optional[int] = None,
) -> None:
    """Match temporal_backward_merge(stats=...) keys when reusing a merge without re-running."""
    kept = int(keep_out.sum().item())
    total = int(token_mask.sum().item())
    stats.update(
        {
            "T": int(T),
            "N": int(N),
            "sim_threshold": float(sim_threshold),
            "skip_if_frame_sim_lt": float(skip_if_frame_sim_lt),
            "skip_if_frame_sim_gt": float(skip_if_frame_sim_gt),
            "skipped_pairs": int(skipped_pairs) if skipped_pairs is not None else -1,
            "kept_tokens": kept,
            "merged_tokens": int(total - kept),
            "fused_anchor_tokens": int(anchor_fused.sum().item()),
        }
    )


@torch.no_grad()
def _collect_temporal_merge_leaf_tokens(
    segment_features: torch.Tensor,
    segment_cls_attention: torch.Tensor,
    segment_global_indices: torch.Tensor,
    token_mask: torch.BoolTensor,
    keep_mask: torch.BoolTensor,
    offset: int,
    seg_T: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tokens removed by temporal backward merge (merged into a previous-frame anchor).
    Features at dropped grid positions are still the leaf embeddings (unchanged in merge).
    Returns (feats, cls_attn, global_idx, frame_ids) with frame_ids = offset + local_t.
    """
    dropped = token_mask & (~keep_mask)
    device = segment_features.device
    feat_dim = int(segment_features.shape[-1])
    dtype = segment_features.dtype
    feats_l: List[torch.Tensor] = []
    attn_l: List[torch.Tensor] = []
    gidx_l: List[torch.Tensor] = []
    fid_l: List[torch.Tensor] = []
    for t in range(seg_T):
        idx = torch.where(dropped[t])[0]
        if idx.numel() == 0:
            continue
        feats_l.append(segment_features[t, idx])
        attn_l.append(segment_cls_attention[t, idx].float().clamp_min(1e-6).to(dtype=dtype))
        gidx_l.append(segment_global_indices[t, idx])
        fid_l.append(
            torch.full(
                (idx.numel(),),
                int(offset + t),
                device=device,
                dtype=torch.long,
            )
        )
    if len(feats_l) == 0:
        zf = torch.zeros((0, feat_dim), device=device, dtype=dtype)
        za = torch.zeros((0,), device=device, dtype=dtype)
        zi = torch.zeros((0,), device=device, dtype=torch.long)
        return zf, za, zi, zi
    return (
        torch.cat(feats_l, dim=0),
        torch.cat(attn_l, dim=0),
        torch.cat(gidx_l, dim=0),
        torch.cat(fid_l, dim=0),
    )


def _semantic_recycle_select_recycle_frames(
    seg_sem: torch.Tensor,
    seg_T: int,
    device: torch.device,
    policy: str,
    *,
    frame_mass: float,
    frame_temp: float,
    num_frames_override: int,
    frame_tau: float,
    frame_min: int,
    frame_max: int,
) -> torch.Tensor:
    """
    Local frame indices (0..seg_T-1) for semantic_recycle recycle budget splitting, ordered by relevance (higher first).

    seg_sem: per-frame relevance vs current text (same signal as existing sem_scores slice).
    """
    if seg_T <= 0:
        return torch.zeros((0,), dtype=torch.long, device=device)
    ss = seg_sem.flatten().to(device=device, dtype=torch.float32)
    if int(ss.numel()) >= int(seg_T):
        ss = ss[: int(seg_T)].clone()
    else:
        pad = torch.ones(int(seg_T) - int(ss.numel()), device=device, dtype=torch.float32)
        ss = torch.cat([ss, pad], dim=0)
    ss = ss.clamp_min(0.0)

    pol = str(policy).strip().lower()
    if pol in ("", "top_quarter", "fixed_quarter", "legacy"):
        k = max(1, (int(seg_T) + 3) // 4)
        k = min(k, int(seg_T))
        _, idx = torch.topk(ss, k=k, largest=True, sorted=True)
        return idx

    if pol in ("query_topk", "topk"):
        k = int(num_frames_override) if int(num_frames_override) > 0 else max(1, (int(seg_T) + 3) // 4)
        k = max(1, min(k, int(seg_T)))
        _, idx = torch.topk(ss, k=k, largest=True, sorted=True)
        return idx

    if pol in ("query_mass", "softmax_mass", "mass"):
        t = max(float(frame_temp), 1e-6)
        w = torch.softmax(ss / t, dim=0)
        order = torch.argsort(w, descending=True)
        cum = torch.cumsum(w[order], dim=0)
        m = min(max(float(frame_mass), 0.50), 0.999)
        n_take = int((cum < m).sum().item()) + 1
        n_take = max(int(frame_min), min(n_take, int(seg_T)))
        if int(frame_max) > 0:
            n_take = min(n_take, int(frame_max))
        return order[:n_take].long()

    if pol in ("query_threshold", "threshold"):
        mx = float(ss.max().item()) if ss.numel() else 1.0
        thr = mx * float(frame_tau)
        sel = torch.where(ss >= thr)[0]
        fmin = max(1, int(frame_min))
        if int(sel.numel()) < fmin:
            _, idx = torch.topk(ss, k=min(fmin, int(seg_T)), largest=True, sorted=True)
            return idx
        vals = ss[sel]
        o = torch.argsort(vals, descending=True)
        idx = sel[o]
        if int(frame_max) > 0 and int(idx.numel()) > int(frame_max):
            idx = idx[: int(frame_max)]
        return idx.long()

    k = max(1, (int(seg_T) + 3) // 4)
    k = min(k, int(seg_T))
    _, idx = torch.topk(ss, k=k, largest=True, sorted=True)
    return idx


def _semantic_recycle_recycle_frames_with_stv_coverage_schedule(
    seg_sem: torch.Tensor,
    seg_T: int,
    device: torch.device,
    astra_config: AstraConfig,
) -> Tuple[torch.Tensor, int]:
    """Baseline recycle frames from policy, then optionally expand/shrink count vs average_stv.

    When recycle_stv_coverage_schedule is True:
    - average_stv <= recycle_stv_cov_fluct_low → top-k uses all seg_T frames (full coverage).
    - average_stv >= recycle_stv_cov_fluct_high → top-k uses baseline policy count only (e.g. ceil(T/4)).
    - Between: linear interpolation of frame count between seg_T and baseline.

    Returns (top_frame_local_indices, n_baseline) where n_baseline is the policy frame count without schedule.
    """
    pol = str(getattr(astra_config, "post_merge_recycle_frame_policy", "top_quarter")).strip().lower()
    top_base = _semantic_recycle_select_recycle_frames(
        seg_sem,
        int(seg_T),
        device,
        pol,
        frame_mass=float(getattr(astra_config, "post_merge_recycle_frame_mass", 0.85)),
        frame_temp=float(getattr(astra_config, "post_merge_recycle_frame_temp", 0.10)),
        num_frames_override=int(getattr(astra_config, "post_merge_recycle_num_frames", 0)),
        frame_tau=float(getattr(astra_config, "post_merge_recycle_frame_tau", 0.55)),
        frame_min=int(getattr(astra_config, "post_merge_recycle_frame_min", 1)),
        frame_max=int(getattr(astra_config, "post_merge_recycle_frame_max", 0)),
    )
    n_baseline = int(top_base.numel())
    if not bool(getattr(astra_config, "recycle_stv_coverage_schedule", False)):
        return top_base, n_baseline
    stv_val = getattr(astra_config, "average_stv", None)
    if stv_val is None or int(seg_T) <= 1:
        setattr(
            astra_config,
            "last_recycle_adapt_stats",
            {
                "enabled": False,
                "mode": "stv_coverage_schedule",
                "reason": "missing_average_stv" if stv_val is None else "seg_T_le_1",
                "n_baseline": n_baseline,
            },
        )
        return top_base, n_baseline
    low = float(getattr(astra_config, "recycle_stv_cov_fluct_low", 0.05))
    high = float(getattr(astra_config, "recycle_stv_cov_fluct_high", 0.2))
    if high <= low:
        high = low + 1e-6
    frac = (float(stv_val) - low) / (high - low)
    frac = max(0.0, min(1.0, frac))
    n_target = int(round(float(seg_T) - frac * (float(seg_T) - float(n_baseline))))
    n_target = max(n_baseline, min(int(seg_T), n_target))
    ss = seg_sem.flatten().to(device=device, dtype=torch.float32)
    if int(ss.numel()) >= int(seg_T):
        ss = ss[: int(seg_T)].clone()
    else:
        pad = torch.ones(int(seg_T) - int(ss.numel()), device=device, dtype=torch.float32)
        ss = torch.cat([ss, pad], dim=0)
    ss = ss.clamp_min(0.0)
    _, idx = torch.topk(ss, k=n_target, largest=True, sorted=True)
    setattr(
        astra_config,
        "last_recycle_adapt_stats",
        {
            "enabled": True,
            "mode": "stv_coverage_schedule",
            "average_stv": float(stv_val),
            "fluct_low": low,
            "fluct_high": high,
            "frac": frac,
            "n_baseline": n_baseline,
            "n_final": int(idx.numel()),
            "seg_T": int(seg_T),
        },
    )
    return idx, n_baseline


def _split_three_way_budget(target_k: int, a: int, b: int, c: int) -> Tuple[int, int, int]:
    """Split target_k into (k_dom, k_visual_cluster, k_sem) by integer parts (largest-remainder)."""
    target_k = int(target_k)
    if target_k <= 0:
        return 0, 0, 0
    a, b, c = max(0, int(a)), max(0, int(b)), max(0, int(c))
    tot = a + b + c
    if tot <= 0:
        return target_k, 0, 0
    k_dom = (target_k * a) // tot
    k_visual_cluster = (target_k * b) // tot
    k_sem = (target_k * c) // tot
    r = target_k - k_dom - k_visual_cluster - k_sem
    rem_fracs: List[Tuple[int, int]] = [
        ((target_k * a) % tot, 0),
        ((target_k * b) % tot, 1),
        ((target_k * c) % tot, 2),
    ]
    rem_fracs.sort(key=lambda x: (-x[0], x[1]))
    for i in range(r):
        which = rem_fracs[i % 3][1]
        if which == 0:
            k_dom += 1
        elif which == 1:
            k_visual_cluster += 1
        else:
            k_sem += 1
    return k_dom, k_visual_cluster, k_sem


def _split_two_way_budget(target_k: int, a: int, b: int) -> Tuple[int, int]:
    """Split target_k into (k_a, k_b) by integer parts (largest-remainder)."""
    target_k = int(target_k)
    if target_k <= 0:
        return 0, 0
    a, b = max(0, int(a)), max(0, int(b))
    tot = a + b
    if tot <= 0:
        return target_k, 0
    k_a = (target_k * a) // tot
    k_b = (target_k * b) // tot
    r = target_k - k_a - k_b
    frac_a = (target_k * a) % tot
    frac_b = (target_k * b) % tot
    # Largest remainder; tie-break prefers k_a (stage-1).
    for _ in range(r):
        if frac_a >= frac_b:
            k_a += 1
        else:
            k_b += 1
    return k_a, k_b


def _estimate_segment_frame_fluctuation(
    segment_features: torch.Tensor,
    segment_cls_attention: torch.Tensor,
    token_mask: Optional[torch.Tensor],
) -> float:
    """
    Estimate average frame fluctuation by adjacent weighted-frame cosine:
      fluct = clamp(1 - mean(cos(f_t, f_{t+1})), 0, 1).
    """
    seg_T = int(segment_features.shape[0])
    if seg_T <= 1:
        return 0.0
    device = segment_features.device
    reps: List[torch.Tensor] = []
    for t in range(seg_T):
        if token_mask is None:
            mask_t = torch.ones(
                (segment_features.shape[1],), device=device, dtype=torch.bool
            )
        else:
            mask_t = token_mask[t].bool()
        idx = torch.where(mask_t)[0]
        if idx.numel() == 0:
            reps.append(torch.zeros((segment_features.shape[-1],), device=device, dtype=torch.float32))
            continue
        ft = segment_features[t, idx].float()
        wt = segment_cls_attention[t, idx].float().clamp_min(1e-6)
        rep = (ft * wt.unsqueeze(-1)).sum(dim=0) / wt.sum().clamp_min(1e-12)
        reps.append(rep)
    fr = torch.stack(reps, dim=0)
    fr = fr / fr.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    sim = (fr[:-1] * fr[1:]).sum(dim=-1).mean()
    fluct = float(torch.clamp(1.0 - sim, min=0.0, max=1.0).item())
    return fluct


def _dynamic_attn_visual_cluster_sem_budget(
    total_k: int,
    frame_fluct: float,
    astra_config: AstraConfig,
) -> Tuple[int, int, int]:
    """
    Dynamic split for semantic_recycle_pruning:
      fluct <= low  : (dom,visual_cluster,sem) = (0.7,0.0,0.3)
      fluct >= high : (dom,visual_cluster,sem) = (0.7,0.2,0.1)
      low~high      : linear interpolation for visual_cluster/sem.
    """
    total_k = int(total_k)
    if total_k <= 0:
        return 0, 0, 0
    dom_ratio = float(getattr(astra_config, "post_merge_dynamic_dom_ratio", 0.70))
    low = float(getattr(astra_config, "post_merge_dynamic_fluct_low", 0.10))
    high = float(getattr(astra_config, "post_merge_dynamic_fluct_high", 0.20))
    visual_cluster_hi = float(getattr(astra_config, "post_merge_dynamic_visual_cluster_high_ratio", 0.20))
    sem_lo = float(getattr(astra_config, "post_merge_dynamic_sem_low_ratio", 0.30))
    sem_hi = float(getattr(astra_config, "post_merge_dynamic_sem_high_ratio", 0.10))

    if high <= low:
        high = low + 1e-6
    if frame_fluct <= low:
        visual_cluster_ratio = 0.0
        sem_ratio = sem_lo
    elif frame_fluct >= high:
        visual_cluster_ratio = visual_cluster_hi
        sem_ratio = sem_hi
    else:
        t = (frame_fluct - low) / (high - low)
        visual_cluster_ratio = visual_cluster_hi * t
        sem_ratio = sem_lo + (sem_hi - sem_lo) * t

    dom_ratio = min(max(dom_ratio, 0.0), 1.0)
    visual_cluster_ratio = min(max(visual_cluster_ratio, 0.0), 1.0)
    sem_ratio = min(max(sem_ratio, 0.0), 1.0)
    s = dom_ratio + visual_cluster_ratio + sem_ratio
    if s <= 1e-8:
        return total_k, 0, 0
    dom_ratio, visual_cluster_ratio, sem_ratio = dom_ratio / s, visual_cluster_ratio / s, sem_ratio / s
    parts = (
        max(0, int(round(dom_ratio * 1000))),
        max(0, int(round(visual_cluster_ratio * 1000))),
        max(0, int(round(sem_ratio * 1000))),
    )
    return _split_three_way_budget(total_k, parts[0], parts[1], parts[2])


@torch.no_grad()
def _di_spm_fuse(
    k_out: int,
    flat_feats: torch.Tensor,
    flat_attn: torch.Tensor,
    flat_gidx: torch.Tensor,
    flat_frame_ids: torch.Tensor,
    leaf_feats: torch.Tensor,
    leaf_attn: torch.Tensor,
    leaf_gidx: torch.Tensor,
    leaf_frame_ids: torch.Tensor,
    astra_config: AstraConfig,
    *,
    use_leaf: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Diversity-Importance Spatial Prune-Merge (DI-SPM).

    Produce k_out "anchor" tokens chosen by visual_guided_pruning (importance×diversity),
    then softly merge the remainder into anchors with density-weighted assignment.

    Returns:
      (anchors_feat, anchors_gidx, donor_feat, donor_meta)
    where donor_meta packs (donor_gidx, donor_attn, donor_frame_ids, donor_max_assign_w) as a (N,4) tensor
    for convenience in downstream text-guided recall (TSR).
    """
    device = flat_feats.device
    dtype = flat_feats.dtype
    feat_dim = int(flat_feats.shape[-1])
    if k_out <= 0 or flat_feats.numel() == 0:
        zf = torch.zeros((0, feat_dim), device=device, dtype=dtype)
        zi = torch.zeros((0,), device=device, dtype=torch.long)
        zm = torch.zeros((0, 4), device=device, dtype=torch.float32)
        return zf, zi, zf, zm

    n_cand = int(flat_feats.shape[0])
    k_out = min(int(k_out), n_cand)
    # 1) Anchor selection by visual_guided_pruning (grid-only pool).
    anc_feat, anc_idx = visual_guided_pruning_based_token_selection(
        features=flat_feats.unsqueeze(0),
        cls_attention=flat_attn.unsqueeze(0),
        num_retained_tokens=k_out,
    )
    anc_idx = anc_idx.squeeze(0).long()
    anc_feat = anc_feat.squeeze(0)
    anc_gidx = flat_gidx[anc_idx]

    # 2) Build donor pool = remainder (+ optional leaves).
    rem_mask = torch.ones(n_cand, dtype=torch.bool, device=device)
    rem_mask[anc_idx] = False
    rem_feats = flat_feats[rem_mask]
    rem_gidx = flat_gidx[rem_mask]
    rem_attn = flat_attn[rem_mask]
    rem_fids = flat_frame_ids[rem_mask]
    if use_leaf and int(leaf_feats.shape[0]) > 0:
        donor_feats = torch.cat([rem_feats, leaf_feats.to(rem_feats.dtype)], dim=0) if rem_feats.numel() > 0 else leaf_feats.to(rem_feats.dtype)
        donor_gidx = torch.cat([rem_gidx, leaf_gidx], dim=0) if rem_gidx.numel() > 0 else leaf_gidx
        donor_attn = torch.cat([rem_attn, leaf_attn.to(rem_attn.dtype)], dim=0) if rem_attn.numel() > 0 else leaf_attn.to(rem_attn.dtype)
        donor_fids = torch.cat([rem_fids, leaf_frame_ids], dim=0) if rem_fids.numel() > 0 else leaf_frame_ids
    else:
        donor_feats = rem_feats
        donor_gidx = rem_gidx
        donor_attn = rem_attn
        donor_fids = rem_fids

    n_d = int(donor_feats.shape[0])
    if n_d == 0:
        zm = torch.zeros((0, 4), device=device, dtype=torch.float32)
        return anc_feat, anc_gidx, donor_feats, zm

    # 3) Density proxy for donors (VisualCluster spirit): average cosine to top-k neighbors (k capped).
    k_den = min(7, max(1, n_d - 1))
    df = donor_feats.float()
    df = df / df.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    sim_dd = torch.matmul(df, df.t())
    sim_dd.fill_diagonal_(-1.0)
    topk = torch.topk(sim_dd, k=k_den, dim=-1).values
    density = topk.mean(dim=-1).clamp_min(0.0)  # (n_d,)

    # 4) Soft assignment donor -> anchors (ContextualMerge-like), density-weighted.
    af = anc_feat.float()
    af = af / af.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    sim_da = torch.matmul(df, af.t())  # (n_d, k_out)
    tau = max(float(getattr(astra_config, "post_merge_di_assign_temp", 1.0)), 1e-6)
    w = torch.softmax(sim_da / tau, dim=-1)  # (n_d, k_out)
    w = w * density.unsqueeze(-1)
    w_sum = w.sum(dim=0).clamp_min(1e-12)  # (k_out,)
    delta = torch.matmul(w.t(), donor_feats.float()) / w_sum.unsqueeze(-1)  # (k_out, d)
    inject = float(getattr(astra_config, "post_merge_di_inject", 0.35))
    inject = min(max(inject, 0.0), 1.0)
    if inject > 0.0:
        out = (1.0 - inject) * anc_feat.float() + inject * delta
        out = out / out.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        anc_out = out.to(dtype=dtype)
    else:
        anc_out = anc_feat

    donor_maxw = w.max(dim=-1).values  # (n_d,)
    donor_meta = torch.stack(
        [
            donor_gidx.to(torch.float32),
            donor_attn.to(torch.float32),
            donor_fids.to(torch.float32),
            donor_maxw.to(torch.float32),
        ],
        dim=-1,
    )  # (n_d, 4)
    return anc_out, anc_gidx, donor_feats, donor_meta


def _minmax01_1d(x: torch.Tensor) -> torch.Tensor:
    """Min-max normalize 1D tensor to ~[0,1]; constant input -> all ones."""
    if x.numel() == 0:
        return x
    mn, mx = x.min(), x.max()
    if float((mx - mn).item()) < 1e-12:
        return torch.ones_like(x, dtype=x.dtype, device=x.device)
    return ((x - mn) / (mx - mn + 1e-12)).clamp(0.0, 1.0)


@torch.no_grad()
def _semantic_recycle_baseline_min_norm_picked_token_rel(
    rel: torch.Tensor,
    rem_frame_ids: torch.Tensor,
    offset: int,
    top_frames: torch.Tensor,
    k_pick: int,
    n0: int,
) -> float:
    """
    Simulate baseline discrete recycle: per selected frame, min-max normalize rel within that frame's remainder,
    take top-need by raw rel; return min normalized score among all picked tokens (empty / missing -> 0.0).
    """
    if n0 <= 0 or k_pick <= 0:
        return 0.0
    base_k = k_pick // n0
    rem_k = k_pick - base_k * n0
    picked_mins: List[float] = []
    for rank in range(n0):
        ft_local = int(top_frames[rank].item())
        need = base_k + (1 if rank < rem_k else 0)
        if need <= 0:
            continue
        mask = rem_frame_ids == int(offset + ft_local)
        idx = torch.where(mask)[0]
        if idx.numel() == 0:
            return 0.0
        kk = min(need, int(idx.numel()))
        loc_rel = rel[idx].float()
        norm = _minmax01_1d(loc_rel)
        top_local = torch.topk(loc_rel, k=kk, largest=True).indices
        picked_norm = norm[top_local]
        picked_mins.append(float(picked_norm.min().item()))
    if len(picked_mins) == 0:
        return 0.0
    return min(picked_mins)


@torch.no_grad()
def _semantic_recycle_baseline_min_norm_cls_on_frames(seg_sem: torch.Tensor, seg_T: int, frame_idx: torch.Tensor) -> float:
    """Min-max normalize segment frame CLS relevance, then min over given local frame indices."""
    if seg_T <= 0 or frame_idx.numel() == 0:
        return 0.0
    ss = seg_sem.flatten().float()[: int(seg_T)]
    if ss.numel() < int(seg_T):
        pad = torch.ones(int(seg_T) - int(ss.numel()), device=ss.device, dtype=torch.float32)
        ss = torch.cat([ss, pad], dim=0)
    cls_norm = _minmax01_1d(ss)
    fi = frame_idx.long().clamp(0, int(seg_T) - 1)
    return float(cls_norm[fi].min().item())


@torch.no_grad()
def _semantic_recycle_discrete_recycle_pick(
    k_pick: int,
    rem_feats: torch.Tensor,
    rem_gidx: torch.Tensor,
    rem_attn: torch.Tensor,
    rem_frame_ids: torch.Tensor,
    text: Optional[torch.Tensor],
    sem_scores: Optional[torch.Tensor],
    offset: int,
    seg_T: int,
    astra_config: AstraConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """semantic_recycle-style discrete semantic recycle: up to k_pick tokens from rem_* (same logic as semantic_recycle_visual_guided_pruning recycle)."""
    device = rem_feats.device
    dtype = rem_feats.dtype
    feat_dim = int(rem_feats.shape[-1]) if rem_feats.numel() else 0
    if k_pick <= 0 or rem_feats.numel() == 0:
        zf = torch.zeros((0, feat_dim), device=device, dtype=dtype)
        zi = torch.zeros((0,), device=device, dtype=torch.long)
        return zf, zi

    allow_sem_fallback = str(os.getenv("ASTRA_ALLOW_SEMANTIC_TEXT_FALLBACK", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }

    # Prefer SigLIP-joint-space pooled-token relevance (pre-mm_projector pooled grid), indexed by global pooled ids.
    rel: Optional[torch.Tensor] = None
    rel_flat = getattr(astra_config, "siglip_pooled_token_relevance", None)
    if rel_flat is not None and isinstance(rel_flat, torch.Tensor) and rel_flat.numel() > 0:
        try:
            mx = int(rem_gidx.max().item()) if rem_gidx.numel() > 0 else -1
            if mx >= 0 and rel_flat.numel() > mx:
                rel = rel_flat.to(device=rem_gidx.device).float()[rem_gidx]
        except Exception:
            rel = None

    if rel is None:
        if text is not None:
            if text.dim() == 2:
                q = text[0]
            else:
                q = text
            q = q.to(rem_feats.device).float()
            q = q / q.norm(p=2, dim=-1, keepdim=False).clamp_min(1e-12)
            rf = rem_feats.float()
            rf = rf / rf.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
            rel = torch.matmul(rf, q)
        elif allow_sem_fallback:
            top = torch.topk(rem_attn.float(), k=min(k_pick, int(rem_attn.numel()))).indices
            return rem_feats[top], rem_gidx[top]
        else:
            raise RuntimeError(
                "Astra semantic recycle: missing both usable siglip_pooled_token_relevance (check llava_arch "
                "query_text + SigLIP simulate pool + indices vs rem_gidx) and LLM-space query text "
                "(text_cls_embed; unset on LLaVA). Cannot silently fall back to attention-only picking. "
                "For debugging only, export ASTRA_ALLOW_SEMANTIC_TEXT_FALLBACK=1."
            )

    if sem_scores is not None and sem_scores.numel() >= (offset + seg_T):
        seg_sem = sem_scores[offset : offset + seg_T].to(rel.device).float()
    else:
        seg_sem = torch.ones((seg_T,), device=rel.device, dtype=torch.float32)

    pol = str(getattr(astra_config, "post_merge_recycle_frame_policy", "top_quarter")).strip().lower()
    top_frames, _ = _semantic_recycle_recycle_frames_with_stv_coverage_schedule(
        seg_sem,
        int(seg_T),
        rel.device,
        astra_config,
    )
    n0 = int(top_frames.numel())
    adapt_stats: Optional[dict] = None
    pol_l = str(pol).strip().lower()
    blocked_pol = pol_l in ("query_mass", "softmax_mass", "mass", "query_threshold", "threshold")
    schedule_on = bool(getattr(astra_config, "recycle_stv_coverage_schedule", False))
    stv_val = getattr(astra_config, "average_stv", None)
    if stv_val is None:
        stv_val = getattr(astra_config, "stv_frame_mu_d", None)  # backward compat
    if (
        bool(getattr(astra_config, "recycle_mu_d_adapt", False))
        and (not schedule_on)
        and (not blocked_pol)
        and stv_val is not None
        and int(seg_T) > 1
        and n0 > 0
        and k_pick > 0
    ):
        gate = float(getattr(astra_config, "recycle_rel_gate", 0.8))
        n_extra_cfg = int(getattr(astra_config, "recycle_mu_d_extra_frames", 0))
        n_extra = max(1, n0 // 4) if n_extra_cfg <= 0 else n_extra_cfg
        n_max = min(int(seg_T), n0 + n_extra)
        n_min = max(1, n0 - n_extra)
        ml = float(getattr(astra_config, "recycle_mu_d_map_low", -1.0))
        if ml < 0:
            ml = float(getattr(astra_config, "stv_budget_mu_low", 0.1))
        mf = float(getattr(astra_config, "recycle_mu_d_map_full", -1.0))
        if mf < 0:
            mf = float(getattr(astra_config, "stv_budget_mu_full", 0.6))
        if mf <= ml:
            mf = ml + 1e-6
        frac = (float(stv_val) - ml) / (mf - ml)
        frac = max(0.0, min(1.0, frac))
        n_target = int(round(n_max - frac * (n_max - n_min)))
        n_target = max(1, min(int(seg_T), n_target))
        n_before_gate = n_target
        gate_token: Optional[float] = None
        gate_cls: Optional[float] = None
        blocked = False
        reason = "ok"
        if n_target < n0:
            gate_token = _semantic_recycle_baseline_min_norm_picked_token_rel(
                rel, rem_frame_ids, offset, top_frames, k_pick, n0
            )
            if gate_token < gate:
                n_target = n0
                blocked = True
                reason = "high_stv_token_rel"
        elif n_target > n0:
            gate_cls = _semantic_recycle_baseline_min_norm_cls_on_frames(seg_sem, int(seg_T), top_frames)
            if gate_cls < gate:
                n_target = n0
                blocked = True
                reason = "low_stv_cls_rel"
        if n_target != n0:
            ss = seg_sem.flatten().to(device=rel.device, dtype=torch.float32)
            if int(ss.numel()) >= int(seg_T):
                ss = ss[: int(seg_T)].clone()
            else:
                pad = torch.ones(int(seg_T) - int(ss.numel()), device=rel.device, dtype=torch.float32)
                ss = torch.cat([ss, pad], dim=0)
            ss = ss.clamp_min(0.0)
            k_take = min(int(seg_T), max(1, n_target))
            _, top_frames = torch.topk(ss, k=k_take, largest=True, sorted=True)
        adapt_stats = {
            "enabled": True,
            "policy": pol_l,
            "average_stv": float(stv_val),
            "mu_map_low": ml,
            "mu_map_full": mf,
            "frac": frac,
            "n0": n0,
            "n_target_raw": n_before_gate,
            "n_target_final": int(top_frames.numel()) if n_target != n0 else n0,
            "gate": gate,
            "gate_token_min_norm": gate_token,
            "gate_cls_min_norm": gate_cls,
            "blocked": blocked,
            "reason": reason,
        }
        setattr(astra_config, "last_recycle_adapt_stats", adapt_stats)
    elif bool(getattr(astra_config, "recycle_mu_d_adapt", False)) and (not schedule_on):
        setattr(
            astra_config,
            "last_recycle_adapt_stats",
            {
                "enabled": False,
                "policy": pol_l,
                "blocked_policy": blocked_pol,
                "average_stv": None if stv_val is None else float(stv_val),
                "seg_T": int(seg_T),
                "n0": n0,
            },
        )
    num_sel_frames = int(top_frames.numel())
    base_k = k_pick // num_sel_frames if num_sel_frames > 0 else 0
    rem_k = k_pick - base_k * num_sel_frames if num_sel_frames > 0 else 0
    picks: List[torch.Tensor] = []
    used = torch.zeros(rel.shape[0], dtype=torch.bool, device=rel.device)
    for rank in range(num_sel_frames):
        ft_local = int(top_frames[rank].item())
        need = base_k + (1 if rank < rem_k else 0)
        if need <= 0:
            continue
        mask = rem_frame_ids == int(offset + ft_local)
        idx = torch.where(mask)[0]
        if idx.numel() == 0:
            continue
        kk = min(need, int(idx.numel()))
        top_local = torch.topk(rel[idx], k=kk, largest=True).indices
        sel = idx[top_local]
        picks.append(sel)
        used[sel] = True
    if len(picks) > 0:
        pick = torch.cat(picks, dim=0)
    else:
        pick = torch.zeros((0,), dtype=torch.long, device=rel.device)
    short = int(k_pick - int(pick.numel()))
    if short > 0:
        cand = torch.where(~used)[0]
        if cand.numel() > 0:
            kk = min(short, int(cand.numel()))
            extra = cand[torch.topk(rel[cand], k=kk, largest=True).indices]
            pick = torch.cat([pick, extra], dim=0)
    if pick.numel() > k_pick:
        pick = pick[:k_pick]
    return rem_feats[pick], rem_gidx[pick]


@torch.no_grad()
def _post_merge_zip_fuse_contextual(
    dom_feat: torch.Tensor,
    dom_gidx: torch.Tensor,
    ctx_feat: torch.Tensor,
    text_cls: Optional[torch.Tensor],
    astra_config: AstraConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ContextualMerge-style fusion: soft-assign each contextual token to dominants (visual + optional query logits),
    aggregate contextual mass per dominant, inject with residual, L2-normalize rows.
    Token count stays len(dom_feat).
    """
    device = dom_feat.device
    dtype = dom_feat.dtype
    k = int(dom_feat.shape[0])
    if k == 0:
        zf = torch.zeros((0, dom_feat.shape[-1]), device=device, dtype=dtype)
        zi = torch.zeros((0,), device=device, dtype=torch.long)
        return zf, zi
    if ctx_feat is None or ctx_feat.numel() == 0:
        return dom_feat, dom_gidx

    lv = float(getattr(astra_config, "post_merge_zip_lambda_vis", 1.0))
    ls = float(getattr(astra_config, "post_merge_zip_lambda_sem", 0.3))
    inject = float(getattr(astra_config, "post_merge_zip_inject", 0.35))
    cross = bool(getattr(astra_config, "post_merge_zip_sem_cross_dom", True))
    tau = max(float(getattr(astra_config, "post_merge_zip_softmax_temp", 1.0)), 1e-6)

    if inject <= 0.0:
        return dom_feat, dom_gidx

    allow_sem_fallback = str(os.getenv("ASTRA_ALLOW_SEMANTIC_TEXT_FALLBACK", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    if ls > 0.0 and text_cls is None and not allow_sem_fallback:
        raise RuntimeError(
            "Astra ZIP fusion: post_merge_zip_lambda_sem > 0 but no LLM-space query vector (text_cls_embed). "
            "LLaVA no longer sets text_cls_embed; either set post_merge_zip_lambda_sem=0 / disable this branch, "
            "or set ASTRA_ALLOW_SEMANTIC_TEXT_FALLBACK=1 to run visual-only logits for ZIP."
        )

    dom_f = dom_feat.float()
    ctx = ctx_feat.float()
    dom_n = dom_f / dom_f.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    ctx_n = ctx / ctx.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    logits = lv * (ctx_n @ dom_n.T)

    q_vec: Optional[torch.Tensor] = None
    if text_cls is not None and ls > 0.0:
        if text_cls.dim() == 2:
            q_vec = text_cls[0].float()
        else:
            q_vec = text_cls.float()
        q_vec = q_vec.to(device).reshape(-1)
        q_vec = q_vec / q_vec.norm(p=2, dim=-1, keepdim=False).clamp_min(1e-12)
        cos_cq = (ctx_n * q_vec.unsqueeze(0)).sum(dim=-1, keepdim=True)
        if cross:
            cos_dq = (dom_n * q_vec.unsqueeze(0)).sum(dim=-1).unsqueeze(0)
            logits = logits + ls * cos_cq * cos_dq
        else:
            logits = logits + ls * cos_cq

    w = torch.softmax(logits / tau, dim=-1)
    delta = w.T @ ctx
    out = dom_f + inject * delta
    out = out / out.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    return out.to(dtype=dtype), dom_gidx


@torch.no_grad()
def _post_merge_protect_fused_semantic_recycle(
    flat_feats: torch.Tensor,
    flat_attn: torch.Tensor,
    flat_gidx: torch.Tensor,
    flat_frame_ids: torch.Tensor,
    flat_fused: torch.BoolTensor,
    target_k: int,
    post_merge_method: str,
    astra_config: AstraConfig,
    offset: int,
    seg_T: int,
    n_cand: int,
    sem_scores: Optional[torch.Tensor],
    leaf_feats: torch.Tensor,
    leaf_attn: torch.Tensor,
    leaf_gidx: torch.Tensor,
    leaf_frame_ids: torch.Tensor,
    frame_fluctuation: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Post-merge selection that always retains temporal-merge anchor tokens (counts>1 in merge),
    unless there are more fused anchors than target_k (then visual_guided_pruning within fused only).
    """
    device = flat_feats.device
    fused_idx = torch.where(flat_fused)[0]
    n_f = int(fused_idx.numel())
    if n_f == 0:
        raise RuntimeError("_post_merge_protect_fused_semantic_recycle: empty fused mask")

    if n_f >= target_k:
        sf, si = visual_guided_pruning_based_token_selection(
            features=flat_feats[fused_idx].unsqueeze(0),
            cls_attention=flat_attn[fused_idx].unsqueeze(0),
            num_retained_tokens=target_k,
        )
        si = si.squeeze(0).long()
        return flat_feats[fused_idx][si], flat_gidx[fused_idx][si]

    # Always keep all fused anchors; fill remaining budget from non-fused pool.
    fused_feat = flat_feats[fused_idx]
    fused_gidx = flat_gidx[fused_idx]
    rem_budget = int(target_k) - n_f
    rem_mask = ~flat_fused
    rem_idx = torch.where(rem_mask)[0]
    if rem_budget <= 0 or rem_idx.numel() == 0:
        return fused_feat, fused_gidx

    rf = flat_feats[rem_idx]
    ra = flat_attn[rem_idx]
    rg = flat_gidx[rem_idx]
    rfi = flat_frame_ids[rem_idx]

    pm = str(post_merge_method).strip().lower()
    if pm == "visual_guided_pruning":
        sf, si = visual_guided_pruning_based_token_selection(
            features=rf.unsqueeze(0),
            cls_attention=ra.unsqueeze(0),
            num_retained_tokens=min(rem_budget, int(rf.shape[0])),
        )
        si = si.squeeze(0).long()
        return torch.cat([fused_feat, rf[si]], dim=0), torch.cat([fused_gidx, rg[si]], dim=0)

    if pm == "visual_guided_cluster_pruning":
        rho = float(getattr(astra_config, "complementary_ratio", 0.30))
        rho = min(max(rho, 0.0), 0.9)
        k_rec = int(round(float(rem_budget) * rho))
        k_rec = max(0, min(k_rec, int(rem_budget) - 1))
        k_dom = int(rem_budget - k_rec)
        dom_feat, dom_idx = visual_guided_pruning_based_token_selection(
            features=rf.unsqueeze(0),
            cls_attention=ra.unsqueeze(0),
            num_retained_tokens=min(k_dom, int(rf.shape[0])),
        )
        dom_idx = dom_idx.squeeze(0).long()
        dom_feat = dom_feat.squeeze(0)
        dom_gidx = rg[dom_idx]
        if k_rec <= 0:
            return torch.cat([fused_feat, dom_feat], dim=0), torch.cat([fused_gidx, dom_gidx], dim=0)
        rem_mask2 = torch.ones(rf.shape[0], dtype=torch.bool, device=device)
        rem_mask2[dom_idx] = False
        rem_feats = rf[rem_mask2]
        rem_gidx = rg[rem_mask2]
        use_leaf = bool(getattr(astra_config, "post_merge_recycle_temporal_leaves", True))
        if use_leaf and int(leaf_feats.shape[0]) > 0:
            rem_feats = torch.cat([rem_feats, leaf_feats.to(rf.dtype)], dim=0)
            rem_gidx = torch.cat([rem_gidx, leaf_gidx], dim=0)
        if rem_feats.numel() == 0:
            comp_feat = rem_feats
            comp_gidx = rem_gidx
        else:
            comp_feat, comp_gidx = _post_merge_compress_remainder_visual_cluster(rem_feats, rem_gidx, k_rec)
        final_feat = torch.cat([fused_feat, dom_feat, comp_feat], dim=0)
        final_gidx = torch.cat([fused_gidx, dom_gidx, comp_gidx], dim=0)
        if final_feat.shape[0] > target_k:
            final_feat = final_feat[:target_k]
            final_gidx = final_gidx[:target_k]
        return final_feat, final_gidx

    if pm == "di_spm_tsr":
        if int(rem_budget) <= 0 or int(rf.shape[0]) == 0:
            return fused_feat, fused_gidx
        a = int(getattr(astra_config, "post_merge_di_parts", 8))
        b = int(getattr(astra_config, "post_merge_tsr_parts", 2))
        k_di, k_tsr = _split_two_way_budget(rem_budget, a, b)
        use_leaf = bool(getattr(astra_config, "post_merge_recycle_temporal_leaves", True))
        # DI-SPM: anchor+soft-merge on remainder pool (grid only), donors include leaves optionally.
        di_feat, di_gidx, donor_feat, donor_meta = _di_spm_fuse(
            k_di,
            rf,
            ra,
            rg,
            rfi,
            leaf_feats,
            leaf_attn,
            leaf_gidx,
            leaf_frame_ids,
            astra_config,
            use_leaf=use_leaf,
        )
        out_feat = [fused_feat, di_feat]
        out_gidx = [fused_gidx, di_gidx]
        # TSR: text-guided semantic recall from "weakly assigned" donors (visual-only miss set).
        if k_tsr > 0 and donor_feat.numel() > 0:
            text = getattr(astra_config, "text_cls_embed", None)
            donor_g = donor_meta[:, 0].long()
            donor_a = donor_meta[:, 1].to(dtype=rf.dtype)
            donor_f = donor_meta[:, 2].long()
            donor_mx = donor_meta[:, 3]
            low_ratio = float(getattr(astra_config, "post_merge_tsr_donor_low_ratio", 0.60))
            low_ratio = min(max(low_ratio, 0.0), 1.0)
            if low_ratio <= 0.0:
                pick_mask = torch.ones((donor_feat.shape[0],), device=device, dtype=torch.bool)
            else:
                n_take = max(1, int(round(float(donor_feat.shape[0]) * low_ratio)))
                idx = torch.topk(-donor_mx, k=min(n_take, int(donor_mx.numel()))).indices
                pick_mask = torch.zeros((donor_feat.shape[0],), device=device, dtype=torch.bool)
                pick_mask[idx] = True
            pool_feat = donor_feat[pick_mask]
            pool_g = donor_g[pick_mask]
            pool_a = donor_a[pick_mask]
            pool_f = donor_f[pick_mask]
            k_eff = min(int(k_tsr), int(pool_feat.shape[0]))
            if k_eff > 0:
                tsr_feat, tsr_gidx = _semantic_recycle_discrete_recycle_pick(
                    k_eff, pool_feat, pool_g, pool_a, pool_f, text, sem_scores, offset, seg_T, astra_config
                )
                out_feat.append(tsr_feat)
                out_gidx.append(tsr_gidx)
        final_feat = torch.cat(out_feat, dim=0)
        final_gidx = torch.cat(out_gidx, dim=0)
        if final_feat.shape[0] > target_k:
            final_feat = final_feat[:target_k]
            final_gidx = final_gidx[:target_k]
        return final_feat, final_gidx

    if pm == "visual_guided_contextual_pruning":
        if int(rem_budget) <= 0 or int(rf.shape[0]) == 0:
            return fused_feat, fused_gidx
        k_fill = min(int(rem_budget), int(rf.shape[0]))
        dom_feat, dom_idx = visual_guided_pruning_based_token_selection(
            features=rf.unsqueeze(0),
            cls_attention=ra.unsqueeze(0),
            num_retained_tokens=k_fill,
        )
        dom_idx = dom_idx.squeeze(0).long()
        dom_feat = dom_feat.squeeze(0)
        dom_gidx = rg[dom_idx]
        rem_mask2 = torch.ones(rf.shape[0], dtype=torch.bool, device=device)
        rem_mask2[dom_idx] = False
        rem_feats = rf[rem_mask2]
        use_leaf = bool(getattr(astra_config, "post_merge_recycle_temporal_leaves", True))
        if use_leaf and int(leaf_feats.shape[0]) > 0:
            ctx_cat = torch.cat([rem_feats, leaf_feats.to(rf.dtype)], dim=0) if rem_feats.numel() > 0 else leaf_feats.to(rf.dtype)
        else:
            ctx_cat = rem_feats
        text_z = getattr(astra_config, "text_cls_embed", None)
        zip_out, zip_g = _post_merge_zip_fuse_contextual(dom_feat, dom_gidx, ctx_cat, text_z, astra_config)
        final_feat = torch.cat([fused_feat, zip_out], dim=0)
        final_gidx = torch.cat([fused_gidx, zip_g], dim=0)
        if final_feat.shape[0] > target_k:
            final_feat = final_feat[:target_k]
            final_gidx = final_gidx[:target_k]
        return final_feat, final_gidx

    if pm == "semantic_recycle_pruning":
        if int(rem_budget) <= 0 or int(rf.shape[0]) == 0:
            return fused_feat, fused_gidx
        dyn = bool(getattr(astra_config, "post_merge_dynamic_visual_cluster_sem", False))
        if dyn:
            k_dom, k_visual_cluster, k_sem = _dynamic_attn_visual_cluster_sem_budget(
                rem_budget, frame_fluctuation, astra_config
            )
        else:
            a = int(getattr(astra_config, "post_merge_triple_dom_parts", 7))
            b = int(getattr(astra_config, "post_merge_triple_visual_cluster_parts", 1))
            c = int(getattr(astra_config, "post_merge_triple_sem_parts", 2))
            k_dom, k_visual_cluster, k_sem = _split_three_way_budget(rem_budget, a, b, c)
        text = getattr(astra_config, "text_cls_embed", None)
        # Allow k_dom==0 for pure-semantic ablations (d=0): skip visual dominant selection.
        if int(k_dom) > 0:
            dom_feat, dom_idx = visual_guided_pruning_based_token_selection(
                features=rf.unsqueeze(0),
                cls_attention=ra.unsqueeze(0),
                num_retained_tokens=min(k_dom, int(rf.shape[0])),
            )
            dom_idx = dom_idx.squeeze(0).long()
            dom_feat = dom_feat.squeeze(0)
            dom_gidx = rg[dom_idx]
            rem_mask2 = torch.ones(rf.shape[0], dtype=torch.bool, device=device)
            rem_mask2[dom_idx] = False
            rem_kept_feats = rf[rem_mask2]
            rem_kept_gidx = rg[rem_mask2]
            rem_kept_attn = ra[rem_mask2]
            rem_kept_fids = rfi[rem_mask2]
        else:
            dom_feat = torch.zeros((0, int(rf.shape[-1])), device=device, dtype=rf.dtype)
            dom_gidx = torch.zeros((0,), device=device, dtype=torch.long)
            dom_idx = torch.zeros((0,), device=device, dtype=torch.long)
            rem_kept_feats = rf
            rem_kept_gidx = rg
            rem_kept_attn = ra
            rem_kept_fids = rfi
        use_leaf = bool(getattr(astra_config, "post_merge_recycle_temporal_leaves", True))
        if use_leaf and int(leaf_feats.shape[0]) > 0:
            rem_feats = torch.cat([rem_kept_feats, leaf_feats.to(rf.dtype)], dim=0)
            rem_gidx = torch.cat([rem_kept_gidx, leaf_gidx], dim=0)
            rem_attn = torch.cat([rem_kept_attn, leaf_attn.to(ra.dtype)], dim=0)
            rem_frame_ids = torch.cat([rem_kept_fids, leaf_frame_ids], dim=0)
        else:
            rem_feats = rem_kept_feats
            rem_gidx = rem_kept_gidx
            rem_attn = rem_kept_attn
            rem_frame_ids = rem_kept_fids
        R = int(rem_feats.shape[0])
        fdim = int(rf.shape[-1])
        visual_cluster_feat = torch.zeros((0, fdim), device=device, dtype=rf.dtype)
        visual_cluster_gidx = torch.zeros((0,), device=device, dtype=torch.long)
        visual_cluster_centers = torch.zeros((0,), device=device, dtype=torch.long)
        sem_gidx_trace = torch.zeros((0,), device=device, dtype=torch.long)
        if k_visual_cluster > 0 and R > 0:
            k_visual_cluster_eff = min(k_visual_cluster, R)
            visual_cluster_feat, visual_cluster_gidx, visual_cluster_centers = _visual_cluster_aggregate_from_remainder(
                rem_feats, rem_gidx, k_visual_cluster_eff
            )
        chunks_f: List[torch.Tensor] = [fused_feat, dom_feat]
        chunks_g: List[torch.Tensor] = [fused_gidx, dom_gidx]
        if k_visual_cluster > 0:
            chunks_f.append(visual_cluster_feat)
            chunks_g.append(visual_cluster_gidx)
        sem_skip_dup = k_visual_cluster > 0 and R > 0 and int(visual_cluster_feat.shape[0]) == R
        if k_sem > 0 and R > 0 and not sem_skip_dup:
            if k_visual_cluster > 0 and int(visual_cluster_feat.shape[0]) < R:
                pool_mask = torch.ones(R, dtype=torch.bool, device=device)
                pool_mask[visual_cluster_centers] = False
                if bool(pool_mask.any().item()):
                    sem_feats = rem_feats[pool_mask]
                    sem_gidx = rem_gidx[pool_mask]
                    sem_attn = rem_attn[pool_mask]
                    sem_fids = rem_frame_ids[pool_mask]
                else:
                    sem_feats = rem_feats
                    sem_gidx = rem_gidx
                    sem_attn = rem_attn
                    sem_fids = rem_frame_ids
            else:
                sem_feats = rem_feats
                sem_gidx = rem_gidx
                sem_attn = rem_attn
                sem_fids = rem_frame_ids
            k_sem_eff = min(k_sem, int(sem_feats.shape[0]))
            sem_feat, sem_gidx_out = _semantic_recycle_discrete_recycle_pick(
                k_sem_eff,
                sem_feats,
                sem_gidx,
                sem_attn,
                sem_fids,
                text,
                sem_scores,
                offset,
                seg_T,
                astra_config,
            )
            chunks_f.append(sem_feat)
            chunks_g.append(sem_gidx_out)
            sem_gidx_trace = sem_gidx_out
        _trace_semantic_recycle_pruning_stats(
            astra_config=astra_config,
            branch="protect_fused",
            offset=offset,
            seg_T=seg_T,
            target_k=target_k,
            frame_fluctuation=frame_fluctuation,
            k_dom=k_dom,
            k_visual_cluster=k_visual_cluster,
            k_sem=k_sem,
            rem_kept_gidx=rem_kept_gidx,
            leaf_gidx=leaf_gidx,
            use_leaf=use_leaf,
            visual_cluster_centers=visual_cluster_centers,
            visual_cluster_gidx=visual_cluster_gidx,
            sem_gidx=sem_gidx_trace,
        )
        final_feat = torch.cat(chunks_f, dim=0)
        final_gidx = torch.cat(chunks_g, dim=0)
        if final_feat.shape[0] > target_k:
            final_feat = final_feat[:target_k]
            final_gidx = final_gidx[:target_k]
        elif final_feat.shape[0] < target_k and final_feat.shape[0] > 0:
            need = int(target_k - final_feat.shape[0])
            rem_pad = torch.ones(rf.shape[0], dtype=torch.bool, device=device)
            rem_pad[dom_idx] = False
            used = set(final_gidx.tolist())
            cand_idx = torch.where(rem_pad)[0]
            if cand_idx.numel() > 0:
                cand_attn = ra[cand_idx].float()
                order = torch.argsort(cand_attn, descending=True)
                extra: List[int] = []
                for ii in cand_idx[order].tolist():
                    if rg[ii].item() in used:
                        continue
                    extra.append(ii)
                    if len(extra) >= need:
                        break
                if extra:
                    ex = torch.tensor(extra, device=device, dtype=torch.long)
                    final_feat = torch.cat([final_feat, rf[ex]], dim=0)
                    final_gidx = torch.cat([final_gidx, rg[ex]], dim=0)
        return final_feat, final_gidx

    # semantic_recycle_visual_guided_pruning on remainder only (same structure as full-sequence semantic_recycle, budget = rem_budget).
    rho = float(getattr(astra_config, "complementary_ratio", 0.30))
    rho = min(max(rho, 0.0), 0.9)
    k_rec = int(round(float(rem_budget) * rho))
    k_rec = max(0, min(k_rec, int(rem_budget) - 1))
    k_dom = int(rem_budget - k_rec)
    text = getattr(astra_config, "text_cls_embed", None)

    dom_feat, dom_idx = visual_guided_pruning_based_token_selection(
        features=rf.unsqueeze(0),
        cls_attention=ra.unsqueeze(0),
        num_retained_tokens=min(k_dom, int(rf.shape[0])),
    )
    dom_idx = dom_idx.squeeze(0).long()
    dom_feat = dom_feat.squeeze(0)
    dom_gidx = rg[dom_idx]

    if k_rec <= 0:
        return torch.cat([fused_feat, dom_feat], dim=0), torch.cat([fused_gidx, dom_gidx], dim=0)

    rem_mask2 = torch.ones(rf.shape[0], dtype=torch.bool, device=device)
    rem_mask2[dom_idx] = False
    rem_feats = rf[rem_mask2]
    rem_gidx = rg[rem_mask2]
    rem_attn = ra[rem_mask2]
    rem_frame_ids = rfi[rem_mask2]
    use_leaf = bool(getattr(astra_config, "post_merge_recycle_temporal_leaves", True))
    if use_leaf and int(leaf_feats.shape[0]) > 0:
        rem_feats = torch.cat([rem_feats, leaf_feats.to(rf.dtype)], dim=0)
        rem_gidx = torch.cat([rem_gidx, leaf_gidx], dim=0)
        rem_attn = torch.cat([rem_attn, leaf_attn.to(ra.dtype)], dim=0)
        rem_frame_ids = torch.cat([rem_frame_ids, leaf_frame_ids], dim=0)
    n_r = int(rem_feats.shape[0])

    if rem_feats.numel() == 0:
        comp_feat = rem_feats
        comp_gidx = rem_gidx
    else:
        comp_feat, comp_gidx = _semantic_recycle_discrete_recycle_pick(
            min(k_rec, n_r),
            rem_feats,
            rem_gidx,
            rem_attn,
            rem_frame_ids,
            text,
            sem_scores,
            offset,
            seg_T,
            astra_config,
        )

    final_feat = torch.cat([fused_feat, dom_feat, comp_feat], dim=0)
    final_gidx = torch.cat([fused_gidx, dom_gidx, comp_gidx], dim=0)
    if final_feat.shape[0] > target_k:
        final_feat = final_feat[:target_k]
        final_gidx = final_gidx[:target_k]
    return final_feat, final_gidx


@torch.no_grad()
def conditional_dpp_select(
    features: torch.Tensor,
    relevance: torch.Tensor,
    k: int,
) -> torch.LongTensor:
    """
    Conditional DPP Fast-MAP selection, following CDPruner:
      L = diag(r) * S * diag(r), S is cosine similarity, r is relevance.
    """
    n, _ = features.shape
    if n == 0 or k <= 0:
        return torch.zeros((0,), dtype=torch.long, device=features.device)
    k = min(int(k), int(n))

    feats = features.float()
    feats = feats / feats.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    sim = torch.matmul(feats, feats.t())

    r = relevance.float().clamp_min(1e-6)
    r = (r - r.min() + 1e-6) / (r.max() - r.min()).clamp_min(1e-12)
    kernel = r.unsqueeze(1) * sim * r.unsqueeze(0)

    di2s = torch.diagonal(kernel).clone()
    cis = torch.zeros((k, n), device=features.device, dtype=torch.float32)
    selected = torch.empty((k,), dtype=torch.long, device=features.device)
    num = 0

    for t in range(k):
        j = torch.argmax(di2s)
        if float(di2s[j].item()) <= 0.0:
            break
        selected[t] = j
        eis = (kernel[j] - torch.einsum("u,un->n", cis[:t, j], cis[:t])) / (
            di2s[j].sqrt() + 1e-8
        )
        cis[t] = eis
        di2s -= eis**2
        di2s[j] = -float("inf")
        num += 1

    if num == 0:
        return torch.topk(r, k=1).indices.to(dtype=torch.long)
    selected = selected[:num]
    return torch.unique(torch.sort(selected).values)


def _segment_lengths_from_boolean_runs(mask: torch.BoolTensor) -> torch.Tensor:
    """Convert a per-frame boolean mask into segment lengths by grouping consecutive runs."""
    if mask.numel() == 0:
        return torch.tensor([], dtype=torch.long, device=mask.device)
    # Boundaries where value changes.
    changes = torch.where(mask[1:] != mask[:-1])[0] + 1
    boundaries = torch.cat(
        [torch.tensor([0], device=mask.device), changes, torch.tensor([mask.numel()], device=mask.device)]
    )
    lengths = boundaries[1:] - boundaries[:-1]
    return lengths.to(dtype=torch.long)


def _allocate_integer_budgets(
    weights: torch.Tensor,
    total_budget: int,
    min_per_frame: int,
    max_per_frame: int,
) -> torch.LongTensor:
    """Allocate integer per-frame budgets from normalized weights."""
    T = int(weights.numel())
    if T == 0:
        return torch.zeros(0, dtype=torch.long, device=weights.device)

    min_per_frame = max(0, min(int(min_per_frame), int(max_per_frame)))
    base = torch.full((T,), min_per_frame, dtype=torch.long, device=weights.device)
    rem = int(total_budget) - int(base.sum().item())
    if rem <= 0:
        return base

    frac = (weights * float(rem)).clamp_min(0.0)
    add = torch.floor(frac).to(torch.long)
    budgets = base + add

    if max_per_frame > 0:
        budgets = budgets.clamp_max(int(max_per_frame))

    used = int((budgets - base).sum().item())
    left = rem - used
    if left <= 0:
        return budgets

    score = frac - torch.floor(frac)
    order = torch.argsort(score, descending=True)
    for idx in order.tolist():
        if left <= 0:
            break
        if budgets[idx].item() < int(max_per_frame):
            budgets[idx] += 1
            left -= 1

    # Last resort: fill remaining budget to frames with room.
    if left > 0:
        for idx in range(T):
            if left <= 0:
                break
            room = int(max_per_frame) - int(budgets[idx].item())
            if room <= 0:
                continue
            take = min(room, left)
            budgets[idx] += take
            left -= take
    return budgets


def _reconcile_integer_vector_sum(
    vec: torch.Tensor,
    target: int,
    min_pf: int,
    max_pf: int,
) -> None:
    """In-place clamp then adjust vec so sum(vec)==target (best-effort)."""
    vec.clamp_(min_pf, max_pf)
    max_iter = int(vec.numel()) * 2000
    it = 0
    while it < max_iter:
        it += 1
        cur = int(vec.sum().item())
        diff = target - cur
        if diff == 0:
            return
        if diff > 0:
            best_i = -1
            best_room = -1
            for i in range(int(vec.numel())):
                room = max_pf - int(vec[i].item())
                if room > best_room:
                    best_room = room
                    best_i = i
            if best_i < 0 or best_room <= 0:
                return
            vec[best_i] += 1
        else:
            best_i = -1
            best_room = -1
            for i in range(int(vec.numel())):
                room = int(vec[i].item()) - min_pf
                if room > best_room:
                    best_room = room
                    best_i = i
            if best_i < 0 or best_room <= 0:
                return
            vec[best_i] -= 1


def _apply_uniform_recycle_story_per_frame_budget(
    per_frame_budget: torch.Tensor,
    segment_lengths: torch.Tensor,
    sem_scores: torch.Tensor,
    astra_config: AstraConfig,
    *,
    min_pf: int,
    max_pf: int,
) -> torch.Tensor:
    """
    Paper-facing per-frame budget narrative: start from uniform integer budgets, then within each
    DynamicSegment segment perform a zero-sum rebalance that moves k_sem counts from non-recycle frames onto
    recycle frames using the same frame split as _semantic_recycle_select_recycle_frames + k_sem integer split
    as _semantic_recycle_discrete_recycle_pick (post_merge_triple_* semantic part).

    Note: stv_guided_dynamic_budget_allocation only uses sum(per_frame_budget) as segment target_k for temporal merge;
    per-frame breakdown is for logging / story. When post_merge_dynamic_visual_cluster_sem is True, skip
    rebalance (k_sem is fluctuation-dependent at post-merge time).
    """
    out = per_frame_budget.clone()
    if bool(getattr(astra_config, "post_merge_dynamic_visual_cluster_sem", False)):
        return out
    device = out.device
    num_frames = int(out.numel())
    off = 0
    a = int(getattr(astra_config, "post_merge_triple_dom_parts", 7))
    b = int(getattr(astra_config, "post_merge_triple_visual_cluster_parts", 1))
    c = int(getattr(astra_config, "post_merge_triple_sem_parts", 2))
    for _seg_i in range(int(segment_lengths.shape[0])):
        L = int(segment_lengths[_seg_i].item())
        if L <= 0:
            continue
        if off + L > num_frames:
            break
        sl = slice(off, off + L)
        seg_vec = out[sl].clone()
        target_s = int(seg_vec.sum().item())
        _k_dom, _k_visual_cluster, k_sem = _split_three_way_budget(target_s, a, b, c)
        if k_sem <= 0:
            off += L
            continue
        seg_sem = sem_scores[off : off + L].to(device=device, dtype=torch.float32).flatten()
        if int(seg_sem.numel()) < L:
            seg_sem = torch.ones(L, device=device, dtype=torch.float32)
        elif int(seg_sem.numel()) > L:
            seg_sem = seg_sem[:L]
        top_frames, _ = _semantic_recycle_recycle_frames_with_stv_coverage_schedule(
            seg_sem,
            L,
            device,
            astra_config,
        )
        nf = int(top_frames.numel())
        if nf <= 0:
            off += L
            continue
        base_k = k_sem // nf
        rem = k_sem - base_k * nf
        delta = torch.zeros(L, dtype=torch.long, device=device)
        for rank in range(nf):
            li = int(top_frames[rank].item())
            delta[li] += base_k + (1 if rank < rem else 0)
        rec_set = {int(top_frames[i].item()) for i in range(nf)}
        non_rec = [i for i in range(L) if i not in rec_set]
        if not non_rec:
            off += L
            continue
        sub = k_sem // len(non_rec)
        rem_sub = k_sem - sub * len(non_rec)
        for j, t in enumerate(non_rec):
            delta[t] -= sub + (1 if j < rem_sub else 0)
        new_seg = seg_vec + delta
        _reconcile_integer_vector_sum(new_seg, target_s, min_pf, max_pf)
        out[sl] = new_seg
        off += L
    return out


def astra_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    astra_config: AstraConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_frames, num_visual_tokens, feat_dim = video_features.shape
    astra_config.average_stv = None
    astra_config.last_recycle_adapt_stats = None

    # FastV-only / ablation path: keep all pooled vision tokens, skip any vision-side pruning/merging.
    if bool(getattr(astra_config, "disable_vision_compression", False)):
        global_indices = torch.arange(num_frames * num_visual_tokens, dtype=torch.long, device=video_features.device)
        tokens = video_features.reshape(num_frames * num_visual_tokens, feat_dim)
        astra_config.visual_token_length = int(tokens.shape[0])
        return tokens, global_indices

    # ContextualMerge (post-pooling) per-frame pruning: text-agnostic, no temporal merge, no segmentation.
    frame_selection_method = getattr(astra_config, "frame_selection_method", "none")
    if str(frame_selection_method).strip().lower() == "contextual_merge":
        keep_ratio = float(getattr(astra_config, "contextual_merge_retention_ratio", 0.10))
        dom_abs = int(getattr(astra_config, "contextual_merge_dominant_tokens_per_frame", 0))
        ctx_abs = int(getattr(astra_config, "contextual_merge_contextual_tokens_per_frame", 0))
        ratio_dom = int(getattr(astra_config, "contextual_merge_dom_ctx_ratio_dom", 54))
        ratio_ctx = int(getattr(astra_config, "contextual_merge_dom_ctx_ratio_ctx", 10))

        # Global indices map (T, N) -> flat indices into the original pooled tokens.
        global_indices = torch.arange(num_frames * num_visual_tokens, dtype=torch.long, device=video_features.device)
        frame_global_indices = global_indices.view(num_frames, num_visual_tokens)

        kept_tokens = []
        kept_gidx = []
        for t in range(num_frames):
            feats = video_features[t]  # (N, D)
            attn = cls_attention[t]  # (N,)

            if dom_abs > 0 or ctx_abs > 0:
                dom_k = max(0, min(dom_abs, num_visual_tokens))
                ctx_k = max(0, min(ctx_abs, num_visual_tokens - dom_k))
            else:
                total_k = max(1, min(int(math.ceil(num_visual_tokens * keep_ratio)), num_visual_tokens))
                parts = max(1, ratio_dom + ratio_ctx)
                dom_k = int(round(total_k * (ratio_dom / parts)))
                dom_k = max(1, min(dom_k, total_k))
                ctx_k = max(0, total_k - dom_k)

            # Dominant: top attention-to-CLS tokens.
            dom_idx = torch.topk(attn, k=dom_k, largest=True).indices
            dom_idx = dom_idx.sort().values
            mask = torch.ones(num_visual_tokens, dtype=torch.bool, device=video_features.device)
            mask[dom_idx] = False

            if ctx_k > 0:
                # Contextual: ContextualMerge-style "zip" aggregation.
                # 1) pick contextual anchors from remaining tokens (uniform stride, like original impl)
                # 2) assign other remaining tokens to nearest anchor by cosine similarity of features
                # 3) aggregate (mean) and add to anchor features
                rem_idx = torch.where(mask)[0]
                rem_feats = feats[rem_idx]  # (R, D)
                R = int(rem_feats.shape[0])
                ctx_k_eff = max(0, min(int(ctx_k), R))
                if ctx_k_eff <= 0:
                    sel_idx = dom_idx
                else:
                    step = max(1, R // ctx_k_eff)
                    anchor_pos = torch.arange(0, R, step, device=video_features.device, dtype=torch.long)[:ctx_k_eff]
                    anchor_feats = rem_feats[anchor_pos]  # (C, D)

                    if ctx_k_eff < R:
                        pos_all = torch.arange(R, device=video_features.device, dtype=torch.long)
                        non_anchor_mask = torch.ones(R, device=video_features.device, dtype=torch.bool)
                        non_anchor_mask[anchor_pos] = False
                        to_merge_pos = pos_all[non_anchor_mask]  # (M,)
                        to_merge_feats = rem_feats[to_merge_pos]  # (M, D)

                        # Cosine similarity (M, C)
                        anchor_norm = F.normalize(anchor_feats, p=2, dim=-1)
                        merge_norm = F.normalize(to_merge_feats, p=2, dim=-1)
                        sim = merge_norm @ anchor_norm.transpose(0, 1)
                        assign = sim.argmax(dim=1)  # (M,)

                        # Sum + count per anchor to compute mean
                        agg_sum = torch.zeros_like(anchor_feats)
                        agg_sum.index_add_(0, assign, to_merge_feats)
                        counts = torch.zeros((ctx_k_eff,), device=video_features.device, dtype=anchor_feats.dtype)
                        ones = torch.ones((to_merge_feats.shape[0],), device=video_features.device, dtype=anchor_feats.dtype)
                        counts.index_add_(0, assign, ones)
                        counts = counts.clamp_min(1.0).unsqueeze(-1)
                        anchor_feats = anchor_feats + agg_sum / counts

                    ctx_idx = rem_idx[anchor_pos].sort().values
                    sel_idx = torch.cat([dom_idx, ctx_idx], dim=0).sort().values
            else:
                sel_idx = dom_idx

            # Use aggregated contextual tokens instead of raw anchor tokens.
            if ctx_k > 0 and ("anchor_pos" in locals()) and (ctx_k_eff > 0):
                kept = torch.cat([feats[dom_idx], anchor_feats], dim=0)
                kept_tokens.append(kept)
                kept_gidx.append(torch.cat([frame_global_indices[t][dom_idx], frame_global_indices[t][rem_idx[anchor_pos]]], dim=0))
            else:
                kept_tokens.append(feats[sel_idx])
                kept_gidx.append(frame_global_indices[t][sel_idx])

        final_tokens = torch.cat(kept_tokens, dim=0)
        final_indices = torch.cat(kept_gidx, dim=0)
        # Keep order by original token index.
        order = final_indices.argsort()
        final_tokens = final_tokens[order]
        final_indices = final_indices[order]
        astra_config.visual_token_length = int(final_tokens.shape[0])
        return final_tokens, final_indices

    # 1. Partition video into segments.
    # Default path: DynamicSegment / no-seg.
    important_mask = None
    frame_classes = None  # 2=important, 1=context, 0=irrelevant
    frame_similarity = None
    frame_selection_method = getattr(astra_config, "frame_selection_method", "none")
    use_frame_selection = (
        (frame_selection_method == "dpp" and getattr(astra_config, "frame_top_p", 0.0) > 0)
        or frame_selection_method == "stv_frame_gumbel"
        or _is_stv_guided_dynamic_budget_allocation(frame_selection_method)
    )
    if use_frame_selection:
        # Optional path: frame-level selection first, then segment by runs of important vs non-important frames.
        if astra_config.frame_selection_method == "dpp":
            qf = getattr(astra_config, "stv_budget_frame_features", None) or getattr(
                astra_config, "stv_frame_frame_features", None
            )
            qt = getattr(astra_config, "stv_budget_text_cls_embed", None) or getattr(
                astra_config, "stv_frame_text_cls_embed", None
            )
            if qf is not None and qt is not None:
                frame_feats = qf
                # CDPruner-style: use only text relevance as quality (SigLIP joint space).
                frame_q = text_relevance_minmax(frame_feats, qt)
            else:
                frame_feats, _ = build_frame_features_and_quality(video_features, cls_attention)
                frame_q = torch.ones(num_frames, device=video_features.device, dtype=video_features.dtype)
            sel = dpp_select_frames_top_p(
                frame_features=frame_feats,
                frame_quality=frame_q,
                top_p=float(astra_config.frame_top_p),
                max_selected=num_frames,
            )
            important_mask = sel.important_mask
            astra_config.last_important_frame_count = int(sel.selected_indices.numel())
            segment_lengths = _segment_lengths_from_boolean_runs(important_mask)
        elif frame_selection_method == "stv_frame_gumbel":
            qf = getattr(astra_config, "stv_budget_frame_features", None) or getattr(
                astra_config, "stv_frame_frame_features", None
            )
            qt = getattr(astra_config, "stv_budget_text_cls_embed", None) or getattr(
                astra_config, "stv_frame_text_cls_embed", None
            )
            if qf is not None and qt is not None:
                frame_feats = qf
                sel = stv_frame_gumbel_select_frames(
                    frame_features=frame_feats,
                    text_cls_embed=qt,
                    top_p=float(getattr(astra_config, "frame_top_p", 0.25)),
                    tau=float(getattr(astra_config, "stv_frame_tau", 0.8)),
                )
                frame_similarity = text_relevance_logits(frame_feats, qt)
                frame_classes = stv_frame_rank_to_three_classes(
                    rank_scores=frame_similarity,
                    important_ratio=float(getattr(astra_config, "stv_frame_important_ratio", 0.25)),
                    context_upper_ratio=float(getattr(astra_config, "stv_frame_context_upper_ratio", 0.75)),
                )
                important_mask = frame_classes == 2
                astra_config.last_important_frame_count = int(important_mask.sum().item())
                # Keep original adjacency-based segmentation for stv_frame_gumbel.
                segment_lengths = None
            else:
                # No SigLIP query/frame features -> skip frame-level gumbel selection.
                segment_lengths = None
        else:
            # Unknown method -> fall back to original behavior.
            segment_lengths = None
    else:
        segment_lengths = None

    if segment_lengths is None:
        if astra_config.do_segment:
            segment_lengths = segment(
                video_features=video_features.mean(1),
                segment_threshold=astra_config.segment_threshold,
                min_segment_num=astra_config.min_segment_num,
                complementary_segment=astra_config.complementary_segment,
            )
        else:
            segment_lengths = torch.tensor([num_frames], dtype=torch.long, device=video_features.device)

    # stv_guided_dynamic_budget_allocation：整段视频为单一 segment，seg_T = 采样帧数 num_frames。
    if _is_stv_guided_dynamic_budget_allocation(frame_selection_method):
        segment_lengths = torch.tensor([num_frames], dtype=torch.long, device=video_features.device)

    num_segments = segment_lengths.shape[0]
    global_indices = torch.arange(num_frames * num_visual_tokens, dtype=torch.long, device=video_features.device)

    # 2. Apply Attention and Diversity-based Token Selection(VGP) or DPP-based selection.
    base_per_frame_budget = math.ceil(num_visual_tokens * astra_config.retention_ratio * astra_config.expansion)

    # Aggressive mode:
    # - important frames: keep more tokens via in-frame diversity selection.
    # - non-important frames:
    #   (a) default: keep a small number of explicit tokens;
    #   (b) optional: merge all non-important tokens into important-frame anchors.
    # - skip TAM + spatial clustering completely
    if important_mask is not None and getattr(astra_config, "aggressive_keyframe_mode", False):
        total_budget = base_per_frame_budget * num_frames
        non_imp_frames = (~important_mask).sum().item()
        imp_frames = important_mask.sum().item()
        non_imp_k = max(1, int(getattr(astra_config, "non_important_tokens_per_frame", 1)))
        do_merge_non_important = bool(getattr(astra_config, "merge_non_important_to_important", False))
        if do_merge_non_important:
            # When non-important frames are merged into important anchors, allocate
            # the whole budget to important frames.
            reserved_non_imp = 0
        else:
            reserved_non_imp = non_imp_frames * non_imp_k
        remaining = max(0, total_budget - reserved_non_imp)

        if imp_frames > 0:
            imp_base = remaining // imp_frames
            imp_rem = remaining % imp_frames
        else:
            imp_base = 0
            imp_rem = 0
        imp_base = min(imp_base, num_visual_tokens)
        per_non_imp = min(non_imp_k, num_visual_tokens)

        frame_global_indices = global_indices.view(num_frames, num_visual_tokens)
        important_tokens = []
        important_indices = []
        important_frame_ids = []
        non_important_frames = []

        imp_rank = 0
        for frame_idx in range(num_frames):
            frame_feats = video_features[frame_idx : frame_idx + 1]  # (1, N, D)
            frame_attn = cls_attention[frame_idx : frame_idx + 1]  # (1, N)
            frame_gidx = frame_global_indices[frame_idx]  # (N,)

            if bool(important_mask[frame_idx].item()):
                per_imp = min(imp_base + (1 if imp_rank < imp_rem else 0), num_visual_tokens)
                imp_rank += 1
                if per_imp <= 0:
                    continue
                # Match segment_compression: honor token_selection_method + cls_attention when needed.
                method = getattr(astra_config, "token_selection_method", "visual_guided_pruning")
                if method == "divprune":
                    method = "div"
                if method not in TOKEN_SELECTION_BY_NAME:
                    method = "visual_guided_pruning"
                needs_cls = method in ("attn", "visual_guided_pruning", "visual_guided_pruning", "dpp")
                additional_kwargs = {"cls_attention": frame_attn} if needs_cls else {}
                selected_features, selected_indices = TOKEN_SELECTION_BY_NAME[method](
                    features=frame_feats,
                    num_retained_tokens=per_imp,
                    **additional_kwargs,
                )
                selected_indices = selected_indices.squeeze(0)
                selected_features = selected_features.squeeze(0)
                selected_gidx = frame_gidx[selected_indices]
                important_tokens.append(selected_features)
                important_indices.append(selected_gidx)
                important_frame_ids.append(
                    torch.full((selected_features.shape[0],), frame_idx, device=video_features.device, dtype=torch.long)
                )
            else:
                non_important_frames.append((frame_feats.squeeze(0), frame_attn.squeeze(0), frame_gidx, frame_idx))

        if len(important_tokens) == 0:
            # Fallback: if no important frame selected, use previous non-important behavior.
            all_tokens = []
            all_indices = []
            for frame_feats, frame_attn, frame_gidx, _ in non_important_frames:
                keep_k = per_non_imp
                topk = torch.topk(frame_attn, k=keep_k, largest=True).indices
                selected_indices = topk.sort().values
                all_tokens.append(frame_feats[selected_indices])
                all_indices.append(frame_gidx[selected_indices])
            if len(all_tokens) == 0:
                fallback_idx = torch.tensor([0], dtype=torch.long, device=video_features.device)
                all_tokens = [video_features[0][fallback_idx]]
                all_indices = [frame_global_indices[0][fallback_idx]]
            final_tokens = torch.cat(all_tokens, dim=0)
            final_indices = torch.cat(all_indices, dim=0)
        else:
            final_tokens = torch.cat(important_tokens, dim=0)
            final_indices = torch.cat(important_indices, dim=0)

            if do_merge_non_important and len(non_important_frames) > 0:
                # Merge non-important frame tokens into important-frame anchors by
                # nearest cosine similarity.
                anchor_tokens = final_tokens
                anchor_norm = anchor_tokens / anchor_tokens.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                merged_sum = torch.zeros_like(anchor_tokens)
                merged_count = torch.zeros(anchor_tokens.shape[0], device=anchor_tokens.device, dtype=anchor_tokens.dtype)

                for frame_feats, _, _, frame_idx in non_important_frames:
                    src_norm = frame_feats / frame_feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    sim = torch.matmul(src_norm, anchor_norm.t())

                    # Temporal preference: non-important frames are merged first into
                    # nearby important anchors.
                    if len(important_frame_ids) > 0:
                        anchor_frame_ids = torch.cat(important_frame_ids, dim=0).to(sim.device)
                        time_dist = (anchor_frame_ids.unsqueeze(0) - frame_idx).abs().to(sim.dtype)
                        sim = sim - 0.01 * time_dist

                    assign_idx = sim.argmax(dim=1)
                    merged_sum.index_add_(0, assign_idx, frame_feats)
                    merged_count.index_add_(
                        0,
                        assign_idx,
                        torch.ones(assign_idx.shape[0], device=assign_idx.device, dtype=merged_count.dtype),
                    )

                merged_count = merged_count.clamp_min(1.0).unsqueeze(-1)
                merged_avg = merged_sum / merged_count
                merge_beta = float(getattr(astra_config, "non_important_merge_beta", 0.5))
                merge_beta = min(max(merge_beta, 0.0), 1.0)
                final_tokens = merge_beta * anchor_tokens + (1.0 - merge_beta) * merged_avg
            elif (not do_merge_non_important) and len(non_important_frames) > 0:
                # Backward-compatible path: keep small explicit tokens for non-important frames.
                other_tokens = []
                other_indices = []
                for frame_feats, frame_attn, frame_gidx, _ in non_important_frames:
                    keep_k = per_non_imp
                    topk = torch.topk(frame_attn, k=keep_k, largest=True).indices
                    selected_indices = topk.sort().values
                    other_tokens.append(frame_feats[selected_indices])
                    other_indices.append(frame_gidx[selected_indices])
                if len(other_tokens) > 0:
                    final_tokens = torch.cat([final_tokens, torch.cat(other_tokens, dim=0)], dim=0)
                    final_indices = torch.cat([final_indices, torch.cat(other_indices, dim=0)], dim=0)

        sorted_indices = final_indices.argsort()
        sorted_tokens = final_tokens[sorted_indices]
        sorted_global_indices = final_indices[sorted_indices]
        astra_config.visual_token_length = sorted_tokens.shape[0]
        return sorted_tokens, sorted_global_indices

    all_segment_features = []
    all_segment_indices = []
    offset = 0
    per_frame_budget_vector = None
    class_per_frame_budget = None
    # stv_guided_dynamic_budget_allocation：第一部分动态 budget = 均匀底稿 + 与第三部分语义回收帧/k_sem 一致的零和重分配。
    if _is_stv_guided_dynamic_budget_allocation(frame_selection_method):
        frame_feats = getattr(astra_config, "stv_budget_frame_features", None)
        if frame_feats is None:
            frame_feats = getattr(astra_config, "stv_frame_frame_features", None)
        if frame_feats is None:
            frame_feats, _ = build_frame_features_and_quality(video_features, cls_attention)
        _dvec, mu_d = adjacent_change_mean(frame_feats)
        _mu_raw = getattr(astra_config, "stv_budget_mu_d_raw", None)
        if bool(getattr(astra_config, "average_stv_from_raw_visual", False)) and _mu_raw is not None:
            astra_config.average_stv = float(_mu_raw)
        else:
            astra_config.average_stv = float(mu_d.detach().float().cpu().item())
        text_cls = getattr(astra_config, "stv_budget_text_cls_embed", None)
        if text_cls is None:
            text_cls = getattr(astra_config, "stv_frame_text_cls_embed", None)
        if text_cls is not None:
            sem_scores = text_relevance_minmax(frame_feats, text_cls)
        else:
            sem_scores = torch.ones(num_frames, device=video_features.device, dtype=video_features.dtype)
        sem_probs = sem_scores
        sem_mix = torch.tensor(0.0, device=video_features.device, dtype=video_features.dtype)

        total_budget = base_per_frame_budget * num_frames
        min_pf = int(getattr(astra_config, "stv_budget_min_tokens_per_frame", 1))

        per_frame_budget_vector = _allocate_integer_budgets(
            weights=torch.full((num_frames,), 1.0 / max(1, num_frames), device=video_features.device),
            total_budget=total_budget,
            min_per_frame=min_pf,
            max_per_frame=num_visual_tokens,
        )
        per_frame_budget_vector = _apply_uniform_recycle_story_per_frame_budget(
            per_frame_budget_vector,
            segment_lengths,
            sem_scores,
            astra_config,
            min_pf=min_pf,
            max_pf=num_visual_tokens,
        )
        important_mask_sb = torch.zeros(num_frames, dtype=torch.bool, device=video_features.device)
        regime = "uniform_recycle_story"
        eps = float(getattr(astra_config, "stv_budget_eps", 1e-6))
        temperature = float(getattr(astra_config, "stv_budget_temperature", 1.0))
        mu_threshold = float(getattr(astra_config, "stv_budget_mu_threshold", 0.10))
        mu_full = float(getattr(astra_config, "stv_budget_mu_full", 0.60))
        if text_cls is not None:
            _, sem_probs, _, sem_mix = stv_budget_weights(
                frame_feats,
                text_cls,
                eps=eps,
                temperature=temperature,
                mu_threshold=mu_threshold,
                mu_full=mu_full,
            )
            sem_scores = sem_probs
        else:
            sem_mix = torch.tensor(0.0, device=video_features.device, dtype=video_features.dtype)

        b = per_frame_budget_vector.float()
        mean_b = float(b.mean().item()) if b.numel() > 0 else 0.0
        std_b = float(b.std(unbiased=False).item()) if b.numel() > 0 else 0.0
        cv = std_b / max(mean_b, 1e-8)
        # Text CLS vs per-frame visual diagnostics (SigLIP joint space).
        qlogits = None
        qprobs = None
        cls_qlogits = None
        cls_off_qlogits = None
        head_off_cos = None
        try:
            qtext = getattr(astra_config, "stv_budget_text_cls_embed", None)
            if qtext is None:
                qtext = getattr(astra_config, "stv_frame_text_cls_embed", None)
            if qtext is not None:
                qlogits = text_relevance_logits(frame_feats, qtext).float()  # (T,)
                qtemp = float(getattr(astra_config, "stv_budget_temperature", 1.0))
                qtemp = max(qtemp, 1e-6)
                qprobs = torch.softmax(qlogits / qtemp, dim=0)
                vcls = getattr(astra_config, "stv_budget_visual_cls_embed", None)
                if vcls is None:
                    vcls = getattr(astra_config, "stv_frame_visual_cls_embed", None)
                if vcls is not None and vcls.shape[0] == frame_feats.shape[0]:
                    cls_qlogits = torch.sum(vcls.float() * qtext.float().unsqueeze(0), dim=-1)  # (T,)
                vcls_off = getattr(astra_config, "stv_budget_visual_cls_embed_official", None)
                if vcls_off is None:
                    vcls_off = getattr(astra_config, "stv_frame_visual_cls_embed_official", None)
                if vcls_off is not None and vcls_off.shape[0] == frame_feats.shape[0]:
                    cls_off_qlogits = torch.sum(vcls_off.float() * qtext.float().unsqueeze(0), dim=-1)  # (T,)
                vcls_head = getattr(astra_config, "stv_budget_visual_cls_embed_head", None)
                if vcls_head is None:
                    vcls_head = getattr(astra_config, "stv_frame_visual_cls_embed_head", None)
                if vcls_head is not None and vcls_off is not None and vcls_head.shape == vcls_off.shape:
                    head_off_cos = torch.sum(vcls_head.float() * vcls_off.float(), dim=-1)  # (T,)
        except Exception:
            qlogits = None
            qprobs = None
            cls_qlogits = None
            cls_off_qlogits = None
            head_off_cos = None

        _mu_d_raw = getattr(astra_config, "stv_budget_mu_d_raw", None)
        astra_config.last_budget_stats = {
            "min": int(per_frame_budget_vector.min().item()) if per_frame_budget_vector.numel() > 0 else 0,
            "max": int(per_frame_budget_vector.max().item()) if per_frame_budget_vector.numel() > 0 else 0,
            "mean": mean_b,
            "std": std_b,
            "cv": cv,
            # average_stv：驱动 recycle_stv / 映射的标量（可为 raw 或 joint，见 average_stv_from_raw_visual）
            "average_stv": float(astra_config.average_stv) if astra_config.average_stv is not None else float(mu_d.item()),
            "average_stv_joint": float(mu_d.item()),
            "mu_d_raw": float(_mu_d_raw) if _mu_d_raw is not None else None,
            "sem_mix": float(sem_mix.item()) if sem_mix is not None else 0.0,
            "semantic_mean": float(sem_scores.mean().item()),
            "uniform_like": cv <= float(getattr(astra_config, "stv_budget_uniform_cv_threshold", 0.08)),
            "per_frame_compression": True,
            "regime": regime,
            "uniform_recycle_story": bool(regime == "uniform_recycle_story"),
            "important_frames": int(important_mask_sb.sum().item()) if important_mask_sb is not None else 0,
            "has_text_cls": bool(text_cls is not None),
            "qlogits_min": float(qlogits.min().item()) if qlogits is not None and qlogits.numel() > 0 else None,
            "qlogits_max": float(qlogits.max().item()) if qlogits is not None and qlogits.numel() > 0 else None,
            "qlogits_mean": float(qlogits.mean().item()) if qlogits is not None and qlogits.numel() > 0 else None,
            "qlogits_std": float(qlogits.std(unbiased=False).item()) if qlogits is not None and qlogits.numel() > 0 else None,
            "qprobs_top1": float(qprobs.max().item()) if qprobs is not None and qprobs.numel() > 0 else None,
            "vcls_qlogits_min": float(cls_qlogits.min().item()) if cls_qlogits is not None and cls_qlogits.numel() > 0 else None,
            "vcls_qlogits_max": float(cls_qlogits.max().item()) if cls_qlogits is not None and cls_qlogits.numel() > 0 else None,
            "vcls_qlogits_std": float(cls_qlogits.std(unbiased=False).item()) if cls_qlogits is not None and cls_qlogits.numel() > 0 else None,
            "official_vcls_qlogits_min": float(cls_off_qlogits.min().item()) if cls_off_qlogits is not None and cls_off_qlogits.numel() > 0 else None,
            "official_vcls_qlogits_max": float(cls_off_qlogits.max().item()) if cls_off_qlogits is not None and cls_off_qlogits.numel() > 0 else None,
            "official_vcls_qlogits_std": float(cls_off_qlogits.std(unbiased=False).item()) if cls_off_qlogits is not None and cls_off_qlogits.numel() > 0 else None,
            "head_official_vcls_cos_mean": float(head_off_cos.mean().item()) if head_off_cos is not None and head_off_cos.numel() > 0 else None,
            "head_official_vcls_cos_std": float(head_off_cos.std(unbiased=False).item()) if head_off_cos is not None and head_off_cos.numel() > 0 else None,
        }
    elif important_mask is not None:
        if frame_selection_method == "stv_frame_gumbel" and frame_classes is not None and frame_similarity is not None:
            segment_types = []
            type_frame_count = {0: 0, 1: 0, 2: 0}
            seg_offset = 0
            for seg_idx in range(num_segments):
                seg_len = int(segment_lengths[seg_idx].item())
                seg_scores = frame_similarity[seg_offset : seg_offset + seg_len]
                seg_classes = frame_classes[seg_offset : seg_offset + seg_len]
                rep_local_idx = int(torch.argmax(seg_scores).item())
                seg_type = int(seg_classes[rep_local_idx].item())  # 2/1/0
                segment_types.append(seg_type)
                type_frame_count[seg_type] += seg_len
                seg_offset += seg_len

            total_budget = base_per_frame_budget * num_frames
            irr_per_frame = int(getattr(astra_config, "stv_frame_irrelevant_tokens_per_frame", 1))
            irr_per_frame = max(1, min(irr_per_frame, num_visual_tokens))
            irr_total = type_frame_count[0] * irr_per_frame
            rem_budget = max(0, total_budget - irr_total)

            imp_ratio = float(getattr(astra_config, "important_budget_ratio", 0.7))
            imp_ratio = min(max(imp_ratio, 0.0), 1.0)
            imp_total = int(round(rem_budget * imp_ratio))
            ctx_total = rem_budget - imp_total

            if type_frame_count[2] == 0:
                ctx_total += imp_total
                imp_total = 0
            if type_frame_count[1] == 0:
                imp_total += ctx_total
                ctx_total = 0

            imp_per_frame = max(1, imp_total // max(1, type_frame_count[2])) if type_frame_count[2] > 0 else 0
            ctx_per_frame = max(1, ctx_total // max(1, type_frame_count[1])) if type_frame_count[1] > 0 else 0
            imp_per_frame = min(imp_per_frame, num_visual_tokens)
            ctx_per_frame = min(ctx_per_frame, num_visual_tokens)
            class_per_frame_budget = {2: imp_per_frame, 1: ctx_per_frame, 0: irr_per_frame}
        else:
            imp_frames = int(important_mask.sum().item())
            other_frames = int((~important_mask).sum().item())
            total_budget = base_per_frame_budget * num_frames
            imp_ratio = float(getattr(astra_config, "important_budget_ratio", 0.7))
            imp_ratio = min(max(imp_ratio, 0.0), 1.0)
            imp_total = int(round(total_budget * imp_ratio))
            other_total = total_budget - imp_total
            # Avoid division by zero; if a group is empty, give everything to the other group.
            if imp_frames == 0:
                imp_per_frame = 0
                other_per_frame = max(1, other_total // max(1, other_frames))
            elif other_frames == 0:
                other_per_frame = 0
                imp_per_frame = max(1, imp_total // max(1, imp_frames))
            else:
                imp_per_frame = max(1, imp_total // imp_frames)
                other_per_frame = max(1, other_total // other_frames)
            # Clamp to available tokens per frame.
            imp_per_frame = min(imp_per_frame, num_visual_tokens)
            other_per_frame = min(other_per_frame, num_visual_tokens)
    else:
        imp_per_frame = None
        other_per_frame = None

    for seg_idx in range(num_segments):
        seg_len = segment_lengths[seg_idx]
        segment_features = video_features[offset : offset + seg_len]
        segment_cls_attention = cls_attention[offset : offset + seg_len]
        segment_global_indices = global_indices.view(num_frames, num_visual_tokens)[offset : offset + seg_len]

        # stv_guided_dynamic_budget_allocation: 动态逐帧预算 + 全帧 temporal backward merge + 剩余 token 用 visual_guided_pruning 剪枝。
        if _is_stv_guided_dynamic_budget_allocation(frame_selection_method) and per_frame_budget_vector is not None:
            frame_budgets = per_frame_budget_vector[offset : offset + seg_len]
            seg_T = int(seg_len.item())

            # stv_frame 逐帧预算之和 = merge 后最终保留的 visual token 数（由 visual_guided_pruning 选出）。
            target_k = int(frame_budgets.sum().item())
            target_k = max(1, min(target_k, seg_T * num_visual_tokens))

            # 1) 全部空间 token 参与后向合并（当前帧 token 与前一帧局部/全局候选比相似度，>=阈值则合并）。
            # 效率：无 soft lower bound 时只跑一次 merge；有 soft LB 且第一次合并已满足 target_k 时复用该结果，避免同阈值下重复 merge。
            tmerge_gh = int(getattr(astra_config, "vision_spatial_h", 0) or 0)
            tmerge_gw = int(getattr(astra_config, "vision_spatial_w", 0) or 0)
            token_mask = torch.ones(
                (seg_T, num_visual_tokens),
                dtype=torch.bool,
                device=video_features.device,
            )
            seg_before = segment_features.clone()
            base_thr = float(getattr(astra_config, "temporal_threshold", 0.8))
            skip_lt = float(getattr(astra_config, "temporal_skip_if_frame_sim_lt", -1.0))
            skip_gt = float(getattr(astra_config, "temporal_skip_if_frame_sim_gt", 1.0))
            dbg = bool(getattr(astra_config, "temporal_debug_stats", False))
            tstats: Optional[dict] = {} if dbg else None
            use_soft_lb = bool(getattr(astra_config, "temporal_soft_lower_bound", False))
            full_prev = bool(getattr(astra_config, "temporal_merge_prev_frame_global", False))
            same_cell = bool(getattr(astra_config, "temporal_merge_same_cell_only", False)) and (not full_prev)
            final_merge_thr = base_thr
            keep_mask: torch.BoolTensor
            fused_mask: torch.BoolTensor

            # Speed-only: direct same-cell merge across frames (skip similarity matching/backward-merge entirely).
            if bool(getattr(astra_config, "temporal_direct_same_cell_merge", False)) and seg_T > 1:
                seg_work = seg_before.clone()
                merged = seg_work.mean(dim=0, keepdim=False)  # (N, D)
                seg_work[0] = merged
                if seg_T > 1:
                    seg_work[1:] = seg_work[0].unsqueeze(0).expand(seg_T - 1, -1, -1)
                keep_mask = torch.zeros((seg_T, num_visual_tokens), dtype=torch.bool, device=seg_work.device)
                keep_mask[0, :] = True
                fused_mask = ~keep_mask
                segment_features.copy_(seg_work)
                final_merge_thr = 1.0
                target_k = max(1, min(target_k, num_visual_tokens))
            elif not use_soft_lb:
                seg_work = seg_before.clone()
                keep_mask, fused_mask = temporal_backward_merge(
                    video_features=seg_work,
                    token_mask=token_mask,
                    sim_threshold=base_thr,
                    skip_if_frame_sim_lt=skip_lt,
                    skip_if_frame_sim_gt=skip_gt,
                    stats=tstats,
                    grid_h=tmerge_gh,
                    grid_w=tmerge_gw,
                    same_cell_only=same_cell,
                    full_prev_frame=full_prev,
                )
                segment_features.copy_(seg_work)
            else:
                seg_probe = seg_before.clone()
                km0, fused0 = temporal_backward_merge(
                    video_features=seg_probe,
                    token_mask=token_mask,
                    sim_threshold=base_thr,
                    skip_if_frame_sim_lt=skip_lt,
                    skip_if_frame_sim_gt=skip_gt,
                    grid_h=tmerge_gh,
                    grid_w=tmerge_gw,
                    same_cell_only=same_cell,
                    full_prev_frame=full_prev,
                )
                kept0 = int(km0.sum().item())

                if kept0 < int(target_k):
                    iters = int(getattr(astra_config, "temporal_soft_lower_bound_iters", 8))
                    iters = max(1, min(iters, 20))
                    lo = float(base_thr)
                    hi = 1.0
                    best_mask = km0
                    best_kept = kept0
                    best_thr = float(base_thr)
                    for _ in range(iters):
                        mid = (lo + hi) * 0.5
                        mid_mask, _ = temporal_backward_merge(
                            video_features=seg_before.clone(),
                            token_mask=token_mask,
                            sim_threshold=float(mid),
                            skip_if_frame_sim_lt=skip_lt,
                            skip_if_frame_sim_gt=skip_gt,
                            grid_h=tmerge_gh,
                            grid_w=tmerge_gw,
                            same_cell_only=same_cell,
                            full_prev_frame=full_prev,
                        )
                        mid_kept = int(mid_mask.sum().item())
                        if mid_kept < int(target_k):
                            lo = mid
                            if mid_kept > best_kept:
                                best_kept = mid_kept
                                best_mask = mid_mask
                                best_thr = float(mid)
                        else:
                            hi = mid
                            best_kept = mid_kept
                            best_mask = mid_mask
                            best_thr = float(mid)
                    final_merge_thr = best_thr
                    seg_work = seg_before.clone()
                    keep_mask, fused_mask = temporal_backward_merge(
                        video_features=seg_work,
                        token_mask=token_mask,
                        sim_threshold=float(final_merge_thr),
                        skip_if_frame_sim_lt=skip_lt,
                        skip_if_frame_sim_gt=skip_gt,
                        stats=tstats,
                        grid_h=tmerge_gh,
                        grid_w=tmerge_gw,
                        same_cell_only=same_cell,
                        full_prev_frame=full_prev,
                    )
                    segment_features.copy_(seg_work)
                else:
                    keep_mask = km0
                    fused_mask = fused0
                    segment_features.copy_(seg_probe)
                    if dbg and tstats is not None:
                        _fill_tmerge_debug_stats(
                            tstats,
                            seg_T,
                            num_visual_tokens,
                            token_mask,
                            keep_mask,
                            fused_mask,
                            float(final_merge_thr),
                            skip_lt,
                            skip_gt,
                        )

            if dbg and tstats is not None:
                # Compact one-line stats per segment to validate temporal merging is happening.
                print(
                    "[Astra][tmerge] "
                    f"T={tstats.get('T')} N={tstats.get('N')} "
                    f"thr={tstats.get('sim_threshold'):.3f} "
                    f"skip_lt={tstats.get('skip_if_frame_sim_lt'):.3f} skip_gt={tstats.get('skip_if_frame_sim_gt'):.3f} "
                    f"skipped_pairs={tstats.get('skipped_pairs')} "
                    f"kept={tstats.get('kept_tokens')} merged={tstats.get('merged_tokens')} "
                    f"target_k={int(target_k)}"
                )

            # 2) 合并后剩余 token + 对应 cls_attention（供 visual_guided_pruning）。
            sttm_feats: List[torch.Tensor] = []
            sttm_gidx: List[torch.Tensor] = []
            sttm_attn: List[torch.Tensor] = []
            sttm_frame_ids: List[torch.Tensor] = []
            sttm_fused: List[torch.Tensor] = []

            for t in range(seg_T):
                idx = torch.where(keep_mask[t])[0]
                if idx.numel() == 0:
                    continue
                sttm_feats.append(segment_features[t, idx])
                sttm_gidx.append(segment_global_indices[t, idx])
                attn_t = segment_cls_attention[t, idx].float().clamp_min(1e-6)
                sttm_attn.append(attn_t.to(dtype=segment_features.dtype))
                sttm_fused.append(fused_mask[t, idx].to(dtype=torch.bool))
                sttm_frame_ids.append(
                    torch.full(
                        (idx.numel(),),
                        int(offset + t),
                        device=video_features.device,
                        dtype=torch.long,
                    )
                )

            if len(sttm_feats) > 0:
                flat_feats = torch.cat(sttm_feats, dim=0)
                flat_gidx = torch.cat(sttm_gidx, dim=0)
                flat_attn = torch.cat(sttm_attn, dim=0).to(flat_feats.device)
                flat_fused = torch.cat(sttm_fused, dim=0)
                flat_frame_ids = torch.cat(sttm_frame_ids, dim=0)
            else:
                flat_feats = torch.zeros((0, feat_dim), device=video_features.device, dtype=video_features.dtype)
                flat_gidx = torch.zeros((0,), device=video_features.device, dtype=torch.long)
                flat_attn = torch.zeros((0,), device=video_features.device, dtype=video_features.dtype)
                flat_fused = torch.zeros((0,), dtype=torch.bool, device=video_features.device)
                flat_frame_ids = torch.zeros((0,), device=video_features.device, dtype=torch.long)

            leaf_feats, leaf_attn, leaf_gidx, leaf_frame_ids = _collect_temporal_merge_leaf_tokens(
                segment_features,
                segment_cls_attention,
                segment_global_indices,
                token_mask,
                keep_mask,
                offset,
                seg_T,
            )

            # 3) Post-merge selection. Keep separate from baseline token_selection_method:
            #    - post_merge_selection_method="visual_guided_pruning": merged -> visual_guided_pruning keep B tokens
            #    - post_merge_selection_method="semantic_recycle_visual_guided_pruning": dominant visual_guided_pruning + recycle; recycle splits k_rec across
            #      frames chosen by post_merge_recycle_frame_policy (default top_quarter by sem_scores vs text CLS).
            #    - post_merge_selection_method="visual_guided_cluster_pruning": dominant visual_guided_pruning + VisualCluster-kNN merge on remainder (Astra-style).
            #    - post_merge_selection_method="visual_guided_contextual_pruning": visual_guided_pruning -> target_k dominants + ContextualMerge-style contextual fusion.
            #    - post_merge_selection_method="semantic_recycle_pruning": visual_guided_pruning -> VisualCluster on remainder -> semantic recycle last.
            #    - post_merge_selection_method="di_spm_tsr": DI-SPM (anchor+soft-merge) + text-guided semantic recall.
            frame_fluctuation = _estimate_segment_frame_fluctuation(
                segment_features, segment_cls_attention, token_mask
            )
            n_cand = int(flat_feats.shape[0])
            if n_cand > 0 and target_k > 0 and n_cand > target_k:
                pm = str(getattr(astra_config, "post_merge_selection_method", "visual_guided_pruning")).strip().lower()
                if pm not in (
                    "visual_guided_pruning",
                    "semantic_recycle_visual_guided_pruning",
                    "visual_guided_cluster_pruning",
                    "visual_guided_contextual_pruning",
                    "semantic_recycle_pruning",
                    "di_spm_tsr",
                    "di-spm-tsr",
                ):
                    pm = "visual_guided_pruning"
                if pm == "di-spm-tsr":
                    pm = "di_spm_tsr"

                protect = bool(getattr(astra_config, "post_merge_protect_fused_anchors", False))
                if protect and bool(flat_fused.any().item()):
                    final_feat, final_gidx = _post_merge_protect_fused_semantic_recycle(
                        flat_feats,
                        flat_attn,
                        flat_gidx,
                        flat_frame_ids,
                        flat_fused,
                        target_k,
                        pm,
                        astra_config,
                        offset,
                        seg_T,
                        n_cand,
                        sem_scores,
                        leaf_feats,
                        leaf_attn,
                        leaf_gidx,
                        leaf_frame_ids,
                        frame_fluctuation,
                    )
                    sort_idx = torch.argsort(final_gidx)
                    segment_features = final_feat[sort_idx]
                    segment_global_indices = final_gidx[sort_idx]
                elif pm == "visual_guided_pruning":
                    sel_feat, sel_idx = _post_merge_visual_guided_pruning_select(
                        astra_config=astra_config,
                        features=flat_feats.unsqueeze(0),
                        cls_attention=flat_attn.unsqueeze(0),
                        num_retained_tokens=min(target_k, n_cand),
                    )
                    sel_idx = sel_idx.squeeze(0)
                    sel_feat = sel_feat.squeeze(0)
                    sel_gidx = flat_gidx[sel_idx]
                    sort_idx = torch.argsort(sel_gidx)
                    segment_features = sel_feat[sort_idx]
                    segment_global_indices = sel_gidx[sort_idx]
                    # Done.
                elif pm == "visual_guided_cluster_pruning":
                    rho = float(getattr(astra_config, "complementary_ratio", 0.30))
                    rho = min(max(rho, 0.0), 0.9)
                    k_rec = int(round(float(target_k) * rho))
                    k_rec = max(0, min(k_rec, int(target_k) - 1))
                    k_dom = int(target_k - k_rec)

                    dom_feat, dom_idx = _post_merge_visual_guided_pruning_select(
                        astra_config=astra_config,
                        features=flat_feats.unsqueeze(0),
                        cls_attention=flat_attn.unsqueeze(0),
                        num_retained_tokens=min(k_dom, n_cand),
                    )
                    dom_idx = dom_idx.squeeze(0)
                    dom_feat = dom_feat.squeeze(0)
                    dom_gidx = flat_gidx[dom_idx]

                    if k_rec <= 0:
                        final_feat = dom_feat
                        final_gidx = dom_gidx
                    else:
                        rem_mask = torch.ones(n_cand, dtype=torch.bool, device=flat_feats.device)
                        rem_mask[dom_idx] = False
                        rem_kept_feats = flat_feats[rem_mask]
                        rem_kept_gidx = flat_gidx[rem_mask]
                        use_leaf = bool(getattr(astra_config, "post_merge_recycle_temporal_leaves", True))
                        if use_leaf and int(leaf_feats.shape[0]) > 0:
                            rem_feats = torch.cat([rem_kept_feats, leaf_feats.to(rem_kept_feats.dtype)], dim=0)
                            rem_gidx = torch.cat([rem_kept_gidx, leaf_gidx], dim=0)
                        else:
                            rem_feats = rem_kept_feats
                            rem_gidx = rem_kept_gidx

                        if rem_feats.numel() == 0:
                            comp_feat = rem_feats
                            comp_gidx = rem_gidx
                        else:
                            comp_feat, comp_gidx = _post_merge_compress_remainder_visual_cluster(
                                rem_feats, rem_gidx, k_rec
                            )

                        final_feat = torch.cat([dom_feat, comp_feat], dim=0)
                        final_gidx = torch.cat([dom_gidx, comp_gidx], dim=0)

                    if final_feat.shape[0] > target_k:
                        final_feat = final_feat[:target_k]
                        final_gidx = final_gidx[:target_k]
                    elif final_feat.shape[0] < target_k and final_feat.shape[0] > 0:
                        need = int(target_k - final_feat.shape[0])
                        rem_mask2 = torch.ones(n_cand, dtype=torch.bool, device=flat_feats.device)
                        rem_mask2[dom_idx] = False
                        used = set(final_gidx.tolist())
                        cand_idx = torch.where(rem_mask2)[0]
                        if cand_idx.numel() > 0:
                            cand_attn = flat_attn[cand_idx].float()
                            order = torch.argsort(cand_attn, descending=True)
                            extra = []
                            for ii in cand_idx[order].tolist():
                                if flat_gidx[ii].item() in used:
                                    continue
                                extra.append(ii)
                                if len(extra) >= need:
                                    break
                            if extra:
                                extra = torch.tensor(extra, device=flat_feats.device, dtype=torch.long)
                                final_feat = torch.cat([final_feat, flat_feats[extra]], dim=0)
                                final_gidx = torch.cat([final_gidx, flat_gidx[extra]], dim=0)

                    sort_idx = torch.argsort(final_gidx)
                    segment_features = final_feat[sort_idx]
                    segment_global_indices = final_gidx[sort_idx]
                elif pm == "semantic_recycle_pruning":
                    dyn = bool(getattr(astra_config, "post_merge_dynamic_visual_cluster_sem", False))
                    if dyn:
                        k_dom, k_visual_cluster, k_sem = _dynamic_attn_visual_cluster_sem_budget(
                            target_k, frame_fluctuation, astra_config
                        )
                    else:
                        a = int(getattr(astra_config, "post_merge_triple_dom_parts", 7))
                        b = int(getattr(astra_config, "post_merge_triple_visual_cluster_parts", 1))
                        c = int(getattr(astra_config, "post_merge_triple_sem_parts", 2))
                        k_dom, k_visual_cluster, k_sem = _split_three_way_budget(target_k, a, b, c)
                    text = getattr(astra_config, "text_cls_embed", None)
                    # Allow k_dom==0 for pure-semantic ablations (d=0): skip visual dominant selection.
                    if int(k_dom) > 0:
                        dom_feat, dom_idx = _post_merge_visual_guided_pruning_select(
                            astra_config=astra_config,
                            features=flat_feats.unsqueeze(0),
                            cls_attention=flat_attn.unsqueeze(0),
                            num_retained_tokens=min(k_dom, n_cand),
                        )
                        dom_idx = dom_idx.squeeze(0)
                        dom_feat = dom_feat.squeeze(0)
                        dom_gidx = flat_gidx[dom_idx]
                    else:
                        dom_feat = torch.zeros(
                            (0, int(flat_feats.shape[-1])),
                            device=flat_feats.device,
                            dtype=flat_feats.dtype,
                        )
                        dom_idx = torch.zeros((0,), device=flat_feats.device, dtype=torch.long)
                        dom_gidx = torch.zeros((0,), device=flat_feats.device, dtype=torch.long)

                    rem_mask = torch.ones(n_cand, dtype=torch.bool, device=flat_feats.device)
                    rem_mask[dom_idx] = False
                    rem_kept_feats = flat_feats[rem_mask]
                    rem_kept_gidx = flat_gidx[rem_mask]
                    rem_kept_attn = flat_attn[rem_mask]
                    rem_kept_frame_ids = flat_frame_ids[rem_mask]
                    use_leaf = bool(getattr(astra_config, "post_merge_recycle_temporal_leaves", True))
                    if use_leaf and int(leaf_feats.shape[0]) > 0:
                        rem_feats = torch.cat([rem_kept_feats, leaf_feats.to(rem_kept_feats.dtype)], dim=0)
                        rem_gidx = torch.cat([rem_kept_gidx, leaf_gidx], dim=0)
                        rem_attn = torch.cat([rem_kept_attn, leaf_attn.to(rem_kept_attn.dtype)], dim=0)
                        rem_frame_ids = torch.cat([rem_kept_frame_ids, leaf_frame_ids], dim=0)
                    else:
                        rem_feats = rem_kept_feats
                        rem_gidx = rem_kept_gidx
                        rem_attn = rem_kept_attn
                        rem_frame_ids = rem_kept_frame_ids

                    R = int(rem_feats.shape[0])
                    feat_dim = int(flat_feats.shape[-1])
                    visual_cluster_feat = torch.zeros((0, feat_dim), device=flat_feats.device, dtype=flat_feats.dtype)
                    visual_cluster_gidx = torch.zeros((0,), device=flat_feats.device, dtype=torch.long)
                    visual_cluster_centers = torch.zeros((0,), device=flat_feats.device, dtype=torch.long)
                    sem_gidx_trace = torch.zeros((0,), device=flat_feats.device, dtype=torch.long)
                    if k_visual_cluster > 0 and R > 0:
                        k_visual_cluster_eff = min(k_visual_cluster, R)
                        visual_cluster_feat, visual_cluster_gidx, visual_cluster_centers = _visual_cluster_aggregate_from_remainder(
                            rem_feats, rem_gidx, k_visual_cluster_eff
                        )

                    chunks_f: List[torch.Tensor] = [dom_feat]
                    chunks_g: List[torch.Tensor] = [dom_gidx]
                    if k_visual_cluster > 0:
                        chunks_f.append(visual_cluster_feat)
                        chunks_g.append(visual_cluster_gidx)

                    sem_skip_dup = k_visual_cluster > 0 and R > 0 and int(visual_cluster_feat.shape[0]) == R
                    if k_sem > 0 and R > 0 and not sem_skip_dup:
                        if k_visual_cluster > 0 and int(visual_cluster_feat.shape[0]) < R:
                            pool_mask = torch.ones(R, dtype=torch.bool, device=rem_feats.device)
                            pool_mask[visual_cluster_centers] = False
                            if bool(pool_mask.any().item()):
                                sem_feats = rem_feats[pool_mask]
                                sem_gidx = rem_gidx[pool_mask]
                                sem_attn = rem_attn[pool_mask]
                                sem_fids = rem_frame_ids[pool_mask]
                            else:
                                sem_feats = rem_feats
                                sem_gidx = rem_gidx
                                sem_attn = rem_attn
                                sem_fids = rem_frame_ids
                        else:
                            sem_feats = rem_feats
                            sem_gidx = rem_gidx
                            sem_attn = rem_attn
                            sem_fids = rem_frame_ids
                        k_sem_eff = min(k_sem, int(sem_feats.shape[0]))
                        sem_feat, sem_gidx_out = _semantic_recycle_discrete_recycle_pick(
                            k_sem_eff,
                            sem_feats,
                            sem_gidx,
                            sem_attn,
                            sem_fids,
                            text,
                            sem_scores,
                            offset,
                            seg_T,
                            astra_config,
                        )
                        chunks_f.append(sem_feat)
                        chunks_g.append(sem_gidx_out)
                        sem_gidx_trace = sem_gidx_out

                    _trace_semantic_recycle_pruning_stats(
                        astra_config=astra_config,
                        branch="main",
                        offset=offset,
                        seg_T=seg_T,
                        target_k=target_k,
                        frame_fluctuation=frame_fluctuation,
                        k_dom=k_dom,
                        k_visual_cluster=k_visual_cluster,
                        k_sem=k_sem,
                        rem_kept_gidx=rem_kept_gidx,
                        leaf_gidx=leaf_gidx,
                        use_leaf=use_leaf,
                        visual_cluster_centers=visual_cluster_centers,
                        visual_cluster_gidx=visual_cluster_gidx,
                        sem_gidx=sem_gidx_trace,
                    )

                    final_feat = torch.cat(chunks_f, dim=0)
                    final_gidx = torch.cat(chunks_g, dim=0)

                    if final_feat.shape[0] > target_k:
                        final_feat = final_feat[:target_k]
                        final_gidx = final_gidx[:target_k]
                    elif final_feat.shape[0] < target_k and final_feat.shape[0] > 0:
                        need = int(target_k - final_feat.shape[0])
                        rem_mask2 = torch.ones(n_cand, dtype=torch.bool, device=flat_feats.device)
                        rem_mask2[dom_idx] = False
                        used = set(final_gidx.tolist())
                        cand_idx = torch.where(rem_mask2)[0]
                        if cand_idx.numel() > 0:
                            cand_attn = flat_attn[cand_idx].float()
                            order = torch.argsort(cand_attn, descending=True)
                            extra: List[int] = []
                            for ii in cand_idx[order].tolist():
                                if flat_gidx[ii].item() in used:
                                    continue
                                extra.append(ii)
                                if len(extra) >= need:
                                    break
                            if extra:
                                ex = torch.tensor(extra, device=flat_feats.device, dtype=torch.long)
                                final_feat = torch.cat([final_feat, flat_feats[ex]], dim=0)
                                final_gidx = torch.cat([final_gidx, flat_gidx[ex]], dim=0)

                    sort_idx = torch.argsort(final_gidx)
                    segment_features = final_feat[sort_idx]
                    segment_global_indices = final_gidx[sort_idx]
                elif pm == "di_spm_tsr":
                    # DI-SPM + TSR: unify diversity+importance into one spatial prune-merge operator,
                    # then apply text-guided semantic recall on the donor pool (visual-only miss set).
                    a = int(getattr(astra_config, "post_merge_di_parts", 8))
                    b = int(getattr(astra_config, "post_merge_tsr_parts", 2))
                    k_di, k_tsr = _split_two_way_budget(target_k, a, b)
                    use_leaf = bool(getattr(astra_config, "post_merge_recycle_temporal_leaves", True))
                    di_feat, di_gidx, donor_feat, donor_meta = _di_spm_fuse(
                        k_di,
                        flat_feats,
                        flat_attn,
                        flat_gidx,
                        flat_frame_ids,
                        leaf_feats,
                        leaf_attn,
                        leaf_gidx,
                        leaf_frame_ids,
                        astra_config,
                        use_leaf=use_leaf,
                    )
                    chunks_f: List[torch.Tensor] = [di_feat]
                    chunks_g: List[torch.Tensor] = [di_gidx]

                    if k_tsr > 0 and donor_feat.numel() > 0:
                        text = getattr(astra_config, "text_cls_embed", None)
                        donor_g = donor_meta[:, 0].long()
                        donor_a = donor_meta[:, 1].to(dtype=flat_feats.dtype)
                        donor_f = donor_meta[:, 2].long()
                        donor_mx = donor_meta[:, 3]
                        low_ratio = float(getattr(astra_config, "post_merge_tsr_donor_low_ratio", 0.60))
                        low_ratio = min(max(low_ratio, 0.0), 1.0)
                        if low_ratio <= 0.0:
                            pick_mask = torch.ones((donor_feat.shape[0],), device=donor_feat.device, dtype=torch.bool)
                        else:
                            n_take = max(1, int(round(float(donor_feat.shape[0]) * low_ratio)))
                            idx = torch.topk(-donor_mx, k=min(n_take, int(donor_mx.numel()))).indices
                            pick_mask = torch.zeros((donor_feat.shape[0],), device=donor_feat.device, dtype=torch.bool)
                            pick_mask[idx] = True
                        pool_feat = donor_feat[pick_mask]
                        pool_g = donor_g[pick_mask]
                        pool_a = donor_a[pick_mask]
                        pool_f = donor_f[pick_mask]
                        k_eff = min(int(k_tsr), int(pool_feat.shape[0]))
                        if k_eff > 0:
                            tsr_feat, tsr_gidx = _semantic_recycle_discrete_recycle_pick(
                                k_eff,
                                pool_feat,
                                pool_g,
                                pool_a,
                                pool_f,
                                text,
                                sem_scores,
                                offset,
                                seg_T,
                                astra_config,
                            )
                            chunks_f.append(tsr_feat)
                            chunks_g.append(tsr_gidx)

                    final_feat = torch.cat(chunks_f, dim=0)
                    final_gidx = torch.cat(chunks_g, dim=0)
                    if final_feat.shape[0] > target_k:
                        final_feat = final_feat[:target_k]
                        final_gidx = final_gidx[:target_k]
                    elif final_feat.shape[0] < target_k and final_feat.shape[0] > 0:
                        # Pad by top remaining attention (rare).
                        need = int(target_k - final_feat.shape[0])
                        used = set(final_gidx.tolist())
                        cand_attn = flat_attn.float()
                        order = torch.argsort(cand_attn, descending=True)
                        extra: List[int] = []
                        for ii in order.tolist():
                            if flat_gidx[ii].item() in used:
                                continue
                            extra.append(ii)
                            if len(extra) >= need:
                                break
                        if extra:
                            ex = torch.tensor(extra, device=flat_feats.device, dtype=torch.long)
                            final_feat = torch.cat([final_feat, flat_feats[ex]], dim=0)
                            final_gidx = torch.cat([final_gidx, flat_gidx[ex]], dim=0)
                    sort_idx = torch.argsort(final_gidx)
                    segment_features = final_feat[sort_idx]
                    segment_global_indices = final_gidx[sort_idx]
                elif pm == "visual_guided_contextual_pruning":
                    dom_feat, dom_idx = visual_guided_pruning_based_token_selection(
                        features=flat_feats.unsqueeze(0),
                        cls_attention=flat_attn.unsqueeze(0),
                        num_retained_tokens=min(target_k, n_cand),
                    )
                    dom_idx = dom_idx.squeeze(0)
                    dom_feat = dom_feat.squeeze(0)
                    dom_gidx = flat_gidx[dom_idx]
                    rem_mask = torch.ones(n_cand, dtype=torch.bool, device=flat_feats.device)
                    rem_mask[dom_idx] = False
                    rem_kept = flat_feats[rem_mask]
                    use_leaf = bool(getattr(astra_config, "post_merge_recycle_temporal_leaves", True))
                    if use_leaf and int(leaf_feats.shape[0]) > 0:
                        ctx_cat = (
                            torch.cat([rem_kept, leaf_feats.to(rem_kept.dtype)], dim=0)
                            if rem_kept.numel() > 0
                            else leaf_feats.to(flat_feats.dtype)
                        )
                    else:
                        ctx_cat = rem_kept
                    text_z = getattr(astra_config, "text_cls_embed", None)
                    final_feat, final_gidx = _post_merge_zip_fuse_contextual(
                        dom_feat, dom_gidx, ctx_cat, text_z, astra_config
                    )
                    sort_idx = torch.argsort(final_gidx)
                    segment_features = final_feat[sort_idx]
                    segment_global_indices = final_gidx[sort_idx]
                elif pm == "semantic_recycle_visual_guided_pruning":
                    # semantic_recycle_visual_guided_pruning: dominant visual_guided_pruning + semantic recycle (no entropy gating).
                    # - Recycle frames: post_merge_recycle_frame_policy (see _semantic_recycle_select_recycle_frames).
                    # - Split k_rec across those frames; per frame take token-text cosine top-k.
                    text = getattr(astra_config, "text_cls_embed", None)

                    rho = float(getattr(astra_config, "complementary_ratio", 0.30))
                    rho = min(max(rho, 0.0), 0.9)
                    k_rec = int(round(float(target_k) * rho))
                    k_rec = max(0, min(k_rec, int(target_k) - 1))
                    k_dom = int(target_k - k_rec)

                    # --- dominant: visual_guided_pruning ---
                    dom_feat, dom_idx = visual_guided_pruning_based_token_selection(
                        features=flat_feats.unsqueeze(0),
                        cls_attention=flat_attn.unsqueeze(0),
                        num_retained_tokens=min(k_dom, n_cand),
                    )
                    dom_idx = dom_idx.squeeze(0)
                    dom_feat = dom_feat.squeeze(0)
                    dom_gidx = flat_gidx[dom_idx]

                    # Remaining candidates.
                    if k_rec <= 0:
                        final_feat = dom_feat
                        final_gidx = dom_gidx
                    else:
                        rem_mask = torch.ones(n_cand, dtype=torch.bool, device=flat_feats.device)
                        rem_mask[dom_idx] = False
                        rem_kept_feats = flat_feats[rem_mask]
                        rem_kept_gidx = flat_gidx[rem_mask]
                        rem_kept_attn = flat_attn[rem_mask]
                        rem_kept_frame_ids = flat_frame_ids[rem_mask]
                        use_leaf = bool(getattr(astra_config, "post_merge_recycle_temporal_leaves", True))
                        if use_leaf and int(leaf_feats.shape[0]) > 0:
                            rem_feats = torch.cat([rem_kept_feats, leaf_feats.to(rem_kept_feats.dtype)], dim=0)
                            rem_gidx = torch.cat([rem_kept_gidx, leaf_gidx], dim=0)
                            rem_attn = torch.cat([rem_kept_attn, leaf_attn.to(rem_kept_attn.dtype)], dim=0)
                            rem_frame_ids = torch.cat([rem_kept_frame_ids, leaf_frame_ids], dim=0)
                        else:
                            rem_feats = rem_kept_feats
                            rem_gidx = rem_kept_gidx
                            rem_attn = rem_kept_attn
                            rem_frame_ids = rem_kept_frame_ids

                        if rem_feats.numel() == 0:
                            comp_feat = rem_feats
                            comp_gidx = rem_gidx
                        else:
                            comp_feat, comp_gidx = _semantic_recycle_discrete_recycle_pick(
                                min(k_rec, int(rem_feats.shape[0])),
                                rem_feats,
                                rem_gidx,
                                rem_attn,
                                rem_frame_ids,
                                text,
                                sem_scores,
                                offset,
                                seg_T,
                                astra_config,
                            )

                        # Combine.
                        final_feat = torch.cat([dom_feat, comp_feat], dim=0)
                        final_gidx = torch.cat([dom_gidx, comp_gidx], dim=0)

                    # Ensure exact budget (in case of any edge-case mismatch).
                    if final_feat.shape[0] > target_k:
                        final_feat = final_feat[:target_k]
                        final_gidx = final_gidx[:target_k]
                    elif final_feat.shape[0] < target_k and final_feat.shape[0] > 0:
                        # Pad by top remaining attention tokens (rare).
                        need = int(target_k - final_feat.shape[0])
                        rem_mask2 = torch.ones(n_cand, dtype=torch.bool, device=flat_feats.device)
                        rem_mask2[dom_idx] = False
                        used = set(final_gidx.tolist())
                        cand_idx = torch.where(rem_mask2)[0]
                        if cand_idx.numel() > 0:
                            cand_attn = flat_attn[cand_idx].float()
                            order = torch.argsort(cand_attn, descending=True)
                            extra = []
                            for ii in cand_idx[order].tolist():
                                if flat_gidx[ii].item() in used:
                                    continue
                                extra.append(ii)
                                if len(extra) >= need:
                                    break
                            if extra:
                                extra = torch.tensor(extra, device=flat_feats.device, dtype=torch.long)
                                final_feat = torch.cat([final_feat, flat_feats[extra]], dim=0)
                                final_gidx = torch.cat([final_gidx, flat_gidx[extra]], dim=0)

                    sort_idx = torch.argsort(final_gidx)
                    segment_features = final_feat[sort_idx]
                    segment_global_indices = final_gidx[sort_idx]
                # end semantic_recycle_visual_guided_pruning branch only; visual_guided_pruning sets segment_features above
            else:
                # No need to prune; just return merged candidates.
                sort_idx = torch.argsort(flat_gidx)
                segment_features = flat_feats[sort_idx]
                segment_global_indices = flat_gidx[sort_idx]
        else:
            if important_mask is not None:
                if frame_selection_method == "stv_frame_gumbel" and frame_classes is not None and frame_similarity is not None:
                    seg_scores = frame_similarity[offset : offset + seg_len]
                    seg_classes = frame_classes[offset : offset + seg_len]
                    rep_local_idx = int(torch.argmax(seg_scores).item())
                    seg_type = int(seg_classes[rep_local_idx].item())
                    per_frame_budget = class_per_frame_budget[seg_type]
                else:
                    is_imp_segment = bool(important_mask[offset].item())
                    per_frame_budget = imp_per_frame if is_imp_segment else other_per_frame
            else:
                per_frame_budget = base_per_frame_budget

            num_visual_guided_pruning_tokens = math.ceil(per_frame_budget * astra_config.alpha) if astra_config.alpha > 0 else 0
            num_sttm_tokens = max(0, per_frame_budget - num_visual_guided_pruning_tokens)
            astra_config.num_visual_guided_pruning_tokens = num_visual_guided_pruning_tokens
            astra_config.num_sttm_tokens = num_sttm_tokens

            segment_features, segment_global_indices = segment_compression(
                segment_features=segment_features,
                segment_global_indices=segment_global_indices,
                cls_attention=segment_cls_attention,
                astra_config=astra_config,
            )
        all_segment_features.append(segment_features)
        all_segment_indices.append(segment_global_indices)
        offset += seg_len
    final_tokens = torch.cat(all_segment_features, dim=0)  # (num_final_tokens, feat_dim)
    final_global_indices = torch.cat(all_segment_indices, dim=0)  # (num_final_tokens,)

    sorted_indices = final_global_indices.argsort()
    sorted_tokens = final_tokens[sorted_indices]  # Sort by global indices.
    # Store the final token length in the `astra_config`.
    astra_config.visual_token_length = sorted_tokens.shape[0]
    # print(f"#Visual Tokens After Vision-Side Compression : {astra_config.visual_token_length}")
    return sorted_tokens, final_global_indices[sorted_indices]


def segment_compression(
    segment_features: torch.Tensor,
    segment_global_indices: torch.Tensor,
    cls_attention: torch.Tensor,
    astra_config: AstraConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compress the segment features by applying Temporal Average Merging (TAM) and Spatial Merging.

    Args:
        segment_features (torch.Tensor): The features of the video segment, of shape (num_frames, num_visual_tokens, feat_dim).
        segment_global_indices (torch.Tensor): The global indices of the video segment, of shape (num_frames, num_visual_tokens).
        cls_attention (torch.Tensor): [CLS] attentions used for per-frame token selection, of shape (num_frames, num_visual_tokens).
        astra_config (AstraConfig): The configuration for Astra.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The final tokens and their global indices after compression.
    """
    num_frames, num_visual_tokens, feat_dim = segment_features.shape

    # 1. Apply Attention and Diversity-based Token Selection (VGP) or DPP-based selection.
    if astra_config.alpha > 0:
        method = astra_config.token_selection_method
        if method == "divprune":
            # Alias for DivPrune-style greedy diversity selection (no cls attention needed).
            method = "div"
        # Methods that require cls_attention as input.
        needs_cls = method in ("attn", "visual_guided_pruning", "visual_guided_pruning", "dpp")
        additional_kwargs = {"cls_attention": cls_attention} if needs_cls else {}
        selected_features, selected_indices = ALL_TOKEN_SELECTION_METHOD[method](
            features=segment_features,
            num_retained_tokens=astra_config.num_visual_guided_pruning_tokens,
            **additional_kwargs,
        )
        selected_global_indices = segment_global_indices.gather(1, index=selected_indices).view(-1)
    else:
        # No token selection
        selected_features = torch.tensor([]).to(segment_features)
        selected_indices = torch.tensor([]).to(segment_global_indices)
        selected_global_indices = torch.tensor([]).to(segment_global_indices)

    mask = torch.ones(num_frames, num_visual_tokens, dtype=torch.bool, device=segment_features.device)
    mask.scatter_(1, selected_indices, False)

    num_other_tokens = astra_config.num_sttm_tokens * num_frames
    # 1. Apply Temporal Average Merging (TAM) to the segment features.
    if num_other_tokens > 0 and astra_config.temporal_threshold < 1.0:
        if num_frames > 1:
            temp_merged_token_list, temp_merged_indices_list = spatiotemporal_compression(
                video_features=segment_features,
                temporal_threshold=astra_config.temporal_threshold,
                token_mask=mask,
                astra_config=astra_config,
            )
            temp_merged_global_indices_list = [segment_global_indices.view(num_frames, -1)[i][temp_merged_indices] for i, temp_merged_indices in enumerate(temp_merged_indices_list)]
        else:
            # Single-frame segment, no temporal merging needed.
            temp_merged_token_list = [segment_features[0]]
            temp_merged_global_indices_list = [segment_global_indices[0]]
    else:
        # No spatial-temporal merging needed.
        temp_merged_token_list = []
        temp_merged_global_indices_list = []

    all_tokens = [selected_features.view(-1, feat_dim)]
    all_global_indices = [selected_global_indices]
    # 2. Apply Spatial Merging to the tokens after temporal merging.
    if num_other_tokens > 0: ## Only apply spatial merging when there are STTM tokens.
        # Calculate adaptive contextual ratio.
        num_current_retained_tokens = sum(len(tokens) for tokens in temp_merged_token_list)
        adapative_contextual_ratio = num_other_tokens / num_current_retained_tokens
        for temp_merged_tokens, temp_merged_global_indices in zip(temp_merged_token_list, temp_merged_global_indices_list):
            num_tokens, _ = temp_merged_tokens.shape
            aggregated_tokens = temp_merged_tokens
            global_token_indices = temp_merged_global_indices
            num_clusters = math.ceil(num_tokens * adapative_contextual_ratio)
            if num_clusters > 0 and adapative_contextual_ratio < 1.0:
                # Density Peak Clustering with kNN (VisualCluster-kNN).
                cluster_indices, cluster_center_indices = visual_cluster_knn(
                    features=temp_merged_tokens.unsqueeze(0),
                    num_clusters=num_clusters,
                    k=min(num_clusters, 7),
                )
                assigned_one_hot = F.one_hot(cluster_indices[0], num_classes=num_clusters).to(segment_features.dtype)
                aggregated_tokens = torch.einsum("n c, n d -> c d", assigned_one_hot, temp_merged_tokens)
                aggregated_tokens = aggregated_tokens / assigned_one_hot.sum(dim=0).unsqueeze(-1)
                global_token_indices = temp_merged_global_indices[cluster_center_indices[0]]
            all_tokens.append(aggregated_tokens)
            all_global_indices.append(global_token_indices)
    segment_final_tokens = torch.cat(all_tokens, dim=0)  # (num_final_tokens, feat_dim)
    segment_final_global_indices = torch.cat(all_global_indices, dim=0)  # (num_final_tokens,)
    return segment_final_tokens, segment_final_global_indices


def segment(
    video_features: torch.Tensor,
    segment_threshold: float,
    min_segment_num: int,
    complementary_segment: bool = True,
) -> torch.Tensor:
    """Segments the video features into distinct segments based on similarity.

    Args:
        video_features (torch.Tensor): The video features to segment.
        segment_threshold (float): The threshold for segmenting.
        min_segment_num (int): The minimum number of segments required.
        complementary_segment (int): Use complementary segmentation to ensure `min_segment_num` constraint.

    Returns:
        torch.Tensor: The lengths of the segments.
    """
    num_frames, feat_dim = video_features.shape

    # 0. Calculate transition similarities
    normed_video_features = video_features / video_features.norm(p=2, dim=-1, keepdim=True)
    transition_similarities = torch.sum(normed_video_features[:-1] * normed_video_features[1:], dim=-1)

    # 1. Find cut indices based on the segment threshold
    cut_indices = torch.where(transition_similarities < segment_threshold)[0]

    # 2. Ensure at least `min_segment_num` segments (Top-K or Uniform complementary segment)
    segment_lengths = additional_segment(
        cut_indices=cut_indices,
        num_frames=num_frames,
        min_segment_num=min_segment_num,
        transition_similarities=transition_similarities,
        segment_threshold=segment_threshold,
        complementary_segment=complementary_segment,
    )
    return segment_lengths


def additional_segment(
    cut_indices: torch.Tensor,
    num_frames: int,
    min_segment_num: int,
    transition_similarities: torch.Tensor,
    segment_threshold: float,
    complementary_segment: bool = True,
):
    num_segments = cut_indices.numel() + 1
    if num_segments < min_segment_num and complementary_segment:
        num_remaining_cut_indices = min_segment_num - num_segments
        transition_similarities[transition_similarities < segment_threshold] = 1.0
        complementary_cut_indices = torch.topk(transition_similarities, k=min(num_remaining_cut_indices, transition_similarities.shape[0]), largest=False).indices
        cut_indices = torch.cat([cut_indices, complementary_cut_indices]).sort().values

    padded_cut_indices = F.pad(cut_indices, (1, 1), value=0)
    padded_cut_indices[0] = -1
    padded_cut_indices[-1] = num_frames - 1
    segment_lengths = torch.diff(padded_cut_indices, n=1, dim=0)
    # print(f"segment lengths: {segment_lengths}")
    return segment_lengths


@torch.no_grad()
def visual_cluster_knn(features: torch.Tensor, num_clusters: int, k: int = 7, valid_token_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply VisualCluster-kNN clustering algorithm to the pooled image features, generating preliminary clustering result.

    Args:
        features (torch.Tensor): Pooled image features (temporal features), of shape (batch_size, seq_len, feat_dim).
        num_clusters (int): The number of clusters.
        k (int): The number of nearest neighbors to consider for local density. Default is 7.
        valid_token_mask (Optional[torch.Tensor]): Boolean Mask indicating valid tokens, of shape (batch_size, seq_len). Default is None.

    Returns:
        torch.Tensor: Cluster indices of shape (batch_size, seq_len).
    """
    invalid_token_mask = ~valid_token_mask if valid_token_mask is not None else None
    bsz, seq_len, feat_dim = features.shape

    # Calculate euclidean distance and local density
    dists = torch.cdist(features.float(), features.float()) / math.sqrt(feat_dim)

    # Mask out invalid tokens
    if valid_token_mask is not None:
        dists = torch.masked_fill(dists, invalid_token_mask.unsqueeze(1).expand(-1, seq_len, -1), dists.max() + 1)
    nearest_dist = torch.topk(dists, k=k, dim=-1, largest=False).values
    density = torch.mean(-(nearest_dist**2), dim=-1).exp()

    # Add little random noise to ensure no tokens have the same density.
    density = density + torch.rand_like(density, device=density.device, dtype=density.dtype) * 1e-6

    # Ensure the density of the empty token be 0
    if valid_token_mask is not None:
        density = torch.masked_fill(density, invalid_token_mask, 0.0)

    # Obtain the minimum distance to the point with higher density.
    mask = density[:, None, :] > density[:, :, None]
    max_dist = dists.view(bsz, -1).max(dim=-1)[0].view(-1, 1, 1)
    modified_dists = torch.where(mask, dists, max_dist)
    dist, _ = torch.min(modified_dists, dim=-1)

    # Calculate clustering score (clustering centers have the highest score)
    score = dist * density
    cluster_center_indices = torch.topk(score, k=num_clusters, dim=-1).indices

    # Obtain the distance matrix w.r.t cluster centers (batch_size, seq_len, num_clusters)
    dists = torch.gather(dists, dim=-1, index=cluster_center_indices.unsqueeze(1).expand(-1, seq_len, -1))
    cluster_indices = torch.argmin(dists, dim=-1)
    # Ensure each cluster center to merge with itself
    cluster_indices.scatter_(
        dim=-1,
        index=cluster_center_indices,
        src=torch.arange(num_clusters).to(cluster_indices).unsqueeze(0).expand(bsz, -1),
    )
    return cluster_indices, cluster_center_indices


@torch.no_grad()
def _visual_cluster_aggregate_from_remainder(
    rem_feats: torch.Tensor,
    rem_gidx: torch.Tensor,
    k_out: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Astra-style VisualCluster-kNN on remainder: cluster means, global indices of centers, and
    local row indices of centers into rem_feats (for excluding from a later semantic pool).
    """
    device = rem_feats.device
    dtype = rem_feats.dtype
    n = int(rem_feats.shape[0])
    zc = torch.zeros((0,), device=device, dtype=torch.long)
    if k_out <= 0 or n == 0:
        zf = torch.zeros((0, rem_feats.shape[-1]), device=device, dtype=dtype)
        zi = torch.zeros((0,), device=device, dtype=torch.long)
        return zf, zi, zc
    if n <= k_out:
        centers = torch.arange(n, device=device, dtype=torch.long)
        return rem_feats, rem_gidx, centers
    num_clusters = max(1, min(int(k_out), n))
    kk = min(num_clusters, 7)
    cluster_indices, cluster_center_indices = visual_cluster_knn(
        features=rem_feats.unsqueeze(0).float(),
        num_clusters=num_clusters,
        k=kk,
    )
    assigned_one_hot = F.one_hot(cluster_indices[0], num_classes=num_clusters).to(rem_feats.dtype)
    aggregated = torch.einsum("n c, n d -> c d", assigned_one_hot, rem_feats)
    aggregated = aggregated / assigned_one_hot.sum(dim=0).unsqueeze(-1).clamp_min(1e-12)
    g_out = rem_gidx[cluster_center_indices[0]]
    centers = cluster_center_indices[0].long()
    return aggregated.to(dtype=dtype), g_out, centers


@torch.no_grad()
def _post_merge_compress_remainder_visual_cluster(
    rem_feats: torch.Tensor,
    rem_gidx: torch.Tensor,
    k_out: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """VisualCluster-kNN on remainder; see _visual_cluster_aggregate_from_remainder."""
    af, ag, _ = _visual_cluster_aggregate_from_remainder(rem_feats, rem_gidx, k_out)
    return af, ag


def spatiotemporal_compression(
    video_features: torch.Tensor,
    temporal_threshold: float,
    token_mask: torch.Tensor,
    astra_config: AstraConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_frames, num_visual_tokens, feat_dim = video_features.shape
    # since we pass the whole segment features, the lower bound should contain VGP tokens.
    lower_bound = (astra_config.num_visual_guided_pruning_tokens + astra_config.num_sttm_tokens) * num_frames
    normed_video_features = video_features / video_features.norm(p=2, dim=-1, keepdim=True)
    cosine_similarities = torch.einsum("b n d, b m d -> b n m", normed_video_features[1:], normed_video_features[:-1])
    # Mask out the selected tokens.
    cosine_similarities[~token_mask[1:].unsqueeze(-1).expand(-1, -1, num_visual_tokens)] = -1.0
    cosine_similarities[~token_mask[:-1].unsqueeze(1).expand(-1, num_visual_tokens, -1)] = -1.0

    max_sims, max_sim_indices = torch.max(cosine_similarities, dim=-1)

    padded_max_sims = F.pad(max_sims, (0, 0, 1, 0), value=-1)
    padded_max_sim_indices = F.pad(max_sim_indices, (0, 0, 1, 0), value=-1)

    token_counts = torch.ones(num_frames, num_visual_tokens).to(video_features)
    mask = padded_max_sims > temporal_threshold
    retaining_token_mask = ~mask

    # Ensure the number of retained tokens after TAM does not exceed the lower bound.
    if retaining_token_mask.int().sum() < lower_bound:
        soft_threshold = padded_max_sims.view(-1).topk(k=(num_frames * num_visual_tokens) - lower_bound).values[-1]
        soft_threshold = max(soft_threshold, -1.0 + 1e-6)
        mask = padded_max_sims > soft_threshold
        retaining_token_mask = ~mask

    for frame_idx in range(num_frames - 1, -1, -1):
        frame_features = video_features[frame_idx]
        frame_token_counts = token_counts[frame_idx]
        frame_max_sim_indices = padded_max_sim_indices[frame_idx]

        # Apply spatiotemporal average merging.
        tokens_to_merge = frame_features[~mask[frame_idx]]
        to_merge_token_counts = frame_token_counts[~mask[frame_idx]]
        if tokens_to_merge.numel() > 0:
            aggregated_tokens = tokens_to_merge / to_merge_token_counts.unsqueeze(-1).to(tokens_to_merge.dtype)
            video_features[frame_idx][~mask[frame_idx]] = aggregated_tokens
            token_counts[frame_idx][~mask[frame_idx]] = 1

        # other tokens are connected to the previous frame's tokens
        other_tokens = frame_features[mask[frame_idx]]
        if other_tokens.numel() > 0:
            # Distribute other tokens to the previous frame's tokens (anchor tokens)
            anchor_token_indices = frame_max_sim_indices[mask[frame_idx]]
            assigned_one_hot = F.one_hot(anchor_token_indices, num_classes=num_visual_tokens).to(video_features.dtype)
            aggregated_tokens = torch.einsum("m n, m d -> n d", assigned_one_hot, other_tokens)  # (num_visual_tokens, feat_dim)
            aggregated_token_counts = assigned_one_hot.sum(dim=0)  # (num_visual_tokens,)
            video_features[frame_idx - 1] += aggregated_tokens
            token_counts[frame_idx - 1] += aggregated_token_counts
            token_counts[frame_idx][mask[frame_idx]] = 0

    # Filter final tokens
    final_tokens = []
    retained_token_indices = []
    for i in range(num_frames):
        frame_mask = retaining_token_mask[i] & token_mask[i]
        frame_retained_tokens = video_features[i][frame_mask]  # (frame_retained_tokens_num, feat_dim)
        frame_retained_indices = torch.where(frame_mask)[0]  # (frame_retained_tokens_num,)
        final_tokens.append(frame_retained_tokens)
        retained_token_indices.append(frame_retained_indices)

    return final_tokens, retained_token_indices


def fastv_prune(
    hidden_states: torch.Tensor,
    causal_mask: Optional[torch.Tensor],
    attentions: Optional[torch.Tensor],
    cache_position: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    astra_config: AstraConfig,
    visual_pos_masks: Optional[torch.BoolTensor] = None,
):
    bsz, seq_length, _ = hidden_states.shape
    device = hidden_states.device
    # Obtain Astra arguments.
    visual_token_start_index = astra_config.visual_token_start_index
    visual_token_length = astra_config.visual_token_length
    if visual_token_start_index is None or visual_token_length is None:
        raise ValueError(
            "Astra fastv_prune: visual_token_start_index/visual_token_length is None. "
            "This indicates the LLaVA multimodal input assembly did not record visual token boundaries. "
            "Please ensure Astra's patched llava_arch is active and that the prompt contains <image>/<video> tokens."
        )
    visual_token_end_index = visual_token_start_index + visual_token_length

    retention_ratio = float(astra_config.llm_retention_ratio)
    num_retained_tokens = math.ceil(visual_token_length * retention_ratio)

    # Compatible to LLaVA-OneVision.
    if visual_pos_masks is None:
        visual_pos_masks = torch.zeros((bsz, seq_length), dtype=torch.bool, device=device)
        visual_pos_masks[:, visual_token_start_index:visual_token_end_index] = True
    non_visual_pos_masks = ~visual_pos_masks

    visual_features = hidden_states[visual_pos_masks, :]
    visual_global_indices = torch.where(visual_pos_masks[0])[0]
    non_visual_global_indices = torch.where(non_visual_pos_masks[0])[0]
    attn = torch.mean(attentions[:, :, -1, :], dim=1)[visual_pos_masks]
    num_available_visual_tokens = int(visual_features.shape[0])

    # Keep k within valid top-k range for the *actual* masked visual length.
    if num_available_visual_tokens <= 0:
        keep_indices = non_visual_global_indices
        hidden_states = hidden_states[:, keep_indices].contiguous()
        cache_position = keep_indices if cache_position is None else cache_position[keep_indices].contiguous()
        position_ids = keep_indices.unsqueeze(0) if position_ids is None else position_ids[..., keep_indices].contiguous()
        position_embeddings = (
            position_embeddings[0][..., keep_indices, :].contiguous(),
            position_embeddings[1][..., keep_indices, :].contiguous(),
        )
        if causal_mask is not None:
            causal_mask = causal_mask[:, :, keep_indices, :][:, :, :, keep_indices].clone()
        astra_config.visual_token_length = 0
        return hidden_states, causal_mask, position_ids, cache_position, position_embeddings, keep_indices

    num_retained_tokens = max(1, min(int(num_retained_tokens), num_available_visual_tokens))

    _, topk_indices = attn_based_token_selection(
        features=visual_features.unsqueeze(0),
        cls_attention=attn.unsqueeze(0),
        num_retained_tokens=num_retained_tokens,
    )
    topk_indices = topk_indices.squeeze(0)
    all_global_indices = [non_visual_global_indices, visual_global_indices[topk_indices]]
    keep_indices = torch.sort(torch.cat(all_global_indices)).values

    # Filter
    hidden_states = hidden_states[:, keep_indices].contiguous()
    cache_position = keep_indices if cache_position is None else cache_position[keep_indices].contiguous()
    position_ids = keep_indices.unsqueeze(0) if position_ids is None else position_ids[..., keep_indices].contiguous()
    position_embeddings = (
        position_embeddings[0][..., keep_indices, :].contiguous(),
        position_embeddings[1][..., keep_indices, :].contiguous(),
    )

    new_seq_length = hidden_states.shape[1]
    if causal_mask is not None:
        # Use keep_indices for correct mask selection (not just first new_seq_length entries).
        # .clone() ensures this is an independent tensor, not a view (prevents dangling pointer in multi-GPU).
        causal_mask = causal_mask[:, :, keep_indices, :][:, :, :, keep_indices].clone()
    # Update astra config.
    astra_config.visual_token_length = int(num_retained_tokens)
    return hidden_states, causal_mask, position_ids, cache_position, position_embeddings, keep_indices
