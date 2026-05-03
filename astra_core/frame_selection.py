from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class FrameSelectionResult:
    important_mask: torch.BoolTensor  # (num_frames,)
    selected_indices: torch.LongTensor  # (k,)
    quality: torch.Tensor  # (num_frames,)


@torch.no_grad()
def stv_frame_gumbel_scores(
    frame_features: torch.Tensor,
    text_cls_embed: torch.Tensor,
    tau: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Q-Frame style scoring with Gumbel perturbation.

    Returns:
        sim: cosine similarity per frame, shape (T,)
        gumbel_score: log-softmax(sim / tau) + gumbel_noise, shape (T,)
    """
    if text_cls_embed.dim() == 2:
        text = text_cls_embed[0]
    else:
        text = text_cls_embed

    feats = frame_features.float()
    feats = feats / (feats.norm(p=2, dim=-1, keepdim=True) + 1e-12)
    text = text.float()
    text = text / (text.norm(p=2, dim=-1, keepdim=False) + 1e-12)
    sim = torch.matmul(feats, text)  # (T,)

    tau = max(float(tau), 1e-6)
    pi = torch.softmax(sim / tau, dim=0)
    g = -torch.log(-torch.log(torch.rand_like(pi).clamp(1e-6, 1 - 1e-6)))
    gumbel_score = torch.log(pi.clamp_min(1e-12)) + g
    return sim, gumbel_score


@torch.no_grad()
def stv_frame_rank_to_three_classes(
    rank_scores: torch.Tensor,
    important_ratio: float = 0.25,
    context_upper_ratio: float = 0.75,
) -> torch.LongTensor:
    """
    Split frames into 3 classes by rank:
    - 2: important (top important_ratio)
    - 1: context   (important_ratio ~ context_upper_ratio)
    - 0: irrelevant (rest)
    """
    T = int(rank_scores.numel())
    imp_n = int(round(T * float(important_ratio)))
    ctx_up = int(round(T * float(context_upper_ratio)))
    imp_n = max(1, min(imp_n, T))
    ctx_up = max(imp_n, min(ctx_up, T))

    order = torch.argsort(rank_scores, descending=True)
    classes = torch.zeros(T, dtype=torch.long, device=rank_scores.device)
    classes[order[:imp_n]] = 2
    classes[order[imp_n:ctx_up]] = 1
    return classes


@torch.no_grad()
def text_relevance_minmax(
    frame_features: torch.Tensor,
    text_cls_embed: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-frame relevance conditioned on a text CLS embedding.

    We follow the spirit of CDPruner:
    - Normalize features
    - Use cosine similarity as relevance
    - Min-max normalize to (0, 1]
    """
    # frame_features: (T, D)
    # text_cls_embed: (D,) or (1, D)
    if text_cls_embed.dim() == 2:
        text = text_cls_embed[0]
    else:
        text = text_cls_embed

    feats = frame_features.float()
    feats = feats / (feats.norm(p=2, dim=-1, keepdim=True) + 1e-12)
    text = text.float()
    text = text / (text.norm(p=2, dim=-1, keepdim=False) + 1e-12)

    sim = torch.matmul(feats, text)  # (T,)
    # min-max normalize
    sim_min = sim.min()
    sim_max = sim.max()
    denom = (sim_max - sim_min).clamp_min(1e-12)
    rel = (sim - sim_min + 1e-6) / denom  # (0, 1+)
    return rel.clamp_min(1e-6)


@torch.no_grad()
def text_relevance_logits(
    frame_features: torch.Tensor,
    text_cls_embed: torch.Tensor,
) -> torch.Tensor:
    """
    Query-frame cosine similarity logits (no min-max normalization).
    Used by Q-Frame style Gumbel-Max frame sampling.
    """
    if text_cls_embed.dim() == 2:
        text = text_cls_embed[0]
    else:
        text = text_cls_embed

    feats = frame_features.float()
    feats = feats / (feats.norm(p=2, dim=-1, keepdim=True) + 1e-12)
    text = text.float()
    text = text / (text.norm(p=2, dim=-1, keepdim=False) + 1e-12)
    sim = torch.matmul(feats, text)  # (T,)
    return sim


@torch.no_grad()
def adjacent_change_mean(frame_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute adjacent-frame change magnitudes and their mean.

    Returns:
        d: adjacent change vector, shape (T-1,)
        mu_d: mean adjacent change, scalar tensor
    """
    T, _ = frame_features.shape
    if T <= 1:
        zero = torch.zeros(0, device=frame_features.device, dtype=frame_features.dtype)
        return zero, torch.tensor(0.0, device=frame_features.device, dtype=frame_features.dtype)

    feats = frame_features.float()
    feats = feats / (feats.norm(p=2, dim=-1, keepdim=True) + 1e-12)
    sim_adj = (feats[1:] * feats[:-1]).sum(dim=-1).clamp(-1.0, 1.0)
    d = 1.0 - sim_adj
    mu_d = d.mean() if d.numel() > 0 else torch.tensor(0.0, device=feats.device, dtype=feats.dtype)
    return d, mu_d


@torch.no_grad()
def stv_budget_weights(
    frame_features: torch.Tensor,
    text_cls_embed: Optional[torch.Tensor],
    eps: float = 1e-6,
    temperature: float = 1.0,
    mu_threshold: float = 0.10,
    mu_full: float = 0.60,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    语义相关逐帧权重 + 平均 STV 门控的均匀混合（用于动态 budget 日志 / 辅助统计）。

    - w_sem = softmax(text_relevance_logits / temperature)
    - sem_mix = clamp((average_stv - mu_threshold) / (mu_full - mu_threshold), 0, 1)
    """
    T, _ = frame_features.shape
    device = frame_features.device

    _, mu_d = adjacent_change_mean(frame_features)
    w_uni = torch.full((T,), 1.0 / max(1, T), device=device, dtype=torch.float32)
    if text_cls_embed is None:
        sem_mix = torch.tensor(0.0, device=device, dtype=frame_features.dtype)
        return (
            w_uni.to(frame_features.dtype),
            torch.ones((T,), device=device, dtype=frame_features.dtype),
            mu_d.to(frame_features.dtype),
            sem_mix,
        )

    logits = text_relevance_logits(frame_features, text_cls_embed).float()
    logits = logits / max(float(temperature), 1e-6)
    # numeric stability: standard softmax
    probs = torch.softmax(logits, dim=0).clamp_min(float(eps))
    probs = probs / probs.sum().clamp_min(1e-12)

    mu_t = float(mu_threshold)
    mu_f = float(mu_full)
    denom = max(1e-12, mu_f - mu_t)
    sem_mix_f = (float(mu_d.item()) - mu_t) / denom
    sem_mix_f = max(0.0, min(1.0, sem_mix_f))
    sem_mix = torch.tensor(sem_mix_f, device=device, dtype=frame_features.dtype)

    w = (1.0 - sem_mix_f) * w_uni + sem_mix_f * probs
    w = w / w.sum().clamp_min(1e-12)
    return w.to(frame_features.dtype), probs.to(frame_features.dtype), mu_d.to(frame_features.dtype), sem_mix


# 旧名兼容（已弃用）
stv_frame_soft_budget_weights = stv_budget_weights


@torch.no_grad()
def stv_frame_gumbel_select_frames(
    frame_features: torch.Tensor,
    text_cls_embed: torch.Tensor,
    top_p: float = 0.1,
    tau: float = 0.8,
) -> FrameSelectionResult:
    """
    Q-Frame style query-aware frame selection:
      1) I = cosine(query, frame)
      2) pi = softmax(I / tau)
      3) p = log(pi) + gumbel_noise
      4) select top-k by p

    We map `top_p` to selected count k:
    - 0 < top_p <= 1: k = ceil(T * top_p)
    - top_p > 1: k = int(top_p)
    """
    device = frame_features.device
    T, _ = frame_features.shape

    logits = text_relevance_logits(frame_features, text_cls_embed)  # (T,)
    tau = float(max(tau, 1e-6))
    probs = torch.softmax(logits / tau, dim=0)

    if top_p <= 0:
        k = 1
    elif top_p <= 1:
        k = int(torch.ceil(torch.tensor(T * float(top_p))).item())
    else:
        k = int(top_p)
    k = max(1, min(k, T))

    eps = torch.rand_like(probs).clamp_min(1e-12)
    g = -torch.log(-torch.log(eps))
    perturbed = torch.log(probs.clamp_min(1e-12)) + g
    selected = torch.topk(perturbed, k=k, largest=True).indices.sort().values

    important_mask = torch.zeros(T, dtype=torch.bool, device=device)
    important_mask[selected] = True
    return FrameSelectionResult(important_mask=important_mask, selected_indices=selected, quality=probs)


@torch.no_grad()
def dpp_select_frames_top_p(
    frame_features: torch.Tensor,
    frame_quality: torch.Tensor,
    top_p: float = 0.9,
    max_selected: Optional[int] = None,
) -> FrameSelectionResult:
    """
    DPP-style greedy MAP selection over frames with an adaptive budget decided by top-p.

    Args:
        frame_features: (T, D) pooled features per frame.
        frame_quality:  (T,) non-negative quality per frame (e.g., pooled CLS attention).
        top_p: choose the smallest k such that sum(q[selected[:k]]) / sum(q) >= top_p.
        max_selected: optional hard cap on k.
    """
    device = frame_features.device
    T, D = frame_features.shape

    q = frame_quality.float().clamp_min(0.0) + 1e-8
    feats = frame_features.float()
    feats = feats / (feats.norm(p=2, dim=-1, keepdim=True) + 1e-12)

    # Similarity kernel.
    sim = feats @ feats.t()  # (T, T)
    L = q.unsqueeze(1) * sim * q.unsqueeze(0)  # (T, T)

    # Greedy MAP (fast, single-sequence DPP).
    # We greedily select up to max_k, then truncate by top-p.
    max_k = int(T if max_selected is None else min(max_selected, T))
    di2s = torch.diagonal(L).clone()  # (T,)
    cis = torch.zeros((max_k, T), device=device)
    selected = torch.empty(max_k, dtype=torch.long, device=device)
    num = 0
    for t in range(max_k):
        j = torch.argmax(di2s).item()
        if di2s[j] <= 0:
            break
        selected[t] = j
        eis = (L[j] - torch.einsum("t,tj->j", cis[:t, j], cis[:t])) / (di2s[j].sqrt() + 1e-8)
        cis[t] = eis
        di2s -= eis**2
        di2s[j] = -float("inf")
        num += 1

    if num == 0:
        # Fallback: pick the best-quality frame.
        selected = torch.topk(q, k=1).indices
    else:
        selected = selected[:num]

    # Decide k by top-p on quality mass (in greedy order).
    if top_p is None or top_p <= 0 or top_p >= 1:
        k = selected.numel()
    else:
        total_q = q.sum().clamp_min(1e-8)
        cum = torch.cumsum(q[selected], dim=0) / total_q
        k = int((cum < top_p).sum().item() + 1)
        k = max(1, min(k, selected.numel()))

    selected = selected[:k]
    selected = torch.unique(selected).sort().values
    important_mask = torch.zeros(T, dtype=torch.bool, device=device)
    important_mask[selected] = True

    return FrameSelectionResult(important_mask=important_mask, selected_indices=selected, quality=q)


@torch.no_grad()
def build_frame_features_and_quality(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build frame-level pooled features and quality scores from token-level inputs.

    - frame_features: mean pooled token features per frame.
    - frame_quality: mean CLS attention per frame.
    """
    frame_features = video_features.mean(dim=1)  # (T, D)
    frame_quality = cls_attention.mean(dim=1)  # (T,)
    return frame_features, frame_quality

