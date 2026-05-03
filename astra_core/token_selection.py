from typing import Tuple

from enum import Enum
import torch


class TokenSelectionMethod(str, Enum):
    ATTN = "attn"
    DIV = "div"
    DIVPRUNE = "divprune"
    VGP = "visual_guided_pruning"
    DPP = "dpp"


def pairwise_cosine_distances(image_features: torch.Tensor) -> torch.Tensor:
    """Calculate pairwise cosine distances for a batch of feature vectors.

    Args:
        image_features (torch.Tensor): Visual features, of shape (bsz, num_visual_tokens, feat_dim)

    Returns:
        torch.Tensor: Pairwise cosine distances, of shape (bsz, num_visual_tokens, num_visual_tokens)
    """
    normed_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    similarities = torch.bmm(normed_features, normed_features.transpose(-1, -2))
    return 1.0 - similarities


def visual_guided_pruning_based_token_selection(
    features: torch.Tensor,
    cls_attention: torch.Tensor,
    num_retained_tokens: int,
) -> torch.Tensor:
    """Select visual tokens based on attention and diversity.

    Args:
        features (torch.Tensor): Visual features, of shape (bsz, num_visual_tokens, feat_dim)
        cls_attention (torch.Tensor): [CLS] attention, of shape (bsz, num_visual_tokens)
        num_retained_tokens (int): Number of tokens to retain

    Returns:
        torch.Tensor: Pruned features, of shape (bsz, num_retained_tokens, feat_dim)
    """
    original_features = features
    features = features.float()
    pooled_features = features.mean(1) # (num_frames, feat_dim)
    global_cls_attention = cls_attention.float() * 1e6  # Scale attention to avoid numerical issues
    bsz, num_visual_tokens, feat_dim = features.shape
    dist_matrix = pairwise_cosine_distances(features)

    # (1) [CLS] attention calibration term (bsz, 1, num_visual_tokens).
    calibration_term1 = global_cls_attention.unsqueeze(1)
    # (2) Event relevance calibration term (bsz, 1, num_visual_tokens).
    local_cls_attention = torch.einsum("b n d, c d -> b c n", features, pooled_features).mean(1)
    calibration_term2 = local_cls_attention.unsqueeze(1)
    # Calibrate distance matrix by [cls] attention and event relevance (bsz, num_visual_tokens, num_visual_tokens)
    dist_matrix = dist_matrix * calibration_term1 * calibration_term2

    # Initialize keeping indices (bsz, num_retained_tokens).
    keep_indices = torch.zeros(bsz, num_retained_tokens, dtype=torch.long, device=features.device)

    # select the first token.
    min_dist = torch.topk(dist_matrix, k=2, dim=1, largest=False).values[:, 1, :]  # (bsz, num_visual_tokens)
    keep_indices[:, 0] = torch.argmax(min_dist, dim=-1)  # (bsz,)

    # Select the rest of the tokens.
    for i in range(1, num_retained_tokens):
        # Get the distances to the already selected tokens.
        dist_sub_matrix = torch.gather(
            dist_matrix,
            dim=1,
            index=keep_indices[:, :i].unsqueeze(-1).expand(-1, -1, num_visual_tokens),
        )
        min_dist = torch.min(dist_sub_matrix, dim=1).values
        keep_indices[:, i] = torch.argmax(min_dist, dim=-1)

    keep_indices = keep_indices.sort().values
    selected_features = torch.gather(original_features, dim=1, index=keep_indices.unsqueeze(-1).expand(-1, -1, feat_dim))  # (bsz, num_retained_tokens, feat_dim)

    return selected_features, keep_indices


def attn_based_token_selection(
    features: torch.Tensor,
    cls_attention: torch.Tensor,
    num_retained_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select visual tokens based on attention.

    Args:
        features (torch.Tensor): Visual features, of shape (bsz, num_visual_tokens, feat_dim)
        cls_attention (torch.Tensor): [CLS] attention, of shape (bsz, num_visual_tokens)
        num_retained_tokens (int): Number of tokens to retain

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Pruned features and their indices.
    """
    bsz, num_visual_tokens, feat_dim = features.shape
    if num_visual_tokens <= 0:
        empty_idx = torch.zeros((bsz, 0), dtype=torch.long, device=features.device)
        empty_feat = features[:, :0, :]
        return empty_feat, empty_idx
    k = max(1, min(int(num_retained_tokens), int(num_visual_tokens)))
    topk_indices = torch.topk(cls_attention, k=k, dim=-1).indices.sort().values
    selected_features = torch.gather(features, dim=1, index=topk_indices.unsqueeze(-1).expand(-1, -1, feat_dim))
    return selected_features, topk_indices


def div_based_token_selection(
    features: torch.Tensor,
    num_retained_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select visual tokens based on diversity.

    Args:
        features (torch.Tensor): Visual features, of shape (bsz, num_visual_tokens, feat_dim)
        num_retained_tokens (int): Number of tokens to retain

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Pruned features and their indices.
    """
    original_features = features
    features = features.float()
    bsz, num_visual_tokens, feat_dim = features.shape
    dist_matrix = pairwise_cosine_distances(features)
    min_dist = torch.topk(dist_matrix, k=2, dim=1, largest=False).values[:, 1, :]  # (bsz, num_visual_tokens)

    keep_indices = torch.zeros(bsz, num_retained_tokens, dtype=torch.long, device=features.device)  # (bsz, num_retained_tokens)
    keep_indices[:, 0] = torch.argmax(min_dist, dim=-1)  # (bsz,)

    for i in range(1, num_retained_tokens):
        dist_sub_matrix = torch.gather(
            dist_matrix,
            dim=1,
            index=keep_indices[:, :i].unsqueeze(-1).expand(-1, -1, num_visual_tokens),
        )
        min_dist = torch.min(dist_sub_matrix, dim=1).values
        # Prevent select the same token again.
        min_dist.scatter_(1, keep_indices[:, :i], -1)  # (bsz, num_visual_tokens)
        keep_indices[:, i] = torch.argmax(min_dist, dim=-1)

    keep_indices = keep_indices.sort().values
    selected_features = torch.gather(original_features, dim=1, index=keep_indices.unsqueeze(-1).expand(-1, -1, feat_dim))  # (bsz, num_retained_tokens, feat_dim)

    return selected_features, keep_indices


def divprune_based_token_selection(
    features: torch.Tensor,
    num_retained_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Diversity pruning (DivPrune-style) token selection.

    This matches the greedy farthest-point style selection used in DivPrune:
    iteratively add the token that maximizes the minimum cosine distance to
    the already selected set.
    """
    return div_based_token_selection(features=features, num_retained_tokens=num_retained_tokens)


def dpp_based_token_selection(
    features: torch.Tensor,
    cls_attention: torch.Tensor,
    num_retained_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select visual tokens using a simplified DPP-style objective.

    We flatten batch and token dimensions to build a global kernel, then run a fast
    MAP-style greedy selection, and finally map the selected indices back to per-frame
    token indices. This is training-free and reuses existing features/attentions.
    """
    bsz, num_visual_tokens, feat_dim = features.shape
    device = features.device

    # Flatten frames into a single sequence.
    flat_feats = features.reshape(-1, feat_dim).float()  # (B*N, D)
    flat_attn = cls_attention.reshape(-1).float() + 1e-6  # (B*N,)

    # Normalize features and build cosine similarity matrix.
    flat_feats = flat_feats / flat_feats.norm(p=2, dim=-1, keepdim=True)
    sim = torch.matmul(flat_feats, flat_feats.t())  # (B*N, B*N)

    # Kernel L = diag(q) * sim * diag(q), where q comes from attention.
    L = flat_attn.unsqueeze(1) * sim * flat_attn.unsqueeze(0)  # (B*N, B*N)

    # Fast MAP inference of (approximate) DPP: single-sequence version.
    T = min(num_retained_tokens * bsz, L.size(0))  # global budget upper bound
    di2s = torch.diagonal(L).clone()  # (B*N,)
    cis = torch.zeros((T, L.size(0)), device=device)  # (T, B*N)
    selected = torch.empty(T, dtype=torch.long, device=device)

    for t in range(T):
        j = torch.argmax(di2s)
        selected[t] = j
        if di2s[j] <= 0:
            break
        eis = (L[j] - torch.einsum("t,tj->j", cis[:t, j], cis[:t])) / (di2s[j].sqrt() + 1e-6)
        cis[t] = eis
        di2s -= eis**2
        di2s[j] = -float("inf")

    selected = torch.unique(torch.sort(selected).values)

    # Fallback: if very few tokens are selected (degenerate case), fall back to attention top-k.
    if selected.numel() < num_retained_tokens:
        top_attn = torch.topk(flat_attn, k=num_retained_tokens).indices
        selected = torch.unique(torch.cat([selected, top_attn]))

    # Map global indices back to per-frame indices.
    flat_idx = selected
    frame_idx = flat_idx // num_visual_tokens
    token_idx = flat_idx % num_visual_tokens

    # Build per-frame selections, ensuring each frame has exactly num_retained_tokens tokens.
    selected_features = torch.zeros(bsz, num_retained_tokens, feat_dim, device=device, dtype=features.dtype)
    selected_indices = torch.zeros(bsz, num_retained_tokens, dtype=torch.long, device=device)

    for b in range(bsz):
        mask = frame_idx == b
        idx_b = token_idx[mask]

        # If this frame has no token selected by global DPP, fall back to attention top-1.
        if idx_b.numel() == 0:
            attn_b = cls_attention[b].float()
            idx_b = torch.topk(attn_b, k=1).indices

        # If less than required, fill with attention-based tokens for this frame.
        if idx_b.numel() < num_retained_tokens:
            attn_b = cls_attention[b].float()
            extra = torch.topk(attn_b, k=num_retained_tokens).indices
            idx_b = torch.unique(torch.cat([idx_b, extra]))[:num_retained_tokens]
        else:
            idx_b = idx_b[:num_retained_tokens]

        selected_indices[b] = idx_b
        selected_features[b] = features[b, idx_b]

    return selected_features, selected_indices


ALL_TOKEN_SELECTION_METHOD = {
    "attn": attn_based_token_selection,
    "visual_guided_pruning": visual_guided_pruning_based_token_selection,
    "div": div_based_token_selection,
    "divprune": divprune_based_token_selection,
    "dpp": dpp_based_token_selection,
}
