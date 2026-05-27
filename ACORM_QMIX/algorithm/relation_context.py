"""
Relation-context modules for Relation-aware ACORM (RACORM).

v3 changes:
    1) Role-pair mining is rank-based by default: top-k relation-similar agents
       are positives and bottom-k relation-dissimilar agents are negatives.
       Threshold-based mining is kept only for ablation.
    2) Relation graph construction supports pre-MLP sparsification. When
       --relation_sparse_topk > 0, only the top-k cheap candidate neighbors are
       passed through the expensive pairwise relation MLP, reducing relation
       computation from O(B*N^2) MLP calls to O(B*N*K) MLP calls.
    3) TensorBoard diagnostics are changed accordingly: rank-positive mean
       similarity, rank-negative mean similarity, rank margin, candidate edge
       fraction, effective edge count/fraction, and sparse-topk are reported.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RelationBatch:
    pair_repr: torch.Tensor             # Dense view: (B, N, N, relation_dim); zeros for pruned edges.
    edge_logits: torch.Tensor           # Dense view: (B, N, N); -1e9 for pruned/masked edges.
    edge_weights: torch.Tensor          # Dense view: (B, N, N), row-normalized over retained neighbors.
    ego_context: torch.Tensor           # (B, N, relation_dim)
    context_similarity: torch.Tensor    # (B, N, N)
    edge_mask: torch.Tensor             # (B, N, N), True for edges used by relation attention.
    candidate_edge_frac: torch.Tensor   # scalar: fraction of non-self edges entering the pairwise MLP.
    effective_edge_count: torch.Tensor  # scalar: average number of attention-retained edges per agent.
    effective_edge_frac: torch.Tensor   # scalar: effective_edge_count / (N - 1).


class RelationContextEncoder(nn.Module):
    """Learns pairwise relation features and per-agent relation contexts.

    Important distinction:
        relation_sparse_topk: pre-MLP candidate pruning. This actually reduces
            the expensive pairwise MLP cost.
        relation_topk: post-MLP attention pruning. This changes the attention
            graph sparsity but does not reduce pairwise MLP cost when used alone.
    """

    def __init__(self, args):
        super().__init__()
        self.n_agents = args.N
        self.agent_dim = args.agent_embedding_dim
        self.state_dim = args.state_dim
        self.relation_dim = getattr(args, "relation_dim", 64)
        self.hidden_dim = getattr(args, "relation_hidden_dim", 128)
        self.use_state = bool(getattr(args, "relation_use_state", True))
        self.mask_self = bool(getattr(args, "relation_mask_self", True))

        # Post-MLP attention pruning. Kept for backward compatibility.
        self.topk = int(getattr(args, "relation_topk", 0))

        # v3: pre-MLP sparse relation construction.
        self.sparse_topk = int(getattr(args, "relation_sparse_topk", 0))
        self.sparse_metric = str(getattr(args, "relation_sparse_metric", "cosine")).lower()
        self.sparsify_before_mlp = bool(getattr(args, "relation_sparsify_before_mlp", True))

        # e_i, e_j, |e_i-e_j|, e_i*e_j, optional projected global state.
        pair_in = self.agent_dim * 4
        if self.use_state:
            self.state_proj = nn.Sequential(
                nn.Linear(self.state_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.agent_dim),
            )
            pair_in += self.agent_dim
        else:
            self.state_proj = None

        self.edge_mlp = nn.Sequential(
            nn.Linear(pair_in, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.relation_dim),
        )
        self.edge_score = nn.Linear(self.relation_dim, 1)
        self.context_proj = nn.Sequential(
            nn.Linear(self.relation_dim, self.relation_dim),
            nn.ReLU(),
            nn.Linear(self.relation_dim, self.relation_dim),
        )

    def _cheap_candidate_scores(self, agent_embeddings: torch.Tensor) -> torch.Tensor:
        """Cheap dense scores used only for pre-MLP sparse candidate selection.

        Returns:
            scores: (B, N, N), larger means more likely to be retained.
        """
        if self.sparse_metric == "cosine":
            e = F.normalize(agent_embeddings, dim=-1)
            return torch.matmul(e, e.transpose(1, 2))
        if self.sparse_metric == "l2":
            # Larger score means closer neighbor.
            return -torch.cdist(agent_embeddings, agent_embeddings, p=2)
        raise ValueError(f"Unsupported relation_sparse_metric: {self.sparse_metric}")

    def _select_pre_mlp_candidates(self, agent_embeddings: torch.Tensor) -> torch.Tensor:
        """Return candidate neighbor indices of shape (B, N, K)."""
        bsz, n_agents, _ = agent_embeddings.shape
        if n_agents <= 1:
            return torch.zeros((bsz, n_agents, 0), dtype=torch.long, device=agent_embeddings.device)

        k = min(max(int(self.sparse_topk), 0), n_agents - 1)
        if k <= 0 or not self.sparsify_before_mlp:
            all_idx = torch.arange(n_agents, device=agent_embeddings.device).view(1, 1, n_agents).expand(bsz, n_agents, -1)
            if self.mask_self:
                nonself = ~torch.eye(n_agents, device=agent_embeddings.device, dtype=torch.bool).view(1, n_agents, n_agents)
                return all_idx[nonself.expand(bsz, -1, -1)].view(bsz, n_agents, n_agents - 1)
            return all_idx

        scores = self._cheap_candidate_scores(agent_embeddings)
        if self.mask_self:
            eye = torch.eye(n_agents, device=scores.device, dtype=torch.bool).unsqueeze(0)
            scores = scores.masked_fill(eye, -1e9)
        return torch.topk(scores, k=k, dim=-1, largest=True).indices

    def forward(self, agent_embeddings: torch.Tensor, states: Optional[torch.Tensor] = None) -> RelationBatch:
        """Compute sparse/dense pairwise relation contexts.

        Args:
            agent_embeddings: (B, N, D_e)
            states: optional (B, state_dim)
        """
        if agent_embeddings.dim() != 3:
            raise ValueError(f"agent_embeddings must be (B,N,D), got {tuple(agent_embeddings.shape)}")
        bsz, n_agents, _ = agent_embeddings.shape
        if n_agents != self.n_agents:
            self.n_agents = n_agents

        device = agent_embeddings.device
        relation_dim = self.relation_dim
        idx = self._select_pre_mlp_candidates(agent_embeddings)  # (B, N, K)
        k_cand = idx.shape[-1]

        pair_repr_dense = agent_embeddings.new_zeros((bsz, n_agents, n_agents, relation_dim))
        edge_logits_dense = agent_embeddings.new_full((bsz, n_agents, n_agents), -1e9)
        edge_mask_dense = torch.zeros((bsz, n_agents, n_agents), dtype=torch.bool, device=device)

        if k_cand > 0:
            e_i = agent_embeddings.unsqueeze(2).expand(-1, -1, k_cand, -1)
            all_e_j = agent_embeddings.unsqueeze(1).expand(-1, n_agents, -1, -1)
            gather_idx = idx.unsqueeze(-1).expand(-1, -1, -1, self.agent_dim)
            e_j = torch.gather(all_e_j, dim=2, index=gather_idx)

            pair_features = [e_i, e_j, torch.abs(e_i - e_j), e_i * e_j]
            if self.use_state:
                if states is None:
                    raise ValueError("relation_use_state=True but states=None")
                s = self.state_proj(states).view(bsz, 1, 1, self.agent_dim).expand(-1, n_agents, k_cand, -1)
                pair_features.append(s)

            pair_input = torch.cat(pair_features, dim=-1)
            pair_repr_sparse = self.edge_mlp(pair_input)                         # (B, N, K, D_r)
            edge_logits_sparse = self.edge_score(pair_repr_sparse).squeeze(-1)    # (B, N, K)

            # Optional post-MLP pruning inside the candidate set. This controls
            # the attention graph but does not change pairwise MLP cost.
            if self.topk > 0 and self.topk < k_cand:
                post_idx = torch.topk(edge_logits_sparse, k=self.topk, dim=-1).indices
                post_mask = torch.zeros_like(edge_logits_sparse, dtype=torch.bool)
                post_mask.scatter_(-1, post_idx, True)
                edge_logits_sparse = edge_logits_sparse.masked_fill(~post_mask, -1e9)

            edge_weights_sparse = torch.softmax(edge_logits_sparse, dim=-1)
            edge_weights_sparse = torch.nan_to_num(edge_weights_sparse, nan=0.0, posinf=0.0, neginf=0.0)
            ego_context = torch.einsum("bik,bikd->bid", edge_weights_sparse, pair_repr_sparse)

            # Scatter sparse tensors to dense diagnostic views.
            pair_scatter_idx = idx.unsqueeze(-1).expand(-1, -1, -1, relation_dim)
            pair_repr_dense.scatter_(2, pair_scatter_idx, pair_repr_sparse)
            edge_logits_dense.scatter_(2, idx, edge_logits_sparse)
            edge_mask_sparse = torch.isfinite(edge_logits_sparse) & (edge_logits_sparse > -1e8)
            edge_mask_dense.scatter_(2, idx, edge_mask_sparse)
            edge_weights_dense = agent_embeddings.new_zeros((bsz, n_agents, n_agents))
            edge_weights_dense.scatter_(2, idx, edge_weights_sparse)
        else:
            ego_context = agent_embeddings.new_zeros((bsz, n_agents, relation_dim))
            edge_weights_dense = agent_embeddings.new_zeros((bsz, n_agents, n_agents))

        ego_context = self.context_proj(ego_context)
        context_similarity = F.cosine_similarity(ego_context.unsqueeze(2), ego_context.unsqueeze(1), dim=-1)

        denom = float(n_agents * max(n_agents - 1, 1))
        candidate_edges = float(n_agents * k_cand)
        candidate_edge_frac = torch.tensor(candidate_edges / denom, device=device)
        effective_edge_count = edge_mask_dense.float().sum(dim=-1).mean() if n_agents > 0 else torch.tensor(0.0, device=device)
        effective_edge_frac = effective_edge_count / float(max(n_agents - 1, 1))

        return RelationBatch(
            pair_repr=pair_repr_dense,
            edge_logits=edge_logits_dense,
            edge_weights=edge_weights_dense,
            ego_context=ego_context,
            context_similarity=context_similarity,
            edge_mask=edge_mask_dense,
            candidate_edge_frac=candidate_edge_frac,
            effective_edge_count=effective_edge_count,
            effective_edge_frac=effective_edge_frac,
        )


class RelationDynamicsPredictor(nn.Module):
    """Predicts next-step agent embedding from current embedding and relation context."""

    def __init__(self, args):
        super().__init__()
        agent_dim = args.agent_embedding_dim
        relation_dim = getattr(args, "relation_dim", 64)
        hidden_dim = getattr(args, "relation_dynamics_hidden_dim", getattr(args, "relation_hidden_dim", 128))
        self.predictor = nn.Sequential(
            nn.Linear(agent_dim + relation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, agent_dim),
        )

    def forward(self, agent_embeddings: torch.Tensor, relation_context: torch.Tensor) -> torch.Tensor:
        return self.predictor(torch.cat([agent_embeddings, relation_context], dim=-1))


def _topk_mask(scores: torch.Tensor, candidate_mask: torch.Tensor, k: int, largest: bool) -> torch.Tensor:
    """Build a boolean top-k/bottom-k mask per row."""
    bsz, n_agents, _ = scores.shape
    out = torch.zeros_like(candidate_mask, dtype=torch.bool)
    if k <= 0 or n_agents <= 1:
        return out
    k_eff = min(k, max(n_agents - 1, 1))
    if largest:
        masked_scores = scores.masked_fill(~candidate_mask, -1e9)
        idx = torch.topk(masked_scores, k=k_eff, dim=-1, largest=True).indices
    else:
        masked_scores = scores.masked_fill(~candidate_mask, 1e9)
        idx = torch.topk(masked_scores, k=k_eff, dim=-1, largest=False).indices
    out.scatter_(-1, idx, True)
    return out & candidate_mask


def _weighted_mean_per_batch(value: torch.Tensor, mask: torch.Tensor, batch_weight: torch.Tensor) -> torch.Tensor:
    """Mean over selected pair values, then weighted over batch."""
    selected_sum = (value * mask.float()).sum(dim=(1, 2), keepdim=True)
    selected_cnt = mask.float().sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
    per_batch = selected_sum / selected_cnt
    return (per_batch * batch_weight).sum() / batch_weight.sum().clamp_min(1.0)


def relation_driven_infonce(
    role_query: torch.Tensor,
    role_key: torch.Tensor,
    bilinear_w: torch.Tensor,
    relation_similarity: torch.Tensor,
    pos_threshold: float = 0.65,
    neg_threshold: float = 0.35,
    temperature: float = 0.2,
    active_mask: Optional[torch.Tensor] = None,
    fallback_neg_k: int = 2,
    fallback_pos_k: int = 1,
    sampling_mode: str = "rank",
    pos_k: int = 1,
    neg_k: int = 1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """InfoNCE where positives/negatives are selected by relation similarity.

    sampling_mode:
        rank:      positives are TopK relation-similar non-self agents;
                   negatives are BottomK relation-dissimilar non-self agents.
        threshold: v2 threshold logic with top/bottom fallback.
        hybrid:    threshold positives if available, otherwise TopK positives;
                   BottomK negatives are always used.
    """
    bsz, n_agents, _ = role_query.shape
    w = bilinear_w.squeeze(0) if bilinear_w.dim() == 3 else bilinear_w

    logits = torch.matmul(torch.matmul(role_query, w), role_key.transpose(1, 2))
    logits = logits / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=-1, keepdim=True)[0].detach()

    eye = torch.eye(n_agents, device=logits.device, dtype=torch.bool).unsqueeze(0).expand(bsz, -1, -1)
    nonself = ~eye
    mode = str(sampling_mode).lower()
    if mode not in {"rank", "threshold", "hybrid"}:
        raise ValueError(f"Unsupported relation_sampling_mode: {sampling_mode}")

    if mode == "rank":
        pos_mask_inter = _topk_mask(relation_similarity, nonself, pos_k, largest=True)
        neg_candidates = nonself & (~pos_mask_inter)
        neg_mask_inter = _topk_mask(relation_similarity, neg_candidates, neg_k, largest=False)
        no_neg = neg_mask_inter.sum(dim=-1, keepdim=True) == 0
        fallback_neg = _topk_mask(relation_similarity, nonself, neg_k, largest=False)
        neg_mask_inter = torch.where(no_neg.expand_as(neg_mask_inter), fallback_neg, neg_mask_inter)
        pos_mask_inter = pos_mask_inter & (~neg_mask_inter)
        no_pos = pos_mask_inter.sum(dim=-1, keepdim=True) == 0
    elif mode == "threshold":
        raw_pos_mask = (relation_similarity >= pos_threshold) & nonself
        raw_neg_mask = (relation_similarity <= neg_threshold) & nonself
        raw_neg_mask = raw_neg_mask & (~raw_pos_mask)
        pos_mask_inter = raw_pos_mask.clone()
        neg_mask_inter = raw_neg_mask.clone()

        no_pos = pos_mask_inter.sum(dim=-1, keepdim=True) == 0
        pos_candidates = nonself & (~neg_mask_inter)
        fallback_pos = _topk_mask(relation_similarity, pos_candidates, fallback_pos_k, largest=True)
        pos_mask_inter = torch.where(no_pos.expand_as(pos_mask_inter), fallback_pos, pos_mask_inter)

        no_neg = neg_mask_inter.sum(dim=-1, keepdim=True) == 0
        neg_candidates = nonself & (~pos_mask_inter)
        fallback_neg_from_candidates = _topk_mask(relation_similarity, neg_candidates, fallback_neg_k, largest=False)
        fallback_neg_from_all = _topk_mask(relation_similarity, nonself, fallback_neg_k, largest=False)
        has_neg_candidate = neg_candidates.sum(dim=-1, keepdim=True) > 0
        fallback_neg = torch.where(
            has_neg_candidate.expand_as(fallback_neg_from_candidates),
            fallback_neg_from_candidates,
            fallback_neg_from_all,
        )
        neg_mask_inter = torch.where(no_neg.expand_as(neg_mask_inter), fallback_neg, neg_mask_inter)
        pos_mask_inter = pos_mask_inter & (~neg_mask_inter)
        neg_mask_inter = neg_mask_inter & (~pos_mask_inter)
    else:  # hybrid
        raw_pos_mask = (relation_similarity >= pos_threshold) & nonself
        no_pos = raw_pos_mask.sum(dim=-1, keepdim=True) == 0
        rank_pos = _topk_mask(relation_similarity, nonself, pos_k, largest=True)
        pos_mask_inter = torch.where(no_pos.expand_as(raw_pos_mask), rank_pos, raw_pos_mask)
        neg_candidates = nonself & (~pos_mask_inter)
        neg_mask_inter = _topk_mask(relation_similarity, neg_candidates, neg_k, largest=False)
        no_neg = neg_mask_inter.sum(dim=-1, keepdim=True) == 0
        fallback_neg = _topk_mask(relation_similarity, nonself, neg_k, largest=False)
        neg_mask_inter = torch.where(no_neg.expand_as(neg_mask_inter), fallback_neg, neg_mask_inter)
        pos_mask_inter = pos_mask_inter & (~neg_mask_inter)

    # Self is only a positive key for numerical stability. It is not counted in
    # inter-agent pos/neg diagnostics.
    pos_mask = pos_mask_inter | eye
    neg_mask = neg_mask_inter
    denominator_mask = pos_mask | neg_mask

    exp_logits = torch.exp(logits) * denominator_mask.float()
    pos_exp = (exp_logits * pos_mask.float()).sum(dim=-1).clamp_min(1e-8)
    denom_exp = exp_logits.sum(dim=-1).clamp_min(1e-8)
    per_anchor_loss = -torch.log(pos_exp / denom_exp)

    if active_mask is not None:
        mask = active_mask.view(bsz, 1).float()
        loss = (per_anchor_loss * mask).sum() / (mask.sum() * n_agents + 1e-8)
        batch_weight = mask.view(bsz, 1, 1)
    else:
        loss = per_anchor_loss.mean()
        batch_weight = torch.ones((bsz, 1, 1), device=logits.device)

    edge_count = float(n_agents * max(n_agents - 1, 1))
    active_b = batch_weight.sum().clamp_min(1.0)
    pos_frac = ((pos_mask_inter.float().sum(dim=(1, 2), keepdim=True) / edge_count) * batch_weight).sum() / active_b
    neg_frac = ((neg_mask_inter.float().sum(dim=(1, 2), keepdim=True) / edge_count) * batch_weight).sum() / active_b

    sim_nonself = relation_similarity.masked_select(nonself)
    if sim_nonself.numel() > 0:
        sim_mean = sim_nonself.mean()
        sim_std = sim_nonself.std(unbiased=False)
    else:
        sim_mean = torch.tensor(0.0, device=logits.device)
        sim_std = torch.tensor(0.0, device=logits.device)

    overlap_frac = ((pos_mask_inter & neg_mask_inter).float().sum(dim=(1, 2), keepdim=True) / edge_count * batch_weight).sum() / active_b
    if active_mask is not None:
        active_episode = active_mask.view(bsz, 1, 1).float()
        no_pos_frac = (no_pos.float().mean(dim=1, keepdim=True) * active_episode).sum() / (active_episode.sum() + 1e-8)
        no_neg_frac = (no_neg.float().mean(dim=1, keepdim=True) * active_episode).sum() / (active_episode.sum() + 1e-8)
    else:
        no_pos_frac = no_pos.float().mean()
        no_neg_frac = no_neg.float().mean()

    pos_sim_mean = _weighted_mean_per_batch(relation_similarity, pos_mask_inter, batch_weight)
    neg_sim_mean = _weighted_mean_per_batch(relation_similarity, neg_mask_inter, batch_weight)
    rank_margin = pos_sim_mean - neg_sim_mean

    mode_id = {"threshold": 0.0, "rank": 1.0, "hybrid": 2.0}[mode]
    metrics = {
        "relation_loss": loss.detach(),
        "relation_pos_frac": pos_frac.detach(),
        "relation_neg_frac": neg_frac.detach(),
        "relation_mask_overlap_frac": overlap_frac.detach(),
        "relation_no_pos_anchor_frac": no_pos_frac.detach(),
        "relation_no_neg_anchor_frac": no_neg_frac.detach(),
        "relation_pos_rank_sim_mean": pos_sim_mean.detach(),
        "relation_neg_rank_sim_mean": neg_sim_mean.detach(),
        "relation_rank_margin": rank_margin.detach(),
        "relation_pos_k_frac": torch.tensor(float(pos_k) / max(n_agents - 1, 1), device=logits.device),
        "relation_neg_k_frac": torch.tensor(float(neg_k) / max(n_agents - 1, 1), device=logits.device),
        "relation_sampling_mode_id": torch.tensor(mode_id, device=logits.device),
        "relation_sim_mean": sim_mean.detach(),
        "relation_sim_std": sim_std.detach(),
        "relation_logits_mean": logits.detach().mean(),
        "relation_logits_std": logits.detach().std(unbiased=False),
        "relation_sim_hist": relation_similarity.detach().reshape(-1).cpu(),
    }
    return loss, metrics


def edge_entropy(edge_weights: torch.Tensor) -> torch.Tensor:
    """Mean row entropy of learned relation attention."""
    p = edge_weights.clamp_min(1e-8)
    return (-(p * torch.log(p)).sum(dim=-1)).mean()
