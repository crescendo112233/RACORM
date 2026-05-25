"""
Relation-context modules for Relation-aware ACORM (RACORM).

This module implements Idea 1 in a conservative way: the relation graph is used
primarily to define positive/negative pairs for ACORM's role contrastive loss.
It does not replace ACORM's original attention-based mixer, so the experiment can
separate "role learning supervision" from "value decomposition attention".

Version v2 fixes three issues found during training diagnostics:
    1) positive and negative masks are forced to be mutually exclusive;
    2) if no threshold-based negative exists, bottom-k least similar non-self
       agents are used as fallback negatives instead of all non-self agents;
    3) a differentiable relation dynamics predictor is provided so the relation
       encoder receives a real training signal beyond hard mask selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RelationBatch:
    pair_repr: torch.Tensor            # (B, N, N, relation_dim)
    edge_logits: torch.Tensor          # (B, N, N)
    edge_weights: torch.Tensor         # (B, N, N), row-normalized over neighbor j
    ego_context: torch.Tensor          # (B, N, relation_dim)
    context_similarity: torch.Tensor   # (B, N, N)


class RelationContextEncoder(nn.Module):
    """Learns pairwise relation features and per-agent relation contexts.

    Args expected on `args`:
        N, agent_embedding_dim, state_dim
        relation_hidden_dim, relation_dim, relation_use_state,
        relation_mask_self, relation_topk
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
        self.topk = int(getattr(args, "relation_topk", 0))

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

    def forward(self, agent_embeddings: torch.Tensor, states: Optional[torch.Tensor] = None) -> RelationBatch:
        """Compute pairwise relation contexts.

        Args:
            agent_embeddings: tensor of shape (B, N, D_e).
            states: optional tensor of shape (B, state_dim).
        """
        if agent_embeddings.dim() != 3:
            raise ValueError(f"agent_embeddings must be (B,N,D), got {tuple(agent_embeddings.shape)}")
        bsz, n_agents, _ = agent_embeddings.shape
        if n_agents != self.n_agents:
            # Keep this non-fatal; SMAC wrappers occasionally expose map-specific N.
            self.n_agents = n_agents

        e_i = agent_embeddings.unsqueeze(2).expand(-1, -1, n_agents, -1)
        e_j = agent_embeddings.unsqueeze(1).expand(-1, n_agents, -1, -1)
        pair_features = [e_i, e_j, torch.abs(e_i - e_j), e_i * e_j]

        if self.use_state:
            if states is None:
                raise ValueError("relation_use_state=True but states=None")
            s = self.state_proj(states).view(bsz, 1, 1, self.agent_dim).expand(-1, n_agents, n_agents, -1)
            pair_features.append(s)

        pair_input = torch.cat(pair_features, dim=-1)
        pair_repr = self.edge_mlp(pair_input)
        edge_logits = self.edge_score(pair_repr).squeeze(-1)

        if self.mask_self:
            eye = torch.eye(n_agents, device=edge_logits.device, dtype=torch.bool).unsqueeze(0)
            edge_logits = edge_logits.masked_fill(eye, -1e9)

        if self.topk > 0 and self.topk < n_agents:
            topk_idx = torch.topk(edge_logits, k=self.topk, dim=-1).indices
            topk_mask = torch.zeros_like(edge_logits, dtype=torch.bool)
            topk_mask.scatter_(-1, topk_idx, True)
            if self.mask_self:
                eye = torch.eye(n_agents, device=edge_logits.device, dtype=torch.bool).unsqueeze(0)
                topk_mask = topk_mask & (~eye)
            edge_logits = edge_logits.masked_fill(~topk_mask, -1e9)

        edge_weights = torch.softmax(edge_logits, dim=-1)
        # If all edges in a row are masked, softmax can produce NaNs. This should
        # not happen for N>1 with topk>=1, but sanitize for robustness.
        edge_weights = torch.nan_to_num(edge_weights, nan=0.0, posinf=0.0, neginf=0.0)

        ego_context = torch.einsum("bij,bijd->bid", edge_weights, pair_repr)
        ego_context = self.context_proj(ego_context)
        context_similarity = F.cosine_similarity(ego_context.unsqueeze(2), ego_context.unsqueeze(1), dim=-1)

        return RelationBatch(
            pair_repr=pair_repr,
            edge_logits=edge_logits,
            edge_weights=edge_weights,
            ego_context=ego_context,
            context_similarity=context_similarity,
        )


class RelationDynamicsPredictor(nn.Module):
    """Predicts next-step agent embedding from current embedding and relation context.

    This auxiliary module supplies a differentiable training objective for the
    relation encoder. The target embedding should be stop-gradient; gradients
    flow through the relation context and the predictor, not into the agent
    trajectory encoder target.
    """

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
        """Return predicted next agent embeddings.

        Args:
            agent_embeddings: (B, N, D_e)
            relation_context: (B, N, D_r)
        Returns:
            predicted next embeddings: (B, N, D_e)
        """
        return self.predictor(torch.cat([agent_embeddings, relation_context], dim=-1))


def _topk_mask(
    scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    k: int,
    largest: bool,
) -> torch.Tensor:
    """Build a boolean top-k/bottom-k mask per row.

    Args:
        scores: (B, N, N)
        candidate_mask: (B, N, N), True entries are eligible.
        k: number of candidates per anchor row.
        largest: True for top-k, False for bottom-k.
    """
    bsz, n_agents, _ = scores.shape
    out = torch.zeros_like(candidate_mask, dtype=torch.bool)
    if k <= 0 or n_agents <= 1:
        return out

    # topk requires a fixed k. Use min(k, N-1); invalid entries are pushed to
    # +/- infinity and removed after scatter.
    k_eff = min(k, max(n_agents - 1, 1))
    if largest:
        masked_scores = scores.masked_fill(~candidate_mask, -1e9)
        idx = torch.topk(masked_scores, k=k_eff, dim=-1, largest=True).indices
    else:
        masked_scores = scores.masked_fill(~candidate_mask, 1e9)
        idx = torch.topk(masked_scores, k=k_eff, dim=-1, largest=False).indices

    out.scatter_(-1, idx, True)
    out = out & candidate_mask
    return out


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
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """InfoNCE where positives/negatives are selected by relation similarity.

    v2 behavior:
      - positive and negative inter-agent masks are forced to be mutually
        exclusive;
      - if thresholding yields no negatives for an anchor, the least similar
        bottom-k non-self agents are used as fallback negatives;
      - if all non-self agents are initially positive, the fallback negatives are
        removed from positives to keep masks disjoint.

    Args:
        role_query: (B, N, D_z), query encoder output.
        role_key: (B, N, D_z), momentum/EMA key encoder output.
        bilinear_w: (D_z, D_z), ACORM's learnable bilinear matrix W.
        relation_similarity: (B, N, N), cosine similarity between relation contexts.
        active_mask: optional (B,) or (B,1) episode-valid mask.
        fallback_neg_k: number of bottom-k negatives when no threshold negative exists.
        fallback_pos_k: number of top-k positives when no threshold positive exists.
    """
    bsz, n_agents, role_dim = role_query.shape
    w = bilinear_w
    if w.dim() == 3:
        w = w.squeeze(0)

    logits = torch.matmul(torch.matmul(role_query, w), role_key.transpose(1, 2))
    logits = logits / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=-1, keepdim=True)[0].detach()

    eye = torch.eye(n_agents, device=logits.device, dtype=torch.bool).unsqueeze(0).expand(bsz, -1, -1)
    nonself = ~eye

    raw_pos_mask = (relation_similarity >= pos_threshold) & nonself
    raw_neg_mask = (relation_similarity <= neg_threshold) & nonself
    raw_neg_mask = raw_neg_mask & (~raw_pos_mask)  # enforce disjoint raw masks

    pos_mask_inter = raw_pos_mask.clone()
    neg_mask_inter = raw_neg_mask.clone()

    # Positive fallback: if an anchor has no inter-agent positive, choose the
    # top-k most similar non-self candidates. This is separate from self-positive.
    no_pos = pos_mask_inter.sum(dim=-1, keepdim=True) == 0
    pos_candidates = nonself & (~neg_mask_inter)
    fallback_pos = _topk_mask(relation_similarity, pos_candidates, fallback_pos_k, largest=True)
    pos_mask_inter = torch.where(no_pos.expand_as(pos_mask_inter), fallback_pos, pos_mask_inter)

    # Negative fallback: if no threshold negative exists, choose bottom-k least
    # similar non-self candidates. Prefer candidates that are not already
    # positive; if none exist, allow bottom-k from all non-self and remove them
    # from positives to maintain mutual exclusion.
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

    # Final hard guarantee: inter-agent positives and negatives are disjoint.
    pos_mask_inter = pos_mask_inter & (~neg_mask_inter)
    neg_mask_inter = neg_mask_inter & (~pos_mask_inter)

    # Self is only a positive key for numerical stability. It is not counted in
    # pos_frac/neg_frac diagnostics.
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

    # Diagnostics for TensorBoard. Fractions are computed over inter-agent edges
    # and optionally weighted by active episodes.
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

    metrics = {
        "relation_loss": loss.detach(),
        "relation_pos_frac": pos_frac.detach(),
        "relation_neg_frac": neg_frac.detach(),
        "relation_mask_overlap_frac": overlap_frac.detach(),
        "relation_no_pos_anchor_frac": no_pos_frac.detach(),
        "relation_no_neg_anchor_frac": no_neg_frac.detach(),
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
