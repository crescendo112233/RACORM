"""
Simple relation-conditioned RACORM / R2Role-QMIX.

This version intentionally removes the ACORM-style auxiliary machinery:
    - no K-means role pair mining;
    - no relation contrastive loss;
    - no relation dynamics prediction loss;
    - no ACORM state-to-role mixer attention by default.

The core hypothesis is tested directly:
    role_i = f(agent_embedding_i, relation_context_i)

Relation context R_i is computed by RelationContextEncoder from all agents'
embeddings and the global state. This is a centralized diagnostic version and is
not constrained to decentralized execution / strict CTDE.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from algorithm.acorm import ACORM_Agent
from algorithm.relation_context import RelationContextEncoder, edge_entropy


class RelationRoleFusion(nn.Module):
    """Fuse base ACORM role z_i with relation context R_i.

    The default is a gated residual fusion:
        r_i = W_R R_i
        g_i = sigmoid(W_g [z_i, r_i])
        z_i^rel = LN(z_i + lambda * g_i * r_i)

    This keeps a safe fallback to the original ACORM role when relation context is
    noisy: the gate can approach zero.
    """

    def __init__(self, role_dim: int, relation_dim: int):
        super().__init__()
        self.rel_proj = nn.Linear(relation_dim, role_dim)
        self.gate = nn.Sequential(
            nn.Linear(role_dim * 2, role_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(role_dim)

    def forward(self, base_role: torch.Tensor, ego_context: torch.Tensor, inject_weight: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rel_role = self.rel_proj(ego_context)
        gate = self.gate(torch.cat([base_role, rel_role], dim=-1))
        fused_role = self.norm(base_role + float(inject_weight) * gate * rel_role)
        return fused_role, gate, rel_role


class RelationACORM_Agent(ACORM_Agent):
    """Direct relation-conditioned role variant.

    This class keeps ACORM's trajectory encoder, role encoder, individual Q
    network and QMIX training loop, but changes how roles are produced:
        z_i = GatedFuse(ACORMRole(e_i), R_i)

    It does not train RACORM's previous contrastive/dynamics objectives. The
    relation encoder is trained directly through the TD loss because R_i enters
    Q_i through z_i.
    """

    def __init__(self, args):
        super().__init__(args)
        self.use_relation_conditioned_role = bool(getattr(args, "use_relation_conditioned_role", True))
        self.use_mixer_role_attention = bool(getattr(args, "use_mixer_role_attention", False))
        self.relation_inject_weight = float(getattr(args, "relation_inject_weight", 0.2))
        self.relation_inject_warmup_steps = int(getattr(args, "relation_inject_warmup_steps", 0))
        self.skip_recl_pretrain = bool(getattr(args, "skip_recl_pretrain", True))

        self.relation_encoder = RelationContextEncoder(args).to(self.device)
        self.relation_role_fusion = RelationRoleFusion(
            role_dim=self.role_embedding_dim,
            relation_dim=int(getattr(args, "relation_dim", self.role_embedding_dim)),
        ).to(self.device)

        # Rebuild the role-side optimiser so relation encoder and relation-role
        # fusion are trained through L_TD together with the agent/role encoders.
        base_role_params = list(self.RECL.role_embedding_net.parameters()) + list(self.RECL.agent_embedding_net.parameters())
        relation_role_params = list(self.relation_encoder.parameters()) + list(self.relation_role_fusion.parameters())
        self.role_parameters = base_role_params + relation_role_params
        self.role_embedding_optimizer = torch.optim.Adam(
            [
                {"params": base_role_params, "lr": self.lr},
                {"params": relation_role_params, "lr": float(getattr(args, "relation_lr", self.lr))},
            ],
            lr=self.lr,
        )
        self.role_lr_decay = StepLR(
            self.role_embedding_optimizer,
            step_size=self.lr_decay_steps,
            gamma=self.lr_decay_rate,
        )
        self.last_train_metrics: Dict[str, torch.Tensor] = {}

    def _current_inject_weight(self) -> float:
        if not self.use_relation_conditioned_role:
            return 0.0
        if self.relation_inject_warmup_steps <= 0:
            return self.relation_inject_weight
        # train_step is an update counter, not environment steps. This is enough
        # for a soft warmup diagnostic and avoids coupling to Runner internals.
        scale = min(1.0, float(self.train_step) / float(self.relation_inject_warmup_steps))
        return self.relation_inject_weight * scale

    def _fuse_roles(self, base_roles: torch.Tensor, ego_context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.use_relation_conditioned_role:
            zeros = torch.zeros_like(base_roles)
            return base_roles, zeros, zeros
        return self.relation_role_fusion(base_roles, ego_context, self._current_inject_weight())

    def _relation_metrics(self, rel, gate: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        metrics: Dict[str, torch.Tensor] = {
            "relation_candidate_edge_frac": rel.candidate_edge_frac.detach().cpu(),
            "relation_effective_edge_count": rel.effective_edge_count.detach().cpu(),
            "relation_effective_edge_frac": rel.effective_edge_frac.detach().cpu(),
            "edge_entropy": edge_entropy(rel.edge_weights).detach().cpu(),
            "edge_max": rel.edge_weights.max(dim=-1)[0].mean().detach().cpu(),
            "relation_sim_mean": rel.context_similarity.detach().mean().cpu(),
            "relation_sim_std": rel.context_similarity.detach().std().cpu(),
            "relation_sparse_topk": torch.tensor(float(getattr(self.args, "relation_sparse_topk", 0))),
            "relation_inject_weight_current": torch.tensor(float(self._current_inject_weight())),
            "mixer_role_attention_enabled": torch.tensor(float(self.use_mixer_role_attention)),
        }
        if gate is not None and gate.numel() > 0:
            metrics["relation_gate_mean"] = gate.detach().mean().cpu()
            metrics["relation_gate_std"] = gate.detach().std().cpu()
            metrics["relation_gate_min"] = gate.detach().min().cpu()
            metrics["relation_gate_max"] = gate.detach().max().cpu()
        return metrics

    def get_role_embedding(self, obs_n, last_a, state=None):
        """Return relation-conditioned role embeddings for online action selection.

        This method uses all agents' embeddings and optionally the global state;
        therefore it is a centralized diagnostic version rather than strict CTDE.
        """
        recl_obs = torch.tensor(np.array(obs_n), dtype=torch.float32, device=self.device)
        recl_last_a = torch.tensor(np.array(last_a), dtype=torch.float32, device=self.device)
        agent_embedding = self.RECL.agent_embedding_forward(recl_obs, recl_last_a, detach=True)
        base_role = self.RECL.role_embedding_forward(agent_embedding, detach=True, ema=False)

        if not self.use_relation_conditioned_role:
            return base_role

        agent_embedding_b = agent_embedding.reshape(1, self.N, self.agent_embedding_dim)
        if getattr(self.args, "relation_use_state", True):
            if state is None:
                # Keep the method robust for older callers. Zero state is not a
                # scientific choice; Runner should pass the real state.
                state_t = torch.zeros((1, self.state_dim), dtype=torch.float32, device=self.device)
            else:
                state_t = torch.tensor(np.array(state), dtype=torch.float32, device=self.device).reshape(1, self.state_dim)
        else:
            state_t = None
        rel = self.relation_encoder(agent_embedding_b, state_t)
        fused_role, _, _ = self._fuse_roles(base_role.reshape(1, self.N, self.role_embedding_dim), rel.ego_context)
        return fused_role.reshape(self.N, self.role_embedding_dim)

    def pretrain_recl(self, replay_buffer):
        """No K-means/contrastive role pretraining in the simple direct version."""
        zero = torch.tensor(0.0, device=self.device)
        self.last_train_metrics = {
            "pretrain/recl_loss_skipped": zero.detach().cpu(),
        }
        return zero

    def train(self, replay_buffer):
        self.train_step += 1
        batch, max_episode_len = replay_buffer.sample(self.batch_size)
        inputs, batch_o, batch_s, batch_r, batch_a, batch_last_a, batch_avail_a_n, batch_active, batch_dw = self.get_inputs(batch)

        qmix_loss, aux_metrics = self.update_qmix(
            inputs, batch_o, batch_s, batch_r, batch_a, batch_last_a,
            batch_avail_a_n, batch_active, batch_dw, max_episode_len,
        )
        metrics: Dict[str, torch.Tensor] = {}
        if qmix_loss is not None:
            metrics["train/qmix_loss"] = qmix_loss.detach().cpu()
        for k, v in aux_metrics.items():
            metrics[f"train/{k}"] = v

        if self.use_hard_update:
            if self.train_step % self.target_update_freq == 0:
                self.target_Q_net.load_state_dict(self.eval_Q_net.state_dict())
                self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())
        else:
            self.soft_update_params(self.eval_Q_net, self.target_Q_net, self.tau)
            self.soft_update_params(self.eval_mix_net, self.target_mix_net, self.tau)
            self.soft_update_params(self.RECL.role_embedding_net, self.RECL.role_embedding_target_net, self.tau)

        if self.use_lr_decay:
            self.qmix_lr_decay.step()
            self.role_lr_decay.step()

        self.last_train_metrics = metrics
        return metrics

    def _compute_relation_conditioned_roles_batch(self, batch_o, batch_last_a, batch_s, max_episode_len):
        """Compute z_i=f(e_i,R_i) for all t=0...T.

        Returns:
            agent_embeddings: (B, T+1, N, D_e)
            role_embeddings:  (B, T+1, N, D_z)
            metrics: averaged detached diagnostics for TensorBoard
        """
        agent_embeddings, base_roles = self.RECL.batch_role_embed_forward(batch_o, batch_last_a, max_episode_len, detach=False)
        bsz = batch_o.shape[0]
        fused_roles = []
        metrics_acc: Dict[str, list] = {}

        for t in range(max_episode_len + 1):
            rel = self.relation_encoder(agent_embeddings[:, t], batch_s[:, t])
            fused_t, gate_t, _ = self._fuse_roles(base_roles[:, t], rel.ego_context)
            fused_roles.append(fused_t)
            step_metrics = self._relation_metrics(rel, gate_t)
            for k, v in step_metrics.items():
                metrics_acc.setdefault(k, []).append(v.float() if torch.is_tensor(v) else torch.tensor(float(v)))

        role_embeddings = torch.stack(fused_roles, dim=1).reshape(bsz, max_episode_len + 1, self.N, self.role_embedding_dim)
        metrics = {k: torch.stack(v).mean().detach().cpu() for k, v in metrics_acc.items()}
        return agent_embeddings, role_embeddings, metrics

    def update_qmix(self, inputs, batch_o, batch_s, batch_r, batch_a, batch_last_a, batch_avail_a_n, batch_active, batch_dw, max_episode_len):
        """QMIX update with relation-conditioned roles and optional mixer MHA removal."""
        self.eval_Q_net.rnn_hidden = None
        self.target_Q_net.rnn_hidden = None

        _, role_embeddings, rel_metrics = self._compute_relation_conditioned_roles_batch(
            batch_o, batch_last_a, batch_s, max_episode_len
        )
        inputs = torch.cat([inputs, role_embeddings], dim=-1)

        q_evals, q_targets = [], []
        self.eval_mix_net.state_gru_hidden = None
        fc_batch_s = F.relu(self.eval_mix_net.state_fc(batch_s.reshape(-1, self.state_dim))).reshape(-1, max_episode_len + 1, self.state_dim)
        state_gru_outs = []

        for t in range(max_episode_len):
            q_eval = self.eval_Q_net(inputs[:, t].reshape(-1, self.QMIX_input_dim))
            q_target = self.target_Q_net(inputs[:, t + 1].reshape(-1, self.QMIX_input_dim))
            q_evals.append(q_eval.reshape(self.batch_size, self.N, -1))
            q_targets.append(q_target.reshape(self.batch_size, self.N, -1))
            self.eval_mix_net.state_gru_hidden = self.eval_mix_net.state_gru(
                fc_batch_s[:, t].reshape(-1, self.state_dim),
                self.eval_mix_net.state_gru_hidden,
            )
            state_gru_outs.append(self.eval_mix_net.state_gru_hidden)

        self.eval_mix_net.state_gru_hidden = self.eval_mix_net.state_gru(
            fc_batch_s[:, max_episode_len].reshape(-1, self.state_dim),
            self.eval_mix_net.state_gru_hidden,
        )
        state_gru_outs.append(self.eval_mix_net.state_gru_hidden)
        state_gru_outs = torch.stack(state_gru_outs, dim=1).reshape(-1, self.N, self.args.state_embed_dim)
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)

        with torch.no_grad():
            q_eval_last = self.eval_Q_net(inputs[:, -1].reshape(-1, self.QMIX_input_dim)).reshape(self.batch_size, 1, self.N, -1)
            q_evals_next = torch.cat([q_evals[:, 1:], q_eval_last], dim=1)
            q_evals_next[batch_avail_a_n[:, 1:] == 0] = -999999
            a_argmax = torch.argmax(q_evals_next, dim=-1, keepdim=True)
            q_targets = torch.gather(q_targets, dim=-1, index=a_argmax).squeeze(-1)

        q_evals = torch.gather(q_evals, dim=-1, index=batch_a.unsqueeze(-1)).squeeze(-1)
        flat_roles = role_embeddings.reshape(-1, self.N, self.role_embedding_dim)

        if self.use_mixer_role_attention:
            att_eval = self.eval_mix_net.attention_net(state_gru_outs, flat_roles, flat_roles).reshape(
                -1, max_episode_len + 1, self.N * self.att_out_dim
            )
            with torch.no_grad():
                att_target = self.target_mix_net.attention_net(state_gru_outs, flat_roles, flat_roles).reshape(
                    -1, max_episode_len + 1, self.N * self.att_out_dim
                )
        else:
            # Remove ACORM's backend state-to-role attention. The existing mix
            # network still expects an attention-sized input, so feed zeros. This
            # removes the MHA computation and prevents the mixer from depending
            # on another role-attention pathway.
            att_shape = (self.batch_size, max_episode_len + 1, self.N * self.att_out_dim)
            att_eval = batch_s.new_zeros(att_shape)
            att_target = batch_s.new_zeros(att_shape)

        q_total_eval = self.eval_mix_net(q_evals, fc_batch_s[:, :-1], att_eval[:, :-1])
        q_total_target = self.target_mix_net(q_targets, fc_batch_s[:, 1:], att_target[:, 1:])
        targets = batch_r + self.gamma * (1 - batch_dw) * q_total_target
        td_error = q_total_eval - targets.detach()
        mask_td_error = td_error * batch_active
        loss = (mask_td_error ** 2).sum() / batch_active.sum()

        self.optimizer.zero_grad()
        self.role_embedding_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.role_parameters, 10)
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, 10)
        self.optimizer.step()
        self.role_embedding_optimizer.step()

        rel_metrics["td_error_abs_mean"] = td_error.detach().abs().mean().cpu()
        rel_metrics["q_total_eval_mean"] = q_total_eval.detach().mean().cpu()
        rel_metrics["q_total_target_mean"] = q_total_target.detach().mean().cpu()
        return loss.detach(), rel_metrics
