"""
Relation-aware ACORM (RACORM) for the ACORM_QMIX codebase.

RACORM subclasses the original ACORM_Agent and replaces ACORM's K-means-based
contrastive role supervision with relation-context-driven contrastive sampling.
The value decomposition path is intentionally left compatible with ACORM's
original attention mixer so that the main experimental difference is the role
learning signal rather than another mixer change.

v2 changes:
    1) positive/negative masks are mutually exclusive;
    2) threshold-negative fallback uses bottom-k least similar non-self agents;
    3) relation dynamics prediction is added so the relation encoder receives a
       differentiable learning signal.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from algorithm.acorm import ACORM_Agent
from algorithm.relation_context import (
    RelationContextEncoder,
    RelationDynamicsPredictor,
    edge_entropy,
    relation_driven_infonce,
)


class RelationACORM_Agent(ACORM_Agent):
    """ACORM with relation-context-driven role contrastive learning."""

    def __init__(self, args):
        super().__init__(args)
        self.relation_loss_weight = float(getattr(args, "relation_loss_weight", 1.0))
        self.kmeans_loss_weight = float(getattr(args, "kmeans_loss_weight", 0.0))
        self.relation_temperature = float(getattr(args, "relation_temperature", 0.2))
        self.relation_pos_threshold = float(getattr(args, "relation_pos_threshold", 0.65))
        self.relation_neg_threshold = float(getattr(args, "relation_neg_threshold", 0.35))
        self.relation_fallback_neg_k = int(getattr(args, "relation_fallback_neg_k", 2))
        self.relation_fallback_pos_k = int(getattr(args, "relation_fallback_pos_k", 1))
        self.relation_dynamics_loss_weight = float(getattr(args, "relation_dynamics_loss_weight", 0.1))

        self.relation_encoder = RelationContextEncoder(args).to(self.device)
        self.relation_dynamics_predictor = RelationDynamicsPredictor(args).to(self.device)
        self.relation_parameters = list(self.relation_encoder.parameters()) + list(self.relation_dynamics_predictor.parameters())
        self.relation_optimizer = torch.optim.Adam(
            self.relation_parameters,
            lr=float(getattr(args, "relation_lr", self.recl_lr)),
        )
        self.relation_lr_decay = StepLR(
            self.relation_optimizer,
            step_size=self.lr_decay_steps,
            gamma=self.lr_decay_rate,
        )
        self.last_train_metrics: Dict[str, torch.Tensor] = {}

    def train(self, replay_buffer):
        """Same high-level loop as ACORM_Agent.train, but returns TB metrics."""
        self.train_step += 1
        batch, max_episode_len = replay_buffer.sample(self.batch_size)
        inputs, batch_o, batch_s, batch_r, batch_a, batch_last_a, batch_avail_a_n, batch_active, batch_dw = self.get_inputs(batch)

        metrics: Dict[str, torch.Tensor] = {}
        if self.train_step % self.train_recl_freq == 0:
            relation_metrics = self.update_recl(batch_o, batch_last_a, batch_s, batch_active, max_episode_len)
            self.soft_update_params(self.RECL.role_embedding_net, self.RECL.role_embedding_target_net, self.role_tau)
            metrics.update({f"train/{k}": v for k, v in relation_metrics.items()})

        qmix_loss = self.update_qmix(
            inputs, batch_o, batch_s, batch_r, batch_a, batch_last_a,
            batch_avail_a_n, batch_active, batch_dw, max_episode_len,
        )
        if qmix_loss is not None:
            metrics["train/qmix_loss"] = qmix_loss.detach().cpu()

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
            self.relation_lr_decay.step()

        self.last_train_metrics = metrics
        return metrics

    def pretrain_recl(self, replay_buffer):
        batch, max_episode_len = replay_buffer.sample(self.batch_size)
        batch_o = batch['obs_n'].to(self.device)
        batch_s = batch['s'].to(self.device)
        batch_last_a = batch['last_onehot_a_n'].to(self.device)
        batch_active = batch['active'].to(self.device)
        relation_metrics = self.update_recl(batch_o, batch_last_a, batch_s, batch_active, max_episode_len)
        self.soft_update_params(self.RECL.role_embedding_net, self.RECL.role_embedding_target_net, self.role_tau)
        self.last_train_metrics = {f"pretrain/{k}": v for k, v in relation_metrics.items()}
        return relation_metrics["relation_loss"]

    def _compute_agent_embedding_sequence(self, batch_o, batch_last_a, max_episode_len):
        """Compute stop-gradient agent embeddings for t=0...T.

        We compute the whole recurrent sequence once. This avoids corrupting the
        GRU hidden state by calling the trajectory encoder twice for current and
        next embeddings during relation dynamics prediction.
        """
        self.RECL.agent_embedding_net.rnn_hidden = None
        embeddings = []
        time_len = min(max_episode_len + 1, batch_o.shape[1], batch_last_a.shape[1])
        with torch.no_grad():
            for t in range(time_len):
                emb_t = self.RECL.agent_embedding_forward(
                    batch_o[:, t].reshape(-1, self.obs_dim),
                    batch_last_a[:, t].reshape(-1, self.action_dim),
                    detach=True,
                ).reshape(batch_o.shape[0], self.N, self.agent_embedding_dim)
                embeddings.append(emb_t.detach())
        return embeddings

    def update_recl(self, batch_o, batch_last_a, batch_s, batch_active, max_episode_len):
        """Relation-driven replacement of ACORM's K-means InfoNCE update.

        ACORM uses K-means over agent embeddings to define positive/negative role
        pairs. RACORM uses learned relation-context similarity instead:
            R_i = Pool_j g(e_i, e_j, s)
            sim_R(i,k) = cosine(R_i, R_k)
        High sim_R pairs are positives; low sim_R pairs are negatives. In v2,
        the relation encoder is trained by a differentiable relation dynamics
        prediction loss:
            e_hat_i^{t+1} = D(e_i^t, R_i^t)
            L_dyn = || e_hat_i^{t+1} - stopgrad(e_i^{t+1}) ||^2
        """
        agent_embeddings_seq = self._compute_agent_embedding_sequence(batch_o, batch_last_a, max_episode_len)
        usable_steps = min(max_episode_len, len(agent_embeddings_seq) - 1)

        total_loss = torch.tensor(0.0, device=self.device)
        total_relation_loss = torch.tensor(0.0, device=self.device)
        total_dynamics_loss = torch.tensor(0.0, device=self.device)
        total_kmeans_loss = torch.tensor(0.0, device=self.device)
        metric_acc = {}
        valid_steps = 0

        for t in range(usable_steps):
            active_t = batch_active[:, t].view(-1)  # (B,)
            if active_t.sum() <= 0.5:
                continue

            # Current and next agent embeddings are stop-gradient. The relation
            # encoder still receives gradients through rel.ego_context in the
            # dynamics predictor below.
            agent_embedding = agent_embeddings_seq[t]
            next_agent_embedding = agent_embeddings_seq[t + 1]

            role_query = self.RECL.role_embedding_forward(
                agent_embedding.reshape(-1, self.agent_embedding_dim),
                detach=False,
                ema=False,
            ).reshape(-1, self.N, self.role_embedding_dim)
            role_key = self.RECL.role_embedding_forward(
                agent_embedding.reshape(-1, self.agent_embedding_dim),
                detach=True,
                ema=True,
            ).reshape(-1, self.N, self.role_embedding_dim)

            rel = self.relation_encoder(agent_embedding, batch_s[:, t])
            relation_loss, rel_metrics = relation_driven_infonce(
                role_query=role_query,
                role_key=role_key,
                bilinear_w=self.RECL.W,
                relation_similarity=rel.context_similarity,
                pos_threshold=self.relation_pos_threshold,
                neg_threshold=self.relation_neg_threshold,
                temperature=self.relation_temperature,
                active_mask=active_t,
                fallback_neg_k=self.relation_fallback_neg_k,
                fallback_pos_k=self.relation_fallback_pos_k,
            )

            # Differentiable relation dynamics prediction. This is the component
            # that makes relation_encoder trainable even though the contrastive
            # masks are selected by non-differentiable threshold/top-k rules.
            pred_next_embedding = self.relation_dynamics_predictor(agent_embedding, rel.ego_context)
            dyn_error = F.mse_loss(pred_next_embedding, next_agent_embedding.detach(), reduction='none').mean(dim=-1)
            dynamics_loss = (dyn_error * active_t.view(-1, 1).float()).sum() / (active_t.sum() * self.N + 1e-8)

            # Optional fallback/ablation: retain original all-agent self-positive
            # bilinear contrastive signal with K-means disabled by default.
            kmeans_loss = torch.tensor(0.0, device=self.device)
            if self.kmeans_loss_weight > 0:
                kmeans_loss = self._legacy_kmeans_loss(role_query, role_key, agent_embedding, active_t)

            loss_t = (
                self.relation_loss_weight * relation_loss
                + self.relation_dynamics_loss_weight * dynamics_loss
                + self.kmeans_loss_weight * kmeans_loss
            )
            total_loss = total_loss + loss_t
            total_relation_loss = total_relation_loss + relation_loss.detach()
            total_dynamics_loss = total_dynamics_loss + dynamics_loss.detach()
            total_kmeans_loss = total_kmeans_loss + kmeans_loss.detach()
            valid_steps += 1

            rel_metrics["edge_entropy"] = edge_entropy(rel.edge_weights).detach()
            rel_metrics["edge_max"] = rel.edge_weights.max(dim=-1)[0].mean().detach()
            rel_metrics["relation_loss"] = relation_loss.detach()
            rel_metrics["relation_dynamics_loss"] = dynamics_loss.detach()
            rel_metrics["relation_total_step_loss"] = loss_t.detach()
            for k, v in rel_metrics.items():
                if k not in metric_acc:
                    metric_acc[k] = []
                metric_acc[k].append(v.detach().cpu() if torch.is_tensor(v) else v)

        if valid_steps == 0:
            return {
                "relation_total_loss": torch.tensor(0.0),
                "relation_loss": torch.tensor(0.0),
                "relation_dynamics_loss": torch.tensor(0.0),
            }

        total_loss = total_loss / valid_steps
        self.RECL_optimizer.zero_grad()
        self.relation_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.RECL.parameters(), 10)
        torch.nn.utils.clip_grad_norm_(self.relation_parameters, 10)
        self.RECL_optimizer.step()
        self.relation_optimizer.step()

        metrics = {
            "relation_total_loss": total_loss.detach().cpu(),
            "relation_loss": (total_relation_loss / valid_steps).detach().cpu(),
            "relation_dynamics_loss": (total_dynamics_loss / valid_steps).detach().cpu(),
            "legacy_kmeans_loss": (total_kmeans_loss / valid_steps).detach().cpu(),
        }
        for k, values in metric_acc.items():
            # Histograms are kept as last-step vectors; scalars are averaged.
            if k.endswith("_hist"):
                metrics[k] = values[-1]
            else:
                vals = [v.float() if torch.is_tensor(v) else torch.tensor(float(v)) for v in values]
                metrics[k] = torch.stack(vals).mean().detach().cpu()
        return metrics

    def _legacy_kmeans_loss(self, role_query, role_key, agent_embedding, active_t):
        """Lightweight legacy fallback: self-positive denominator contrast.

        The full original K-means branch remains in ACORM_Agent. This fallback is
        intentionally simple and only used when --kmeans_loss_weight > 0.
        """
        logits = torch.matmul(torch.matmul(role_query, self.RECL.W), role_key.transpose(1, 2))
        logits = logits - logits.max(dim=-1, keepdim=True)[0].detach()
        labels = torch.arange(self.N, device=logits.device).view(1, self.N, 1).expand(logits.shape[0], -1, -1)
        loss = F.cross_entropy(logits.reshape(-1, self.N), labels.reshape(-1), reduction="none")
        loss = loss.view(logits.shape[0], self.N)
        return (loss * active_t.view(-1, 1).float()).sum() / (active_t.sum() * self.N + 1e-8)

    def update_qmix(self, inputs, batch_o, batch_s, batch_r, batch_a, batch_last_a, batch_avail_a_n, batch_active, batch_dw, max_episode_len):
        """Copy of ACORM_Agent.update_qmix with loss returned for TensorBoard."""
        self.eval_Q_net.rnn_hidden = None
        self.target_Q_net.rnn_hidden = None
        _, role_embeddings = self.RECL.batch_role_embed_forward(batch_o, batch_last_a, max_episode_len, detach=False)
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
        role_embeddings = role_embeddings.reshape(-1, self.N, self.role_embedding_dim)
        att_eval = self.eval_mix_net.attention_net(state_gru_outs, role_embeddings, role_embeddings).reshape(-1, max_episode_len + 1, self.N * self.att_out_dim)
        with torch.no_grad():
            att_target = self.target_mix_net.attention_net(state_gru_outs, role_embeddings, role_embeddings).reshape(-1, max_episode_len + 1, self.N * self.att_out_dim)

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
        return loss.detach()
