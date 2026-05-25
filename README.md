# RACORM: Relation-aware ACORM / Idea 1

This folder contains complete files to add or replace in the official NJU-RL/ACORM repository. It is not a patch script. Copy these files into the same relative paths of your ACORM checkout.

## Files

Modified files:

- `ACORM_QMIX/main.py`
- `ACORM_QMIX/run.py`

New files:

- `ACORM_QMIX/algorithm/relation_context.py`
- `ACORM_QMIX/algorithm/racorm.py`
- `run_racorm_mmm2.sh`

## Usage

```bash
git clone https://github.com/NJU-RL/ACORM.git
cd ACORM
cp -r /path/to/ACORM_RACORM_files/* .

python ./ACORM_QMIX/main.py \
  --algorithm RACORM \
  --env_name MMM2 \
  --cluster_num 3 \
  --max_train_steps 3050000 \
  --tb_log_dir /root/tf-logs
```

Start TensorBoard:

```bash
tensorboard --logdir /root/tf-logs --bind_all
```

## Core idea

Original ACORM uses K-means over agent embeddings to construct role-positive and role-negative pairs. RACORM keeps ACORM's role encoder and attention-guided mixing network, but replaces the main contrastive role supervision with relation-context-driven sampling:

\[
r_{ij} = g(e_i, e_j, |e_i-e_j|, e_i\odot e_j, s)
\]

\[
R_i = \sum_{j\ne i}\alpha_{ij}r_{ij}
\]

\[
\operatorname{sim}_R(i,k)=\cos(R_i,R_k)
\]

Pairs with high relation-context similarity are positives; pairs with low similarity are negatives. This makes the role contrastive signal more relation-aware rather than purely cluster-driven.

## TensorBoard tags

Events are written under `/root/tf-logs` by default. Important tags:

- `eval/win_rate`
- `eval/mean_episode_reward`
- `train/epsilon`
- `train/qmix_loss`
- `train/relation_loss`
- `train/relation_total_loss`
- `train/legacy_kmeans_loss`
- `train/relation_pos_frac`
- `train/relation_neg_frac`
- `train/relation_sim_mean`
- `train/relation_sim_std`
- `train/edge_entropy`
- `train/edge_max`
- `train/relation_sim_hist`
- `pretrain/agent_embedding_loss`
- `pretrain/recl_loss`

## Recommended ablations

Original ACORM:

```bash
python ./ACORM_QMIX/main.py --algorithm ACORM --env_name MMM2 --cluster_num 3 --tb_log_dir /root/tf-logs
```

RACORM:

```bash
python ./ACORM_QMIX/main.py --algorithm RACORM --env_name MMM2 --cluster_num 3 --tb_log_dir /root/tf-logs
```

RACORM without global state in relation encoder:

```bash
python ./ACORM_QMIX/main.py --algorithm RACORM --env_name MMM2 --relation_use_state false --tb_log_dir /root/tf-logs
```

Sparse relation graph:

```bash
python ./ACORM_QMIX/main.py --algorithm RACORM --env_name MMM2 --relation_topk 2 --tb_log_dir /root/tf-logs
```

RACORM with weak legacy fallback:

```bash
python ./ACORM_QMIX/main.py --algorithm RACORM --env_name MMM2 --kmeans_loss_weight 0.1 --tb_log_dir /root/tf-logs
```
