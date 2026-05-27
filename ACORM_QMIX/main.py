import argparse

import torch

from run import Runner


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for QMIX, VDN, ACORM and RACORM in SMAC environment")

    parser.add_argument("--max_train_steps", type=int, default=5000000, help="Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=int, default=10000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=32, help="Evaluate times")
    # parser.add_argument("--save_freq", type=int, default=int(1e5), help="Save frequency")

    parser.add_argument("--algorithm", type=str, default="RACORM", choices=["QMIX", "VDN", "ACORM", "RACORM"],
                        help="QMIX, VDN, ACORM, or RACORM")

    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon_decay_steps", type=float, default=80000,
                        help="How many steps before the epsilon decays to the minimum")
    parser.add_argument("--epsilon_min", type=float, default=0.02, help="Minimum epsilon")
    parser.add_argument("--buffer_size", type=int, default=5000, help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")

    parser.add_argument("--qmix_hidden_dim", type=int, default=32, help="The dimension of the hidden layer of the QMIX network")
    parser.add_argument("--hyper_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of the hyper-network")
    parser.add_argument("--hyper_layers_num", type=int, default=2, help="The number of layers of hyper-network")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of RNN")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of MLP")
    parser.add_argument("--add_last_action", type=str2bool, default=True, help="Whether to add last actions into the observation")

    parser.add_argument("--use_hard_update", type=str2bool, default=False, help="Whether to use hard update")
    parser.add_argument("--use_lr_decay", type=str2bool, default=True, help="Whether to use learning rate decay")
    parser.add_argument("--lr_decay_steps", type=int, default=500, help="every steps decay steps")
    parser.add_argument("--lr_decay_rate", type=float, default=0.98, help="learn decay rate")
    parser.add_argument("--target_update_freq", type=int, default=100, help="Update frequency of the target network")
    parser.add_argument("--tau", type=float, default=0.005, help="If use soft update")

    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--env_name', type=str, default='MMM2')  # ['3m', '8m', '2s3z']

    # Plot and logging.
    parser.add_argument("--sns_plot", type=str2bool, default=False, help="Whether to use seaborn plot")
    parser.add_argument("--tb_plot", type=str2bool, default=True, help="Whether to use tensorboard plot")
    parser.add_argument("--tb_log_dir", type=str, default="/root/tf-logs", help="TensorBoard event root directory")

    # ACORM role representation learning.
    parser.add_argument("--agent_embedding_dim", type=int, default=128, help="The dimension of the agent embedding")
    parser.add_argument("--role_embedding_dim", type=int, default=64, help="The dimension of the role embedding")
    parser.add_argument("--use_ln", type=str2bool, default=False, help="Whether to use layer normalization")
    parser.add_argument("--cluster_num", type=int, default=int(3), help="the cluster number of k-means in original ACORM")
    parser.add_argument("--recl_lr", type=float, default=8e-4, help="Learning rate")
    parser.add_argument("--agent_embedding_lr", type=float, default=1e-3, help="agent_embedding Learning rate")
    parser.add_argument("--train_recl_freq", type=int, default=200, help="Train frequency of the RECL network")
    parser.add_argument("--role_tau", type=float, default=0.005, help="If use soft update")
    parser.add_argument("--multi_steps", type=int, default=1, help="Train frequency of the RECL network")
    parser.add_argument("--role_mix_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of the QMIX network")

    # RACORM / Idea 1: relation-driven role contrastive learning.
    parser.add_argument("--relation_hidden_dim", type=int, default=128, help="Hidden dimension of the pairwise relation encoder")
    parser.add_argument("--relation_dim", type=int, default=64, help="Output dimension of each pairwise relation and ego relation context")
    parser.add_argument("--relation_lr", type=float, default=8e-4, help="Learning rate of the relation encoder")
    parser.add_argument("--relation_temperature", type=float, default=0.2, help="Temperature for relation-driven InfoNCE")
    parser.add_argument("--relation_sampling_mode", type=str, default="rank", choices=["rank", "threshold", "hybrid"],
                        help="How relation positives/negatives are mined: rank is the v3 default; threshold is v2 ablation")
    parser.add_argument("--relation_pos_k", type=int, default=1,
                        help="Top-k most relation-similar non-self agents used as positives in rank/hybrid mode")
    parser.add_argument("--relation_neg_k", type=int, default=1,
                        help="Bottom-k least relation-similar non-self agents used as negatives in rank/hybrid mode")
    parser.add_argument("--relation_pos_threshold", type=float, default=0.65,
                        help="Threshold-mode positive similarity threshold; used only when relation_sampling_mode is threshold/hybrid")
    parser.add_argument("--relation_neg_threshold", type=float, default=0.35,
                        help="Threshold-mode negative similarity threshold; used only when relation_sampling_mode is threshold")
    parser.add_argument("--relation_loss_weight", type=float, default=1.0, help="Weight of relation-driven role contrastive loss")
    parser.add_argument("--kmeans_loss_weight", type=float, default=0.0,
                        help="Optional legacy fallback loss weight; 0 disables the fallback by default")
    parser.add_argument("--relation_topk", type=int, default=0,
                        help="Post-MLP top-k neighbors for sparse relation attention; 0 keeps all candidate neighbors")
    parser.add_argument("--relation_sparse_topk", type=int, default=0,
                        help="Pre-MLP sparse candidate neighbors for efficient relation graph construction; 0 uses dense graph")
    parser.add_argument("--relation_sparse_metric", type=str, default="cosine", choices=["cosine", "l2"],
                        help="Cheap metric used for pre-MLP relation_sparse_topk candidate selection")
    parser.add_argument("--relation_sparsify_before_mlp", type=str2bool, default=True,
                        help="Whether relation_sparse_topk is applied before the pairwise MLP to reduce computation")
    parser.add_argument("--relation_use_state", type=str2bool, default=True,
                        help="Whether to use the global state in the pairwise relation encoder")
    parser.add_argument("--relation_mask_self", type=str2bool, default=True,
                        help="Whether to mask self edges in relation attention")
    parser.add_argument("--relation_fallback_neg_k", type=int, default=2,
                        help="Threshold-mode fallback bottom-k negatives when thresholding yields no negatives; not used in rank mode")
    parser.add_argument("--relation_fallback_pos_k", type=int, default=1,
                        help="Threshold-mode fallback top-k positives when thresholding yields no inter-agent positives; not used in rank mode")
    parser.add_argument("--relation_dynamics_loss_weight", type=float, default=0.1,
                        help="Weight of differentiable relation dynamics prediction loss")
    parser.add_argument("--relation_dynamics_hidden_dim", type=int, default=128,
                        help="Hidden dimension of relation dynamics predictor")

    # ACORM attention mixer.
    parser.add_argument("--att_dim", type=int, default=128, help="The dimension of the attention net")
    parser.add_argument("--att_out_dim", type=int, default=64, help="The dimension of the attention net")
    parser.add_argument("--n_heads", type=int, default=4, help="multi-head attention")
    parser.add_argument("--soft_temperature", type=float, default=1.0, help="multi-head attention")
    parser.add_argument("--state_embed_dim", type=int, default=64, help="The dimension of the gru state net")

    # Save path.
    parser.add_argument('--save_path', type=str, default='./result/acorm')
    parser.add_argument('--model_path', type=str, default='./model/acorm')

    args = parser.parse_args()
    args.epsilon_decay = (args.epsilon - args.epsilon_min) / args.epsilon_decay_steps

    torch.multiprocessing.set_start_method('spawn')

    runner = Runner(args)
    runner.run()
