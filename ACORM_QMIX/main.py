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
    parser = argparse.ArgumentParser(
        "Hyperparameter Setting for QMIX, VDN, ACORM and RACORM in SMAC/SMACv2"
    )

    # Training and evaluation.
    parser.add_argument("--max_train_steps", type=int, default=5000000)
    parser.add_argument("--evaluate_freq", type=int, default=10000)
    parser.add_argument("--evaluate_times", type=float, default=32)
    parser.add_argument("--algorithm", type=str, default="RACORM", choices=["QMIX", "VDN", "ACORM", "RACORM"])
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon_decay_steps", type=float, default=80000)
    parser.add_argument("--epsilon_min", type=float, default=0.02)
    parser.add_argument("--buffer_size", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--gamma", type=float, default=0.99)

    # Network sizes.
    parser.add_argument("--qmix_hidden_dim", type=int, default=32)
    parser.add_argument("--hyper_hidden_dim", type=int, default=64)
    parser.add_argument("--hyper_layers_num", type=int, default=2)
    parser.add_argument("--rnn_hidden_dim", type=int, default=64)
    parser.add_argument("--mlp_hidden_dim", type=int, default=64)
    parser.add_argument("--add_last_action", type=str2bool, default=True)

    # Optimisation.
    parser.add_argument("--use_hard_update", type=str2bool, default=False)
    parser.add_argument("--use_lr_decay", type=str2bool, default=True)
    parser.add_argument("--lr_decay_steps", type=int, default=500)
    parser.add_argument("--lr_decay_rate", type=float, default=0.98)
    parser.add_argument("--target_update_freq", type=int, default=100)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument('--device', type=str, default='cuda:0')

    # Environment.
    parser.add_argument('--env_backend', type=str, default='smac', choices=['smac', 'smacv2'],
                        help='Use original SMAC or SMACv2.')
    parser.add_argument('--env_name', type=str, default='MMM2',
                        help='SMAC map name, or SMACv2 scenario name such as protoss_5_vs_5.')

    # SMACv2 options. The default path matches the user environment.
    parser.add_argument('--smacv2_path', type=str, default='/home/zheping/RL/smacv2-main')
    parser.add_argument('--smacv2_race', type=str, default='auto', choices=['auto', 'terran', 'protoss', 'zerg'])
    parser.add_argument('--smacv2_n_units', type=int, default=5)
    parser.add_argument('--smacv2_n_enemies', type=int, default=5)
    parser.add_argument('--smacv2_map_name', type=str, default='',
                        help='Override generated map name, e.g., 10gen_protoss. Empty = infer from race.')
    parser.add_argument('--smacv2_map_x', type=int, default=32)
    parser.add_argument('--smacv2_map_y', type=int, default=32)
    parser.add_argument('--smacv2_start_position_dist', type=str, default='surrounded_and_reflect')
    parser.add_argument('--smacv2_surround_p', type=float, default=0.5)
    parser.add_argument('--smacv2_team_gen_observe', type=str2bool, default=True)
    parser.add_argument('--smacv2_conic_fov', type=str2bool, default=False)
    parser.add_argument('--smacv2_obs_own_pos', type=str2bool, default=True)
    parser.add_argument('--smacv2_use_unit_ranges', type=str2bool, default=True)
    parser.add_argument('--smacv2_min_attack_range', type=float, default=2.0)
    parser.add_argument('--smacv2_debug', type=str2bool, default=False)
    parser.add_argument('--smacv2_capability_config_path', type=str, default='',
                        help='Optional JSON/YAML capability config path. Overrides auto config.')
    parser.add_argument('--smacv2_capability_config_json', type=str, default='',
                        help='Optional JSON capability config string. Overrides auto config/path.')

    # Plot and logging.
    parser.add_argument("--sns_plot", type=str2bool, default=False)
    parser.add_argument("--tb_plot", type=str2bool, default=True)
    parser.add_argument("--tb_log_dir", type=str, default="/root/tf-logs")

    # ACORM role representation learning.
    parser.add_argument("--agent_embedding_dim", type=int, default=128)
    parser.add_argument("--role_embedding_dim", type=int, default=64)
    parser.add_argument("--use_ln", type=str2bool, default=False)
    parser.add_argument("--cluster_num", type=int, default=3)
    parser.add_argument("--recl_lr", type=float, default=8e-4)
    parser.add_argument("--agent_embedding_lr", type=float, default=1e-3)
    parser.add_argument("--train_recl_freq", type=int, default=200)
    parser.add_argument("--role_tau", type=float, default=0.005)
    parser.add_argument("--multi_steps", type=int, default=1)
    parser.add_argument("--role_mix_hidden_dim", type=int, default=64)

    # RACORM relation-guided role contrastive learning.
    parser.add_argument("--relation_hidden_dim", type=int, default=128)
    parser.add_argument("--relation_dim", type=int, default=64)
    parser.add_argument("--relation_lr", type=float, default=8e-4)
    parser.add_argument("--relation_temperature", type=float, default=0.2)
    parser.add_argument("--relation_loss_weight", type=float, default=1.0)
    parser.add_argument("--kmeans_loss_weight", type=float, default=0.0)

    # v3 rank-based sampling. Keep threshold mode for ablation only.
    parser.add_argument("--relation_sampling_mode", type=str, default="rank", choices=["rank", "threshold", "hybrid"])
    parser.add_argument("--relation_pos_k", type=int, default=1)
    parser.add_argument("--relation_neg_k", type=int, default=1)
    parser.add_argument("--relation_pos_threshold", type=float, default=0.65)
    parser.add_argument("--relation_neg_threshold", type=float, default=0.35)
    parser.add_argument("--relation_fallback_neg_k", type=int, default=2)
    parser.add_argument("--relation_fallback_pos_k", type=int, default=1)

    # Relation graph sparsification. relation_sparse_topk=0 keeps dense graph.
    parser.add_argument("--relation_topk", type=int, default=0,
                        help="Post-MLP top-k attention pruning; usually leave 0 when using relation_sparse_topk.")
    parser.add_argument("--relation_sparse_topk", type=int, default=0,
                        help="Pre-MLP top-k candidate neighbors. 0 keeps dense O(N^2) graph.")
    parser.add_argument("--relation_sparse_metric", type=str, default="cosine", choices=["cosine", "l2"])
    parser.add_argument("--relation_sparsify_before_mlp", type=str2bool, default=True)

    parser.add_argument("--relation_use_state", type=str2bool, default=True)
    parser.add_argument("--relation_mask_self", type=str2bool, default=True)
    parser.add_argument("--relation_dynamics_loss_weight", type=float, default=0.1)
    parser.add_argument("--relation_dynamics_hidden_dim", type=int, default=128)

    # ACORM attention mixer.
    parser.add_argument("--att_dim", type=int, default=128)
    parser.add_argument("--att_out_dim", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--soft_temperature", type=float, default=1.0)
    parser.add_argument("--state_embed_dim", type=int, default=64)

    # Save path.
    parser.add_argument('--save_path', type=str, default='./result/acorm')
    parser.add_argument('--model_path', type=str, default='./model/acorm')

    args = parser.parse_args()
    args.epsilon_decay = (args.epsilon - args.epsilon_min) / args.epsilon_decay_steps

    torch.multiprocessing.set_start_method('spawn')
    runner = Runner(args)
    runner.run()
