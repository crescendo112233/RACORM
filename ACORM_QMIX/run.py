import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tensorboardX import SummaryWriter

from algorithm.acorm import ACORM_Agent
from algorithm.racorm import RelationACORM_Agent
from algorithm.vdn_qmix import VDN_QMIX
from env_factory import make_env
from util.replay_buffer import ReplayBuffer


ROLE_ALGORITHMS = ["ACORM", "RACORM"]


class Runner:
    def __init__(self, args):
        self.args = args
        self.env_name = self.args.env_name
        self.seed = self.args.seed

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.env = make_env(self.args)
        self.env_info = self.env.get_env_info()
        self.args.N = self.env_info["n_agents"]
        self.args.obs_dim = self.env_info["obs_shape"]
        self.args.state_dim = self.env_info["state_shape"]
        self.args.action_dim = self.env_info["n_actions"]
        self.args.episode_limit = self.env_info["episode_limit"]

        print(f"env_backend={getattr(self.args, 'env_backend', 'smac')}")
        print(f"env_name={self.env_name}")
        print("number of agents={}".format(self.args.N))
        print("obs_dim={}".format(self.args.obs_dim))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))
        if getattr(self.args, 'env_backend', 'smac') == 'smacv2' and hasattr(self.env, 'get_capabilities'):
            try:
                print(f"smacv2_capabilities={self.env.get_capabilities()}")
            except Exception:
                pass

        self.save_path = args.save_path
        self.model_path = args.model_path
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.args.tb_log_dir, exist_ok=True)

        time_path = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_env_name = f"{getattr(self.args, 'env_backend', 'smac')}_{self.env_name}"
        self.writer = SummaryWriter(
            log_dir='{}/{}/{}/{}_seed_{}_{}'.format(
                self.args.tb_log_dir,
                self.args.algorithm,
                log_env_name,
                log_env_name,
                self.seed,
                time_path,
            )
        )

        if args.algorithm in ['QMIX', 'VDN']:
            self.agent_n = VDN_QMIX(self.args)
        elif args.algorithm == 'ACORM':
            self.agent_n = ACORM_Agent(self.args)
        elif args.algorithm == 'RACORM':
            self.agent_n = RelationACORM_Agent(self.args)
        else:
            raise ValueError(f"Unsupported algorithm: {args.algorithm}")

        self.replay_buffer = ReplayBuffer(self.args, self.args.buffer_size)
        self.epsilon = self.args.epsilon
        self.win_rates = []
        self.evaluate_reward = []
        self.total_steps = 0
        self.agent_embed_pretrain_epoch, self.recl_pretrain_epoch = 0, 0
        self.pretrain_agent_embed_loss, self.pretrain_recl_loss = [], []
        self.args.agent_embed_pretrain_epochs = 120
        self.args.recl_pretrain_epochs = 100
        if getattr(self.agent_n, 'skip_recl_pretrain', False):
            self.args.recl_pretrain_epochs = 0

    def run(self):
        evaluate_num = -1
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()
                evaluate_num += 1

            _, _, episode_steps = self.run_episode_smac(evaluate=False)

            if self.args.algorithm in ROLE_ALGORITHMS and self.agent_embed_pretrain_epoch < self.args.agent_embed_pretrain_epochs:
                if self.replay_buffer.current_size >= self.args.batch_size:
                    self.agent_embed_pretrain_epoch += 1
                    agent_embedding_loss = self.agent_n.pretrain_agent_embedding(self.replay_buffer)
                    self.pretrain_agent_embed_loss.append(agent_embedding_loss.item())
                    if self.args.tb_plot:
                        self.writer.add_scalar(
                            'pretrain/agent_embedding_loss',
                            agent_embedding_loss.item(),
                            global_step=self.agent_embed_pretrain_epoch,
                        )
            else:
                if self.args.algorithm in ROLE_ALGORITHMS and self.recl_pretrain_epoch < self.args.recl_pretrain_epochs:
                    self.recl_pretrain_epoch += 1
                    recl_loss = self.agent_n.pretrain_recl(self.replay_buffer)
                    self.pretrain_recl_loss.append(recl_loss.item())
                    if self.args.tb_plot:
                        self.writer.add_scalar('pretrain/recl_loss', recl_loss.item(), global_step=self.recl_pretrain_epoch)
                    self._write_train_metrics(getattr(self.agent_n, 'last_train_metrics', {}))
                else:
                    self.total_steps += episode_steps
                    if self.replay_buffer.current_size >= self.args.batch_size:
                        metrics = self.agent_n.train(self.replay_buffer)
                        self._write_train_metrics(metrics)

        self.evaluate_policy()
        self.save_model()
        self.env.close()
        self.writer.close()

    def save_model(self):
        model_path = f'{self.model_path}/{getattr(self.args, "env_backend", "smac")}_{self.env_name}_seed{self.seed}_'
        torch.save(self.agent_n.eval_Q_net, model_path + 'q_net.pth')
        if hasattr(self.agent_n, 'RECL'):
            torch.save(self.agent_n.RECL.role_embedding_net, model_path + 'role_net.pth')
            torch.save(self.agent_n.RECL.agent_embedding_net, model_path + 'agent_embed_net.pth')
        if hasattr(self.agent_n, 'eval_mix_net'):
            if hasattr(self.agent_n.eval_mix_net, 'attention_net'):
                torch.save(self.agent_n.eval_mix_net.attention_net, model_path + 'attention_net.pth')
            torch.save(self.agent_n.eval_mix_net, model_path + 'mix_net.pth')
        if hasattr(self.agent_n, 'relation_encoder'):
            torch.save(self.agent_n.relation_encoder, model_path + 'relation_encoder.pth')
        if hasattr(self.agent_n, 'relation_dynamics_predictor'):
            torch.save(self.agent_n.relation_dynamics_predictor, model_path + 'relation_dynamics_predictor.pth')

    def _write_train_metrics(self, metrics):
        if not self.args.tb_plot or not metrics:
            return
        for key, value in metrics.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
                if value.numel() == 1:
                    self.writer.add_scalar(key, float(value.item()), global_step=self.total_steps)
                else:
                    self.writer.add_histogram(key, value, global_step=self.total_steps)
            elif isinstance(value, (int, float, np.number)):
                self.writer.add_scalar(key, float(value), global_step=self.total_steps)

    def evaluate_policy(self):
        win_times = 0
        evaluate_reward = 0
        for _ in range(int(self.args.evaluate_times)):
            win_tag, episode_reward, _ = self.run_episode_smac(evaluate=True)
            if win_tag:
                win_times += 1
            evaluate_reward += episode_reward
        win_rate = win_times / self.args.evaluate_times
        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.win_rates.append(win_rate)
        self.evaluate_reward.append(evaluate_reward)
        print("total_steps:{} \t win_rate:{} \t evaluate_reward:{}".format(self.total_steps, win_rate, evaluate_reward))

        if self.args.tb_plot:
            self.writer.add_scalar('eval/win_rate', win_rate, global_step=self.total_steps)
            self.writer.add_scalar('eval/mean_episode_reward', evaluate_reward, global_step=self.total_steps)
            self.writer.add_scalar('train/epsilon', self.epsilon, global_step=self.total_steps)

        if self.args.sns_plot:
            sns.set_style('whitegrid')
            plt.figure()
            x_step = np.array(range(len(self.win_rates)))
            sns.lineplot(x=x_step, y=np.array(self.win_rates).flatten(), label=self.args.algorithm)
            plt.ylabel('win_rates', fontsize=14)
            plt.xlabel(f'step*{self.args.evaluate_freq}', fontsize=14)
            plt.title(f'{self.args.algorithm} on {getattr(self.args, "env_backend", "smac")}/{self.env_name}')
            plt.savefig(f'{self.save_path}/{getattr(self.args, "env_backend", "smac")}_{self.env_name}_seed{self.seed}.jpg')
            plt.close()
            np.save(f'{self.save_path}/{getattr(self.args, "env_backend", "smac")}_{self.env_name}_seed{self.seed}.npy', np.array(self.win_rates))
            np.save(f'{self.save_path}/{getattr(self.args, "env_backend", "smac")}_{self.env_name}_seed{self.seed}_return.npy', np.array(self.evaluate_reward))

    def run_episode_smac(self, evaluate=False):
        win_tag = False
        episode_reward = 0
        self.env.reset()
        self.agent_n.eval_Q_net.rnn_hidden = None
        if self.args.algorithm in ROLE_ALGORITHMS:
            self.agent_n.RECL.agent_embedding_net.rnn_hidden = None
        last_onehot_a_n = np.zeros((self.args.N, self.args.action_dim))

        for episode_step in range(self.args.episode_limit):
            obs_n = self.env.get_obs()
            s = self.env.get_state()
            avail_a_n = self.env.get_avail_actions()
            epsilon = 0 if evaluate else self.epsilon

            if self.args.algorithm in ROLE_ALGORITHMS:
                if self.args.algorithm == 'RACORM':
                    role_embedding = self.agent_n.get_role_embedding(obs_n, last_onehot_a_n, s)
                else:
                    role_embedding = self.agent_n.get_role_embedding(obs_n, last_onehot_a_n)
                a_n = self.agent_n.choose_action(obs_n, last_onehot_a_n, role_embedding, avail_a_n, epsilon)
            else:
                a_n = self.agent_n.choose_action(obs_n, last_onehot_a_n, avail_a_n, epsilon)

            r, done, info = self.env.step(a_n)
            win_tag = True if done and 'battle_won' in info and info['battle_won'] else False
            episode_reward += r

            if not evaluate:
                if done and episode_step + 1 != self.args.episode_limit:
                    dw = True
                else:
                    dw = False
                self.replay_buffer.store_transition(episode_step, obs_n, s, avail_a_n, last_onehot_a_n, a_n, r, dw)
                self.epsilon = self.epsilon - self.args.epsilon_decay if self.epsilon - self.args.epsilon_decay > self.args.epsilon_min else self.args.epsilon_min

            last_onehot_a_n = np.eye(self.args.action_dim)[a_n]
            if done:
                break

        if not evaluate:
            obs_n = self.env.get_obs()
            s = self.env.get_state()
            avail_a_n = self.env.get_avail_actions()
            self.replay_buffer.store_last_step(episode_step + 1, obs_n, s, avail_a_n, last_onehot_a_n)

        return win_tag, episode_reward, episode_step + 1
