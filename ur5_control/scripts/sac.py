import random
from collections import namedtuple

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from network import PolicyNetwork
from network import DualQNetwork
from network import weight_init

from auto_encoder import Decoder


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


# 各時刻の状態と行動と報酬を保存するリプレイメモリー
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity  # サイクルバッファ

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class SAC():
    def __init__(self, device, state_dim, action_space, obs_shape, num_layers, num_filters, decoder_latent_lambda,
                 buffer_size, gamma, soft_target_tau, encoder_soft_target_tau, target_entropy, policy_lr, q_lr, alpha_lr):
        super(SAC, self).__init__()

        self.device = device

        self.state_dim = state_dim
        self.action_dim = action_space.shape[0]

        # Envアクション用にスケールする
        self.action_center = torch.FloatTensor((action_space.high + action_space.low) / 2).to(self.device)
        self.action_scale = torch.FloatTensor(action_space.high - self.action_center.cpu().detach().numpy()).to(self.device)

        # Neural Networks
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim, obs_shape, num_layers, num_filters, policy_lr).to(self.device)

        self.q_net = DualQNetwork(self.state_dim, self.action_dim, obs_shape, num_layers, num_filters, q_lr).to(self.device)
        self.target_q_net = DualQNetwork(self.state_dim, self.action_dim, obs_shape, num_layers, num_filters, q_lr).to(self.device)

        # for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
        #     target_param.data.copy_(param.data)
        self.target_q_net.load_state_dict(self.q_net.state_dict())  # target_q_netとq_netを同期

        self.policy_net.encoder.tie_conv_weights_from(self.q_net.encoder)

        self.decoder = Decoder(obs_shape[slice(2, 5)], state_dim, num_layers, num_filters, decoder_latent_lambda).to(self.device)
        self.decoder.apply(weight_init)

        # Experiences
        self.replay_memory = ReplayMemory(buffer_size)

        self.target_entropy = -self.action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        # Hyper Parameters
        self.gamma = gamma
        self.soft_target_tau = soft_target_tau
        self.encoder_soft_target_tau = encoder_soft_target_tau
        self.target_entropy = target_entropy

        self.count = 0

    def sample_action(self, state):
        mean, stddev = self.policy_net(state)

        # Reparameterization
        normal_random = torch.normal(0, 1, size=mean.shape).to(self.device)
        action_org = mean + stddev * normal_random

        # Squashed Gaussian Policy
        action = torch.tanh(action_org)

        return action.to(self.device), mean.to(self.device), stddev.to(self.device), action_org.to(self.device)

    def sample_action_for_env(self, state):
        action, _, _, _ = self.sample_action(state)
        env_action = action * self.action_scale + self.action_center

        return env_action.cpu().detach().numpy()[0], action.cpu().detach().numpy()[0]

    # 正規分布でのactionの対数確率密度関数logμ(a|s)
    def compute_logpi(self, mean, stddev, action):
        a1 = -0.5 * np.log(2*np.pi)
        a2 = -torch.log(stddev)
        a3 = -0.5 * (((action - mean) / stddev) ** 2)
        return (a1 + a2 + a3).to(self.device)

    # tanhで変換されたactionのlogπ(a|s)をaction_orgを使って計算
    def compute_logpi_sgp(self, mean, stddev, action_org):
        logmu = self.compute_logpi(mean, stddev, action_org)
        tmp = 1 - torch.tanh(action_org) ** 2
        tmp = torch.clip(tmp, 1e-10, 1.0)  # log(0)回避
        logpi = logmu - torch.sum(torch.log(tmp), 1, keepdim=True)
        return logpi.to(self.device)

    def update(self, batch_size, q_net_sync=False):
        # 経験をバッチでサンプリング
        transitions = self.replay_memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        obs_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        n_obs_batch = torch.cat(batch.next_state).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        done_batch = torch.cat(batch.done).to(self.device)

        alpha = torch.exp(self.log_alpha).to(self.device)

        # Q(s,a)の推定値を計算し, Q値の損失関数を計算
        with torch.no_grad():
            n_action, n_mean, n_stddev, n_action_org = self.sample_action(n_obs_batch)

            n_logpi = self.compute_logpi_sgp(n_mean, n_stddev, n_action_org)
            n_q1, n_q2 = self.target_q_net(n_obs_batch, n_action)

            q_est = reward_batch + (1 - done_batch) * self.gamma * torch.minimum(n_q1, n_q2) - (alpha * n_logpi)
        q1, q2 = self.q_net(obs_batch, action_batch.detach())
        q1_loss = F.mse_loss(q1, q_est.float())
        q2_loss = F.mse_loss(q2, q_est.float())
        q_loss = q1_loss + q2_loss

        # q_lossからQNetworkを学習
        self.q_net.optimizer.zero_grad()
        q_loss.backward()
        self.q_net.optimizer.step()

        # 方策の損失関数を計算
        action, mean, stddev, action_org = self.sample_action(obs_batch)  # 現在の方策π(θ)で選ばれるactionについて評価
        logpi = self.compute_logpi_sgp(mean, stddev, action_org)
        q1, q2 = self.q_net(obs_batch, action)
        q_min = torch.minimum(q1, q2)
        policy_loss = -(q_min - alpha.detach() * logpi).mean()

        # policy_lossからPolicyNetworkを学習
        self.policy_net.optimizer.zero_grad()
        policy_loss.backward()
        self.policy_net.optimizer.step()

        # αの自動調整
        alpha_loss = (-alpha * (logpi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Encoder/Decoderの学習
        target_obs = obs_batch[:, -1]
        h = self.q_net.encoder(target_obs)

        rec_obs = self.decoder(h)

        rec_loss = F.mse_loss(rec_obs, target_obs)

        # L2拘束
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()

        encoder_loss = rec_loss + self.decoder.latent_lambda * latent_loss

        self.q_net.encoder.optimizer.zero_grad()
        self.decoder.optimizer.zero_grad()
        encoder_loss.backward()
        self.q_net.encoder.optimizer.step()
        self.decoder.optimizer.step()

        self.count += 1

        # ソフトターゲットで更新 (Q-NetとEncoder)
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_target_tau) + param.data * self.soft_target_tau)

        for target_enc_param, enc_param in zip(self.target_q_net.encoder.parameters(), self.q_net.encoder.parameters()):
            target_enc_param.data.copy_(target_enc_param.data * (1.0 - self.encoder_soft_target_tau) + enc_param.data * self.encoder_soft_target_tau)

        # q_net_syncフラグが有効ならq_netを同期させる
        if q_net_sync:
            for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(param.data)

            for target_enc_param, enc_param in zip(self.target_q_net.encoder.parameters(), self.q_net.encoder.parameters()):
                target_enc_param.data.copy_(enc_param.data)

        return policy_loss.cpu().detach().numpy(), q_loss.cpu().detach().numpy(), rec_loss.cpu().detach().numpy()
