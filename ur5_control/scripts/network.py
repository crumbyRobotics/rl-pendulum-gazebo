import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from auto_encoder import Encoder


# ネットワークの重み初期化
def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


# 方策ネットワーク（画像を入力としてアクションの平均と分散を出力）
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, obs_shape, num_layers, num_filters, lr=0.003):  # obs_shape: (B, B, C, H, W)
        super(PolicyNetwork, self).__init__()

        self.encoder = Encoder(obs_shape[slice(2, 5)], state_dim, num_layers, num_filters)

        self.linear1 = nn.Linear(state_dim * obs_shape[1], 64)  # 3つの連続した画像を入力（速度・加速度）
        self.linear2 = nn.Linear(64, 64)
        self.pi_mean = nn.Linear(64, action_dim)
        self.pi_stddev = nn.Linear(64, action_dim)

        self.apply(weight_init)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, obs_batch, detach=False):
        s = []
        for obs in obs_batch:
            s.append(self.encoder(obs, detach).reshape(1, -1))  # (1, state_dim)
        x = torch.cat(s)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mean = self.pi_mean(x)
        log_stddev = self.pi_stddev(x)

        stddev = torch.exp(log_stddev)

        return mean, stddev


# 状態行動価値関数ネットワーク（画像とアクションを入力としてQ値を出力）
class DualQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, obs_shape, num_layers, num_filters, lr=0.003):  # obs_shape: (B, B, C, H, W)
        super(DualQNetwork, self).__init__()

        self.encoder = Encoder(obs_shape[slice(2, 5)], state_dim, num_layers, num_filters)

        # QNetwork 3
        self.linear1 = nn.Linear(state_dim * obs_shape[1] + action_dim, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 64)
        self.q1 = nn.Linear(64, 1)

        # QNetwork 2
        self.linear4 = nn.Linear(state_dim * obs_shape[1] + action_dim, 64)
        self.linear5 = nn.Linear(64, 64)
        self.linear6 = nn.Linear(64, 64)
        self.q2 = nn.Linear(64, 1)

        self.apply(weight_init)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, obs_batch, action, detach=False):
        s = []
        for obs in obs_batch:
            s.append(self.encoder(obs, detach).reshape(1, -1))  # (1, state_dim)
        state = torch.cat(s)
        x = torch.cat([state, action], 1)  # combination s and a

        # # QNetwork 1
        x1 = F.relu(self.linear1(x))
        x1 = F.relu(self.linear2(x1))
        x1 = F.relu(self.linear3(x1))
        q_value1 = self.q1(x1)

        # QNetwork 2
        x2 = F.relu(self.linear4(x))
        x2 = F.relu(self.linear5(x2))
        x2 = F.relu(self.linear6(x2))
        q_value2 = self.q2(x2)

        return q_value1, q_value2
