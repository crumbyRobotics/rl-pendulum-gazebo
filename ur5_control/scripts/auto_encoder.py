import torch
import torch.nn as nn
import torch.nn.functional as F


# 畳み込み後の画像サイズを計算
def conv2d_size_out(size, kernel_size, stride):
    return (size - (kernel_size - 1) - 1) // stride + 1


# 逆畳み込み後の画像サイズを計算
def convtran2d_size_out(size, kernel_size, stride, output_padding=0):
    return (size - 1) * stride + (kernel_size - 1) + output_padding + 1


# 重みを同期させる
def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


# obs -> featureにエンコード (SACでfeatureをstateとして用いる)
class Encoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, lr=0.001):
        super().__init__()

        assert len(obs_shape) == 3  # obs_shape : CHW

        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.kernel_size = 3
        self.stride = 2

        self.convs = nn.ModuleList([nn.Conv2d(obs_shape[0], self.num_filters, self.kernel_size, stride=self.stride)])
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(self.num_filters, self.num_filters, self.kernel_size, stride=self.stride))

        h = obs_shape[1]
        w = obs_shape[2]
        convh = conv2d_size_out(h, self.kernel_size, self.stride)
        convw = conv2d_size_out(w, self.kernel_size, self.stride)
        for i in range(num_layers - 1):
            convh = conv2d_size_out(convh, self.kernel_size, self.stride)
            convw = conv2d_size_out(convw, self.kernel_size, self.stride)
        self.linear_input_size = convw * convh * self.num_filters
        self.fc = nn.Linear(self.linear_input_size, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, obs, detach=False):
        conv = F.relu(self.convs[0](obs))
        for i in range(1, self.num_layers):
            conv = F.relu(self.convs[i](conv))

        h = conv.view(conv.size(0), -1)  # CHWを行ベクトルに変換
        if detach:
            h.detach()

        h = self.fc(h)
        feature = torch.tanh(self.ln(h))

        return feature

    def tie_conv_weights_from(self, source):
        tie_weights(src=source.fc, trg=self.fc)
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])


# feature -> obsにデコード
class Decoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, latent_lambda=0.0, lr=0.001):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.kernel_size = 3
        self.stride = 2

        h = obs_shape[1]
        w = obs_shape[2]
        for i in range(num_layers - 1):
            h = conv2d_size_out(h, self.kernel_size, self.stride)
            w = conv2d_size_out(w, self.kernel_size, self.stride)
        self.convh = conv2d_size_out(h, self.kernel_size, self.stride)
        self.convw = conv2d_size_out(w, self.kernel_size, self.stride)

        linear_input_size = self.convw * self.convh * self.num_filters
        self.fc = nn.Linear(self.feature_dim, linear_input_size)

        self.deconvs = nn.ModuleList()

        for i in range(self.num_layers - 1):
            self.deconvs.append(nn.ConvTranspose2d(self.num_filters, self.num_filters, self.kernel_size, stride=self.stride))
        self.deconvs.append(nn.ConvTranspose2d(num_filters, obs_shape[0], self.kernel_size, stride=self.stride, output_padding=1))

        self.latent_lambda = latent_lambda
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=self.latent_lambda)

    def forward(self, h):
        h = F.relu(self.fc(h))
        deconv = h.view(-1, self.num_filters, self.convh, self.convw)
        for i in range(0, self.num_layers - 1):
            deconv = F.relu(self.deconvs[i](deconv))
        obs = torch.sigmoid(self.deconvs[-1](deconv))  # 0-1に規格化. clipだと微分不可で進まない

        return obs
