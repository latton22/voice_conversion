import torch.nn as nn
import torch
import random
import os
import sys
from importlib import import_module
config = sys.argv[1]
config_dir = os.path.dirname(config)
config_bname = os.path.splitext(os.path.basename(config))[0]
sys.path.append(config_dir)
config = import_module(config_bname)

class SpecAugment(nn.Module):
    def __init__(self, freq_mask_num=0, time_mask_num=0, freq_width=27, maximum_time_mask_ratio=0.05, rng=None, spec_mask_value=-10.21, label_mask_value=28):
        super().__init__()
        self._rng = random.Random() if rng is None else rng
        self.freq_mask_num = freq_mask_num
        self.time_mask_num = time_mask_num
        self.freq_width = freq_width
        self.maximum_time_mask_ratio = maximum_time_mask_ratio
        self.spec_mask_value = spec_mask_value
        self.label_mask_value = label_mask_value

    @torch.no_grad()
    def forward(self, input_spec, output_label):
        batchsize, channel, length = input_spec.shape
        for idx in range(batchsize):
            # Frequency Masking
            for i in range(self.freq_mask_num):
                window = self._rng.randint(0, self.freq_width)
                x_left = self._rng.randint(0, channel - window)
                input_spec[idx, x_left : x_left + window, :] = self.spec_mask_value

            # Time Masking
            for i in range(self.time_mask_num):
                time_width = max(1, int(length * self.maximum_time_mask_ratio))
                window = self._rng.randint(0, time_width)
                y_left = self._rng.randint(0, max(1, length - window))
                input_spec[idx, :, y_left : y_left + window] = self.spec_mask_value
                # output_label[idx, y_left : y_left + window] = self.label_mask_value
        return input_spec, output_label

# CTCデコーダを使用するバージョンのContextNetを実装.
# 参考: https://github.com/ishine/ContextNet
class ContextNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_encoder = AudioEncoder()
        self.decoder = nn.Sequential(
            nn.Dropout(p=0.15),
            nn.Linear(256, config.ppg_dim),
            nn.LogSoftmax(dim=2)
        )

    def forward(self, x):
        h = self.audio_encoder(x)
        h = h.permute(0, 2, 1)
        output = self.decoder(h)
        return output

# 畳み込み層のstrideは系列長を保つため1で統一した.
class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        network = []

        # Input Layer
        network.append(ExpandChannel(config.mspec_dim, config.expand_input_dim))
        network.append(nn.Conv1d(config.mspec_dim + config.expand_input_dim, config.hidden_dim[0], kernel_size=config.conv_kernel, padding=(config.conv_kernel-1)//2))
        network.append(nn.BatchNorm1d(config.hidden_dim[0]))
        network.append(Swish())
        network.append(SELayer(config.hidden_dim[0]))

        # Hidden Layers
        for i in range(1, len(config.hidden_dim)):
            network.append(PiramidConvBlock(config.hidden_dim[i-1], config.hidden_dim[i]))
            # network.append(nn.Dropout(p=0.05)) # time-masking

        # Output Layer
        network.append(nn.Conv1d(config.hidden_dim[-1], 256, kernel_size=config.conv_kernel, padding=(config.conv_kernel-1)//2))
        network.append(nn.BatchNorm1d(256))
        network.append(Swish())
        network.append(SELayer(256))

        self.audio_encoder = nn.Sequential(*network)

    def forward(self, x):
        output = self.audio_encoder(x)
        return output

class ExpandChannel(nn.Module):
    def __init__(self, input_dim, expand_dim):
        super().__init__()

        self.shortcut = self._shortcut()
        self.pwconv = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Conv1d(input_dim, expand_dim, kernel_size=1)
        )

    def forward(self, x):
        h1 = self.shortcut(x)
        h2 = self.pwconv(x)
        output = torch.cat((h1, h2), 1)
        return output

    def _shortcut(self):
        return lambda x: x

# C0~C22で使用する畳み込みブロックの定義
class PiramidConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        network = []
        self.input_dim = input_dim
        # network.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(config.piramid_scale - 1):
            network.append(mix_conv(self.input_dim, self.input_dim + input_dim // config.piramid_scale))
            network.append(nn.BatchNorm1d(self.input_dim + input_dim // config.piramid_scale))
            network.append(Swish())
            self.input_dim += input_dim // config.piramid_scale
        network.append(mix_conv(self.input_dim, output_dim))
        network.append(nn.BatchNorm1d(output_dim))
        network.append(Swish())
        self.conv_net = nn.Sequential(*network)

        self.se_layer = SELayer(output_dim)
        self.shortcut = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size=1),
            nn.BatchNorm1d(output_dim)
        )

    def forward(self, x):
        h = self.conv_net(x)
        h = self.se_layer(h)
        output = h + self.shortcut(x)
        return output

    def _shortcut(self):
        return lambda x: x

# Squeeze-and-Excitation層(発話のグローバルな情報を反映させる)
class SELayer(nn.Module):
    def __init__(self, input_dim, reduction=8):
        super().__init__()
        self.shortcut = self._shortcut()
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, input_dim//reduction), # baseline : bias=False
            Swish(),
            nn.Linear(input_dim//reduction, input_dim), # baseline : bias=False
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, l = x.size()
        weight = self.gap(x).squeeze(-1)
        weight = self.bottleneck(weight).unsqueeze(-1)
        output = torch.mul(self.shortcut(x), weight.expand(b,c,l))
        return output

    def _shortcut(self):
        return lambda x: x

class mix_conv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.maxpool = nn.MaxPool1d(config.maxpool_kernel, stride=1, padding=(config.maxpool_kernel-1)//2)
        self.dwconv  = nn.Conv1d(input_dim, input_dim, kernel_size=config.conv_kernel, padding=(config.conv_kernel-1)//2, groups=input_dim)
        self.pwconv  = nn.Conv1d(input_dim * 2, output_dim, kernel_size=1)

    def forward(self, x):
        h1 = self.maxpool(x)
        h2 = self.dwconv(x)
        h = torch.cat((h1, h2), 1)
        output = self.pwconv(h)
        return output

# 活性化関数Swish. 勾配の計算式を明示すると計算コストが削減できるらしい.
class Swish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return swish_function.apply(x)

class swish_function(torch.autograd.Function):
    # https://medium.com/the-artificial-impostor/more-memory-efficient-swish-activation-function-e07c22c12a76
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
