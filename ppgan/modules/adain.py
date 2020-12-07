import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.fluid.layers import pool2d


class ResBlk(nn.Layer):
    def __init__(self, dim_in, dim_out, actv=None,
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2D(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2D(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2D(dim_in)
            self.norm2 = nn.InstanceNorm2D(dim_in)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2D(dim_in, dim_out, 1, 1, 0, bias_attr=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = pool2d(input=x, pool_size=2, pool_type='avg', pool_stride=2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        # x = self.actv(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.conv1(x)
        if self.downsample:
            x = pool2d(input=x, pool_size=2, pool_type='avg', pool_stride=2)
        if self.normalize:
            x = self.norm2(x)
        # x = self.actv(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class AdaIN(nn.Layer):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2D(num_features)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        # h = porch.varbase_to_tensor(h)
        # h = h.view(h.size(0), h.size(1), 1, 1)
        h = paddle.reshape(x=h, shape=[h.shape[0], h.shape[1], 1, 1])
        gamma, beta = paddle.chunk(h, chunks=2, axis=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Layer):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=None, upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2D(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2D(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2D(dim_in, dim_out, 1, 1, 0, bias_attr=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='NEAREST')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        # x = self.actv(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='NEAREST')
        x = self.conv1(x)
        x = self.norm2(x, s)
        # x = self.actv(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out

