import paddle.nn as nn
import numpy as np
import paddle
from paddle.fluid.dygraph import to_variable

from ppgan.models.discriminators.builder import DISCRIMINATORS
from ppgan.modules.adain import ResBlk


@DISCRIMINATORS.register()
class Discriminator(nn.Layer):
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512, **kwargs):
        super().__init__()
        dim_in = 2 ** 14 // img_size
        blocks = []
        blocks += [nn.Conv2D(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(negative_slope=0.2)]
        blocks += [nn.Conv2D(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(negative_slope=0.2)]
        blocks += [nn.Conv2D(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):
        out = self.main(x)
        # out = porch.Tensor(out)
        # out = out.view(out.size(0), -1)  # (batch, num_domains)
        out = paddle.reshape(x=out, shape=[out.shape[0], -1])
        # idx = to_variable(np.arange(y.shape[0]))
        # out = out[idx, y]  # (batch)
        # s = porch.take(out, list(zip(range(y.shape[0]), y.numpy().astype(int).tolist())))

        indices_tuple = list(zip(range(y.shape[0]), y.numpy().astype(int).tolist()))
        indices_list = list(map(list, indices_tuple))
        s = paddle.gather_nd(out, to_variable(np.asarray(indices_list)))
        return s
