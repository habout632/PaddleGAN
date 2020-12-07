import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from paddle import to_tensor

from ppgan.models.generators.builder import GENERATORS
from ppgan.modules.adain import ResBlk, AdainResBlk


class HighPass(nn.Layer):
    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.filter = to_tensor(np.asarray([[-1, -1, -1],
                                              [-1, 8., -1],
                                              [-1, -1, -1]]) / w_hpf)

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.shape[1], 1, 1, 1)
        return nn.Conv2D(x, filter, padding=1, groups=x.shape[1])


@GENERATORS.register()
class Generator(nn.Layer):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=1):
        super().__init__()
        dim_in = 2 ** 14 // img_size
        self.img_size = img_size
        self.from_rgb = nn.Conv2D(3, dim_in, 3, 1, 1)
        self.encode = nn.LayerList()
        self.decode = nn.LayerList()
        self.to_rgb = nn.Sequential(
            # nn.InstanceNorm2d(dim_in, affine=True),
            nn.InstanceNorm2D(dim_in),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2D(dim_in, 3, 1, 1, 0))

        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            if len(self.decode) > 0:
                self.decode.insert(
                    index=0, sublayer=AdainResBlk(dim_out, dim_in, style_dim,
                                                  w_hpf=w_hpf, upsample=True))  # stack-like
            else:
                self.decode.append(AdainResBlk(dim_out, dim_in, style_dim,
                                               w_hpf=w_hpf, upsample=True))
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))

        if w_hpf > 0:
            # device = porch.device(
            #     'cuda' if porch.cuda.is_available() else 'cpu')
            self.hpf = HighPass(w_hpf, "device")

    def forward(self, x, s, masks=None):

        x = self.from_rgb(x)

        cache = {}
        for block in self.encode:
            if (masks is not None) and (x.shape[2] in [32, 64, 128]):
                cache[x.shape[2]] = x
            x = block(x)
        for block in self.decode:
            x = block(x, s)
            if (masks is not None) and (x.shape[2] in [32, 64, 128]):
                mask = masks[0] if x.shape[2] in [32] else masks[1]
                mask = F.interpolate(mask, size=x.shape[2], mode='NEAREST')
                x = x + self.hpf(mask * cache[x.shape[2]])
        y = self.to_rgb(x)
        return y


@GENERATORS.register()
class MappingNetwork(nn.Layer):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
        super().__init__()
        layers = []
        layers += [nn.Linear(latent_dim, 512)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.LayerList()
        for _ in range(num_domains):
            self.unshared.append(nn.Sequential(nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, style_dim)
                                            ))

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = paddle.stack(out, axis=1)  # (batch, num_domains, style_dim)
        # s = porch.take(out, list(zip(range(y.shape[0]), y.numpy().astype(int).tolist())))

        indices_tuple = list(zip(range(y.shape[0]), y.numpy().astype(int).tolist()))
        indices_list = list(map(list, indices_tuple))
        s = paddle.gather_nd(out, to_tensor(np.asarray(indices_list)))
        return s

    def finetune(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = paddle.stack(out, axis=1)  # (batch, num_domains, style_dim)

        # s = porch.take(out, list(zip(range(y.size(0)), y.numpy().astype(int).tolist())))
        indices_tuple = list(zip(range(y.shape[0]), y.numpy().astype(int).tolist()))
        indices_list = list(map(list, indices_tuple))
        s = paddle.gather_nd(out, to_tensor(np.asarray(indices_list)))
        return s, h, out


@GENERATORS.register()
class StyleEncoder(nn.Layer):
    def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512):
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
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.LayerList()
        for _ in range(num_domains):
            self.unshared.append(nn.Linear(dim_out, style_dim))

    def forward(self, x, y):
        """

        :param x: reference image
        :param y:
        :return:
        """

        h = self.shared(x)
        # h = to_variable(h)
        # h = h.reshape(h.shape[0], -1)
        h = paddle.reshape(x=h, shape=[h.shape[0], -1])
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = paddle.stack(out, axis=1)  # (batch_size, num_domains, style_dim)
        indices_tuple = list(zip(range(y.shape[0]), y.numpy().astype(int).tolist()))
        indices_list = list(map(list, indices_tuple))
        s = paddle.gather_nd(out, to_tensor(np.asarray(indices_list)))
        # s = porch.take(out, list(zip(range(y.shape[0]), y.numpy().astype(int).tolist())))
        # idx = to_variable(np.asarray(range(y.shape[0])))
        # s = out[range(y.shape[0]), y]
        # s = index_select(out, y, dim=1)
        # s = out[idx, y]  # (batch_size, style_dim)
        return s
