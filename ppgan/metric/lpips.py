"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""
import argparse

import paddle.nn as nn
import paddle

from paddle import fluid, to_tensor
from paddle.fluid.layers import mean, stack

from ppgan.metric.alexnet import alexnet


def normalize(x, eps=1e-10):
    return x * paddle.rsqrt(paddle.sum(x ** 2, axis=1, keepdim=True) + eps)


class AlexNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self.layers = alexnet(None).features
        self.channels = []
        for name, layer in self.layers._sub_layers.items():
            if isinstance(layer, paddle.nn.Conv2D):
                self.channels.append(layer._out_channels)

    def forward(self, x):
        fmaps = []
        for name, layer in self.layers._sub_layers.items():
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                fmaps.append(x)
        return fmaps


class Conv1x1(nn.Layer):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2D(in_channels, out_channels, 1, 1, 0, bias_attr=False))

    def forward(self, x):
        return self.main(x)


class LPIPS(nn.Layer):
    def __init__(self, pretrained_weights_fn=None):
        super().__init__()
        self.alexnet = AlexNet()
        self.lpips_weights = nn.LayerList()
        for channels in self.alexnet.channels:
            self.lpips_weights.append(Conv1x1(channels, 1))
        if pretrained_weights_fn is not None:
            self._load_lpips_weights(pretrained_weights_fn)
        # imagenet normalization for range [-1, 1]
        self.mu = paddle.reshape(to_tensor([-0.03, -0.088, -0.188]), shape=(1, 3, 1, 1))
        self.sigma = paddle.reshape(to_tensor([0.458, 0.448, 0.450]), shape=(1, 3, 1, 1))

    def _load_lpips_weights(self, pretrained_weights_fn):
        own_state_dict = self.state_dict()
        state_dict = paddle.load(pretrained_weights_fn)
        self.load_state_dict = state_dict
        # for name, param in state_dict.items():
        #     if name in own_state_dict:
        #         own_state_dict[name]=torch.c

    def forward(self, x: paddle.Tensor, y: paddle.Tensor):
        x = (x - self.mu) / self.sigma
        y = (y - self.mu) / self.sigma
        x_fmaps = self.alexnet(x)
        y_fmaps = self.alexnet(y)

        lpips_value = 0
        for x_fmap, y_fmap, conv1x1 in zip(x_fmaps, y_fmaps, self.lpips_weights):
            x_fmap = normalize(x_fmap)
            y_fmap = normalize(y_fmap)
            z = paddle.pow(x_fmap - y_fmap, 2)
            lpips_value += paddle.mean(conv1x1(z))
            # print("paddle alexnet mean", torch.mean(z).numpy(),lpips_value.numpy())
        return lpips_value


@fluid.dygraph.no_grad
def calculate_lpips_given_images(group_of_images: list):
    """

    :param group_of_images: list of Tensor
    :return:
    """

    # device = porch.device('cuda' if porch.cuda.is_available() else 'cpu')
    lpips = LPIPS(pretrained_weights_fn="/data2/LPIPS_pretrained.pdparams")
    lpips.eval()
    lpips_values = []
    num_rand_outputs = len(group_of_images)

    # calculate the average of pairwise distances among all random outputs
    for i in range(num_rand_outputs - 1):
        for j in range(i + 1, num_rand_outputs):
            lpips_values.append(lpips(group_of_images[i], group_of_images[j]))
    lpips_value = mean(stack(lpips_values, axis=0))
    return lpips_value.numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', type=str, nargs=2, help='paths to real and fake images')
    parser.add_argument('--img_size', type=int, default=256, help='image resolution')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size to use')
    args = parser.parse_args()
    group_of_images = [paddle.randn(shape=(8, 3, 256, 256)) for _ in range(10)]
    lpips_value = calculate_lpips_given_images(group_of_images)

    print('LPIPS: ', lpips_value)
