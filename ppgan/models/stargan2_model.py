#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from paddle.fluid.layers import sigmoid_cross_entropy_with_logits
from paddle.regularizer import L2Decay

from ppgan.solver.lr_scheduler import build_lr_scheduler
from .base_model import BaseModel

from .builder import MODELS
from .generators.builder import build_generator
from .discriminators.builder import build_discriminator
from .losses import GANLoss

from ..solver import build_optimizer
from ..modules.init import init_weights
from ..utils.image_pool import ImagePool


@MODELS.register()
class StarGAN2Model(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    StarGAN2 paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    def __init__(self, cfg):
        """Initialize the CycleGAN class.

        Parameters:
            opt (config)-- stores all the experiment flags; needs to be a subclass of Dict
        """
        super(StarGAN2Model, self).__init__(cfg)

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.nets['generator'] = build_generator(cfg.model.generator)
        self.nets['mapping_network'] = build_generator(cfg.model.mapping_network)
        self.nets['style_encoder'] = build_generator(cfg.model.style_encoder)

        self.nets['generator_ema'] = build_generator(cfg.model.generator)
        self.nets['mapping_network_ema'] = build_generator(cfg.model.mapping_network)
        self.nets['style_encoder_ema'] = build_generator(cfg.model.style_encoder)

        self.nets['discriminator'] = build_discriminator(cfg.model.discriminator)

        # define discriminators
        if self.is_train:
            # if cfg.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
            #     assert (
            #         cfg.dataset.train.input_nc == cfg.dataset.train.output_nc)
            #
            # # create image buffer to store previously generated images
            # self.fake_A_pool = ImagePool(cfg.dataset.train.pool_size)
            #
            # # create image buffer to store previously generated images
            # self.fake_B_pool = ImagePool(cfg.dataset.train.pool_size)
            #
            # # define loss functions
            # self.criterionGAN = GANLoss(cfg.model.gan_mode)
            # self.criterionCycle = paddle.nn.L1Loss()
            # self.criterionIdt = paddle.nn.L1Loss()

            # self.build_lr_scheduler()
            self.optimizers['optimizer_G'] = build_optimizer(
                cfg.optimizer,
                cfg.lr,
                parameter_list=self.nets['generator'].parameters(),
                weight_decay=L2Decay(coeff=cfg.optimizer.weight_decay)
            )
            self.optimizers['optimizer_D'] = build_optimizer(
                cfg.optimizer,
                cfg.lr,
                parameter_list=self.nets['discriminator'].parameters(),
                weight_decay=L2Decay(coeff=cfg.optimizer.weight_decay)
            )
            self.optimizers['optimizer_E'] = build_optimizer(
                cfg.optimizer,
                cfg.lr,
                parameter_list=self.nets['style_encoder'].parameters(),
                weight_decay=L2Decay(coeff=cfg.optimizer.weight_decay)
            )
            self.optimizers['optimizer_F'] = build_optimizer(
                cfg.optimizer,
                cfg.lr_f,
                parameter_list=self.nets['mapping_network'].parameters(),
                weight_decay=L2Decay(coeff=cfg.optimizer.weight_decay)
            )

            self.lambda_reg = cfg.model.discriminator.lambda_reg
            self.lambda_cyc = cfg.model.discriminator.lambda_cyc
            self.lambda_ds = cfg.model.discriminator.lambda_ds
            self.lambda_sty = cfg.model.discriminator.lambda_sty
            self.w_hpf = cfg.model.generator.w_hpf

    def convert_input(self, x, transpose=False):
        """
        convert input numpy array to tensor
        uint8 --> float32
        NHWC --> NCHW
        :param x:
        :return:
        """
        # x = paddle.to_tensor(x)
        x = paddle.cast(x, dtype="float32")
        if transpose:
            x = paddle.transpose(x, perm=[0, 3, 1, 2])
        return x

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Args:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        mode = 'train' if self.is_train else 'test'
        self.x_src = self.convert_input(input["x_src"], transpose=True)
        self.x_ref = self.convert_input(input["x_ref"], transpose=True)
        self.x_ref2 = self.convert_input(input["x_ref2"], transpose=True)

        self.y_src = self.convert_input(input["y_src"])
        self.y_ref = self.convert_input(input["y_ref"])
        self.z_trg = self.convert_input(input["z_trg"])
        self.z_trg2 = self.convert_input(input["z_trg2"])

    def forward(self):
        """
        Run forward pass; called by both functions <optimize_parameters> and <test>.
        """
        # if hasattr(self, 'real_A'):
        #     self.fake_B = self.nets['netG_A'](self.real_A)  # G_A(A)
        #     self.rec_A = self.nets['netG_B'](self.fake_B)  # G_B(G_A(A))
        #
        #     # visual
        #     self.visual_items['real_A'] = self.real_A
        #     self.visual_items['fake_B'] = self.fake_B
        #     self.visual_items['rec_A'] = self.rec_A
        #
        # if hasattr(self, 'real_B'):
        #     self.fake_A = self.nets['netG_B'](self.real_B)  # G_B(B)
        #     self.rec_B = self.nets['netG_A'](self.fake_A)  # G_A(G_B(B))
        #
        #     # visual
        #     self.visual_items['real_B'] = self.real_B
        #     self.visual_items['fake_A'] = self.fake_A
        #     self.visual_items['rec_B'] = self.rec_B

        out = self.nets["discriminator"](self.x_src, self.y_src)

        # with fake images
        with paddle.fluid.dygraph.no_grad():
            if self.z_trg is not None:
                s_trg = self.nets["mapping_network"](self.z_trg, self.y_trg)
            else:  # x_ref is not None
                s_trg = self.nets["style_encoder"](self.x_ref, self.y_trg)

            x_fake = self.nets["generator"](self.x_src, s_trg, masks=None)
        x_fake = self.nets["generator"](self.x_src, s_trg, masks=None)

    def adv_loss(self, logits, target):
        assert target in [1, 0]
        targets = paddle.full(logits.shape, fill_value=target)

        loss = sigmoid_cross_entropy_with_logits(logits, targets)
        return loss

    # TODO find a way to implement autograd
    def r1_reg(self, d_out, x_in):
        return 0.0
        from paddle import fluid
        # zero-centered gradient penalty for real images
        batch_size = x_in.shape[0]
        try:
            grad_dout = fluid.dygraph.grad(
                outputs=d_out.sum(), inputs=x_in,
                create_graph=False, retain_graph=True, only_inputs=True
            )[0]
            grad_dout2 = porch.Tensor(grad_dout).pow(2)
            assert (grad_dout2.shape == x_in.shape)
            reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
            return reg
        except:
            return 0.0

    def compute_d_loss(self, x_real, y_org, y_trg, z_trg=None, x_ref=None, masks=None):
        assert (z_trg is None) != (x_ref is None)
        # with real images
        x_real.stop_gradient = False
        out = self.nets["discriminator"](x_real, y_org)
        loss_real = self.adv_loss(out, 1)
        loss_reg = self.r1_reg(out, x_real)

        # with fake images
        with paddle.fluid.dygraph.no_grad():
            if z_trg is not None:
                s_trg = self.nets["mapping_network"](z_trg, y_trg)
            else:  # x_ref is not None
                s_trg = self.nets["style_encoder"](x_ref, y_trg)

            x_fake = self.nets["generator"](x_real, s_trg, masks=masks)
        out = self.nets["discriminator"](x_fake, y_trg)
        loss_fake = self.adv_loss(out, 0)

        loss = paddle.fluid.layers.sum(loss_real + loss_fake + self.lambda_reg * loss_reg)
        return loss
               # Munch(real=loss_real.numpy().flatten()[0],
               #             fake=loss_fake.numpy().flatten()[0],
               #             reg=loss_reg)

    def compute_g_loss(self, x_real, y_org, y_trg, z_trgs=None, x_refs=None, masks=None):
        assert (z_trgs is None) != (x_refs is None)
        if z_trgs is not None:
            z_trg, z_trg2 = z_trgs
        if x_refs is not None:
            x_ref, x_ref2 = x_refs

        # adversarial loss
        if z_trgs is not None:
            s_trg = self.nets["mapping_network"](z_trg, y_trg)
        else:
            s_trg = self.nets["style_encoder"](x_ref, y_trg)

        x_fake = self.nets["generator"](x_real, s_trg, masks=masks)
        out = self.nets["discriminator"](x_fake, y_trg)
        loss_adv = self.adv_loss(out, 1)

        # style reconstruction loss
        s_pred = self.nets["style_encoder"](x_fake, y_trg)
        loss_sty = paddle.mean(paddle.abs(s_pred - s_trg))

        # diversity sensitive loss
        if z_trgs is not None:
            s_trg2 = self.nets["mapping_network"](z_trg2, y_trg)
        else:
            s_trg2 = self.nets["style_encoder"](x_ref2, y_trg)
        x_fake2 = self.nets["generator"](x_real, s_trg2, masks=masks)
        x_fake2 = x_fake2.detach()
        loss_ds = paddle.mean(paddle.abs(x_fake - x_fake2))

        # cycle-consistency loss
        # masks = self.nets.fan.get_heatmap(x_fake) if self.w_hpf > 0 else None
        masks = None
        s_org = self.nets["style_encoder"](x_real, y_org)
        x_rec = self.nets["generator"](x_fake, s_org, masks=masks)
        loss_cyc = paddle.mean(paddle.abs(x_rec - x_real))

        loss = loss_adv + self.lambda_sty * loss_sty \
               - self.lambda_ds * loss_ds + self.lambda_cyc * loss_cyc
        return loss
            # , Munch(adv=loss_adv.numpy().flatten()[0],
            #                sty=loss_sty.numpy().flatten()[0],
            #                ds=loss_ds.numpy().flatten()[0],
            #                cyc=loss_cyc.numpy().flatten()[0]), x_fake[0]

    def _reset_grad(self):
        for optimizer in self.optimizers.values():
            optimizer.clear_gradients()

    def optimize_parameters(self):
        """
        Calculate losses, gradients, and update network weights; called in every training iteration
        """
        # masks = self.nets.fan.get_heatmap(self.x_src) if self.w_hpf > 0 else None
        masks= None

        # train discriminator first 100 iterations
        d_loss = self.compute_d_loss(
            self.x_src, self.y_src, self.y_ref, z_trg=self.z_trg, masks=masks)
        self._reset_grad()
        d_loss.backward()
        self.optimizers["optimizer_D"].minimize(d_loss)

        d_loss = self.compute_d_loss(
            self.x_src, self.y_src, self.y_ref, x_ref=self.x_ref, masks=masks)
        self._reset_grad()
        d_loss.backward()
        self.optimizers["optimizer_D"].minimize(d_loss)

        # train the generator  g_losses_latent, sample_1
        g_loss = self.compute_g_loss(
            self.x_src, self.y_src, self.y_ref, z_trgs=[self.z_trg, self.z_trg2], masks=masks)
        self._reset_grad()
        g_loss.backward()
        self.optimizers["optimizer_G"].minimize(g_loss)
        self.optimizers["optimizer_E"].minimize(g_loss)
        self.optimizers["optimizer_F"].minimize(g_loss)

        # , g_losses_ref, sample_2
        g_loss = self.compute_g_loss(
            self.x_src, self.y_src, self.y_ref, x_refs=[self.x_ref, self.x_ref2], masks=masks)
        self._reset_grad()
        g_loss.backward()
        self.optimizers["optimizer_G"].minimize(g_loss)
