import math
import torch.nn.functional as F
import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim

class RateDistortionClassificationLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=0.01, mu=0.05, nu=1, metric="mse", return_type="all"):
        super().__init__()
        self.lmbda = lmbda
        self.mu = mu
        self.nu = nu
        self.return_type = return_type

        if metric == "mse":
            self.metric = nn.MSELoss()
        elif metric == "ms-ssim":
            # 确保 ms_ssim 函数已经在某个地方定义
            self.metric = ms_ssim
        else:
            raise NotImplementedError(f"{metric} is not implemented!")

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def discriminator_loss(self, out_generator,out_real,out_fake, target, label):
        out_disc_real=out_real
        out_disc_fake=out_fake
        # 判别器损失
        D_real_logits_S, D_real_logits_C = out_disc_real
        D_fake_logits_S, D_fake_logits_C = out_disc_fake

        D_S_real = torch.mean(self.bce_loss(D_real_logits_S, torch.ones_like(D_real_logits_S)))
        D_S_fake = torch.mean(self.bce_loss(D_fake_logits_S, torch.zeros_like(D_fake_logits_S)))
        D_C_real = torch.mean(self.cross_entropy_loss(D_real_logits_C, label))

        D_L = 0.5 * (D_S_real + D_S_fake) + self.nu * D_C_real
        return D_L
    
    def generator_loss(self, out_generator,out_real,out_fake, target, label):
        N, _, H, W = target.size()
        num_pixels = N * H * W

    
        out_generator= out_generator
        out_disc_fake=out_fake
        # 编码率损失
        bpp_loss = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in out_generator["likelihoods"].values()
        )

        # 判别器和分类器的损失
        
        D_fake_logits_S, D_fake_logits_C = out_disc_fake
        # 生成器损失
        G_M = self.metric(out_generator["x_hat"], target)
        G_S_fake = torch.mean(self.bce_loss(D_fake_logits_S, torch.ones_like(D_fake_logits_S)))
        G_C_fake = torch.mean(self.cross_entropy_loss(D_fake_logits_C, label))
        # G_L = 0.02*255 ** 2 *G_M + bpp_loss + 0.0001 * (G_S_fake + G_C_fake)
        G_L = 0.001*255 ** 2 *G_M + bpp_loss + 0.001 * (G_S_fake + G_C_fake)
        return G_L,bpp_loss,G_M

    def generator_loss1(self, out_generator, out_real, out_fake, target, label):
        N, _, H, W = target.size()
        num_pixels = N * H * W

        out_generator = out_generator
        out_disc_fake = out_fake
        # 编码率损失
        bpp_loss = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in out_generator["likelihoods"].values()
        )
        y_hat_loss=self.metric(out_generator["y"], out_generator["y_hat"])
        # 判别器和分类器的损失

        D_fake_logits_S, D_fake_logits_C = out_disc_fake
        # 生成器损失
        G_M = self.metric(out_generator["x_hat"], target)
        G_S_fake = torch.mean(self.bce_loss(D_fake_logits_S, torch.ones_like(D_fake_logits_S)))
        G_C_fake = torch.mean(self.cross_entropy_loss(D_fake_logits_C, label))
        # G_L = 0.02*255 ** 2 *G_M + bpp_loss + 0.0001 * (G_S_fake + G_C_fake)
        G_L = 0.03 * 255 ** 2 * G_M + bpp_loss + 0.001*y_hat_loss+0.0001 * (G_S_fake + G_C_fake)
        return G_L, bpp_loss, G_M

    def forward(self, out_generator,out_real,out_fake, target, label):
        N, _, H, W = target.size()
        num_pixels = N * H * W

    
        out_generator= out_generator
        out_disc_real=out_real
        out_disc_fake=out_fake
        # 编码率损失
        bpp_loss = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in out_generator["likelihoods"].values()
        )

        # 判别器和分类器的损失
        D_real_logits_S, D_real_logits_C = out_disc_real
        D_fake_logits_S, D_fake_logits_C = out_disc_fake

        D_S_real = self.bce_loss(D_real_logits_S, torch.ones_like(D_real_logits_S))
        D_S_fake = self.bce_loss(D_fake_logits_S, torch.zeros_like(D_fake_logits_S))
        D_C_real = self.cross_entropy_loss(D_real_logits_C, label)

        D_L = 0.5 * (D_S_real + D_S_fake) + self.nu * D_C_real

        # 生成器损失
        G_M = self.metric(out_generator["x_hat"], target)
        G_S_fake = torch.mean(self.bce_loss(D_fake_logits_S, torch.ones_like(D_fake_logits_S)))
        G_C_fake = torch.mean(self.cross_entropy_loss(D_fake_logits_C, label))
        G_L = 0.005*255 ** 2 *G_M + bpp_loss + 0.0001 * (G_S_fake + G_C_fake)
        loss=G_L+D_L

        return D_L, G_L,bpp_loss,G_M,loss

    def compress_loss(self, out_generator, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in out_generator["likelihoods"].values()
        )
        if self.metric == ms_ssim:
            out["ms_ssim_loss"] = self.metric(out_generator["x_hat"], target, data_range=1)
            distortion = 1 - out["ms_ssim_loss"]
        else:
            out["mse_loss"] = self.metric(out_generator["x_hat"], target)
            distortion = 255 ** 2 * out["mse_loss"]
        out["loss"] = 0.01 * distortion + out["bpp_loss"]
        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]