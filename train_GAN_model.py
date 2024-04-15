import shutil
import sys
import wandb
from torchvision import models, utils, datasets, transforms
import random
import timm
sys.path.append("models")
sys.path.append("losses")
import torch
import torch.optim as optim
from  tqdm import tqdm
from pytorch_msssim import ms_ssim
import math
import config
from models import compress_models
from losses.rate_distortion import RateDistortionClassificationLoss
from models.efficientnet import EfficientNet_b3
import lpips
import torch
import compressai.zoo as zoo

# 初始化LPIPS模型
loss_fn_alex = lpips.LPIPS(net='alex')  # 使用alexnet
loss_fn_vgg = lpips.LPIPS(net='vgg')    # 使用vgg

def calculate_lpips_distance(img0, img1, net_type='alex'):
    """
    计算图像img0和img1之间的LPIPS距离。

    参数:
    img0 (torch.Tensor): 第一张未正规化的图像，大小为(1, 3, H, W)
    img1 (torch.Tensor): 第二张未正规化的图像，大小为(1, 3, H, W)
    net_type (str): 使用的网络类型，'alex'或'vgg'

    返回:
    float: LPIPS距离
    """
    # 正规化图像数据到[-1, 1]
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    img0_normalized = normalize(img0)
    img1_normalized = normalize(img1)

    # 根据选择的网络类型计算LPIPS距离
    if net_type == 'alex':
        distance = loss_fn_alex(img0_normalized, img1_normalized)
    elif net_type == 'vgg':
        distance = loss_fn_vgg(img0_normalized, img1_normalized)
    else:
        raise ValueError("Unsupported net_type. Choose 'alex' or 'vgg'.")

    return distance.item()  # 返回一个python float

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()

def compute_perceptual_similarity(x_content, recon_content):
    diff = x_content - recon_content
    diff_sq = torch.square(diff)
    num = torch.mean(diff_sq)
    den = torch.mean(torch.square(x_content))
    ps = num / den
    return ps

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


from tqdm import tqdm


def train_one_epoch_tqdm(
        generator, discriminator, criterion, train_dataloader,
        gen_optimizer, disc_optimizer, aux_optimizer, epoch, clip_max_norm
):
    generator.train()
    discriminator.train()
    device = next(generator.parameters()).device

    # 使用 tqdm 显示进度条
    pbar = tqdm(train_dataloader, desc=f"Train epoch {epoch}")

    for i, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)

        # 一. 更新判别器:
        # 重置梯度
        disc_optimizer.zero_grad()

        # 生成数据
        with torch.no_grad():  # 避免在这一步计算生成器的梯度
            out_generator = generator(inputs)
        out_real = discriminator(inputs)
        out_fake = discriminator(out_generator["x_hat"].detach())  # 避免梯度流向生成器

        # 计算损失
        D_L = criterion.discriminator_loss(out_generator, out_real, out_fake, inputs, labels)

        # 反向传播和优化
        D_L.backward()
        if clip_max_norm:
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), clip_max_norm)
        disc_optimizer.step()

        # 二. 更新生成器:
        # 生成数据
        out_generator = generator(inputs)
        out_fake = discriminator(out_generator["x_hat"])

        # 计算损失
        G_L, bpp_loss, GM = criterion.generator_loss(out_generator, out_real, out_fake, inputs, labels)

        # 反向传播和优化 - 主优化器
        gen_optimizer.zero_grad()
        G_L.backward()
        if clip_max_norm:
            torch.nn.utils.clip_grad_norm_(generator.parameters(), clip_max_norm)
        gen_optimizer.step()

        # 反向传播和优化 - 辅助优化器
        aux_loss = generator.aux_loss()
        aux_optimizer.zero_grad()
        aux_loss.backward()
        aux_optimizer.step()

        # 更新进度条的描述
        pbar.set_postfix(
            disc_loss=D_L.item(),
            gen_loss=G_L.item(),
            bpp_loss=bpp_loss.item(),
            mse_loss=GM.item(),
            aux_loss=aux_loss.item(),
        )
def test_epoch(epoch, test_dataloader, generator,discriminator,criterion):
    generator.eval()
    discriminator.eval()

    device = next(generator.parameters()).device

    model = timm.create_model("resnet50.a1_in1k", pretrained=False)
    weights = torch.load("/home/user2/pretrained_models/resnet50.a1_in1k.bin")
    model.load_state_dict(weights)

    classifier_model = model

    classifier_model.to(device)
    classifier_model.eval()

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr=AverageMeter()
    accuracy=AverageMeter()
    with torch.no_grad():
        for d in test_dataloader:
            labels = d[1].to(device)

            d = d[0].to(device)

            out_generator=generator(d)
            out_real=discriminator(d)
            out_fake=discriminator(out_generator["x_hat"])
            # D_L, G_L,bpp_loss,G_M
            out_criterion = criterion(out_generator,out_real,out_fake, d, labels)
        
            aux_loss.update(generator.aux_loss())
            bpp_loss.update(out_criterion[2])
            mse_loss.update(out_criterion[3])
            psnr.update(compute_psnr(d, out_generator["x_hat"]))
            loss.update(out_criterion[4])
            # 识别精度
            out_labels =classifier_model(out_generator["x_hat"])
            _, predicted = torch.max(out_labels, 1)
            accuracy_current = (predicted == labels).sum().item() / len(labels)
            accuracy.update(accuracy_current)
    wandb.log({"acc": accuracy.avg, "loss": loss.avg,"bpp":bpp_loss.avg,"PSNR":psnr.avg,"mse_loss":mse_loss.avg})
    print(
        f"Test epoch {epoch}: Average results:"
        f"\tPSNR: {psnr.avg:.3f} |"
        f"\tAccuracy: {accuracy.avg:.5f} |"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tmse loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.3f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    return loss.avg

def save_checkpoint(state,filename="checkpoint_GAN.pth.tar"):
    torch.save(state, filename)




def main(argv):
    torch.autograd.set_detect_anomaly(True)
    config.print_config()

    device = config.device
    train_dataloader,test_dataloader = config.get_imagenet_dataloader()
    # generator = zoo.bmshj2018_hyperprior(1, metric='mse', pretrained=True)
    generator=compress_models.ScaleHyperpriorWithPostProcess(192,320)
    checkpoint = torch.load("/home/user2/LiDexin/ICR/PostProcess.pth.tar")
    generator.post_process.load_state_dict(checkpoint)
    # 冻结post_process模块的参数
    for param in generator.post_process.parameters():
        param.requires_grad = False
    discriminator = compress_models.Discriminator(num_classes=1000, dropout_rate=0.5)
    generator=generator.to(device)
    discriminator=discriminator.to(device)

    #配置优化器
    gen_optimizer,aux_optimizer = config.get_gen_optimizer(generator)
    disc_optimizer = config.get_disc_optimizer(discriminator)

    criterion = RateDistortionClassificationLoss(lmbda=config.lmbda,mu=config.mu,nu=config.nu,metric="mse")

    last_epoch = 0
    if config.load_checkpoint:  # load from previous checkpoint
        print("Loading checkpoint...")
        checkpoint = torch.load("/home/user2/LiDexin/ICR/checkpoint_GAN.pth.tar", map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])


    for epoch in range(last_epoch, config.num_epochs):
        print(f"\nGen Learning rate: {gen_optimizer.param_groups[0]['lr']}\n")
        print(f"\nDisc Learning rate: {disc_optimizer.param_groups[0]['lr']}\n")
        train_one_epoch_tqdm(
            generator,
            discriminator,
            criterion,
            train_dataloader,
            gen_optimizer,
            disc_optimizer,
            aux_optimizer,
            epoch,
            config.clip_max_norm)

        if epoch%50==0:
            loss= test_epoch(epoch, test_dataloader, generator,discriminator,criterion)
            save_checkpoint(
                state={
                    "epoch": epoch,
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict()
                },
            )

if __name__ == "__main__":
    wandb.init(
        project="compressai", config={
            "learning_rate": config.lr,
            "bpp":0.8
        })
    main(sys.argv[1:])
    wandb.finish()