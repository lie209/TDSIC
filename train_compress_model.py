import shutil
import sys

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
        generator, criterion, train_dataloader,
        gen_optimizer, aux_optimizer, epoch, clip_max_norm
):
    generator.train()
    device = next(generator.parameters()).device

    # 使用 tqdm 显示进度条
    pbar = tqdm(train_dataloader, desc=f"Train epoch {epoch}")

    for i, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)

        # 二. 更新生成器:
        # 生成数据
        out_generator = generator(inputs)

        # 计算损失
        out= criterion.compress_loss(out_generator, inputs)

        # 反向传播和优化 - 主优化器
        gen_optimizer.zero_grad()
        out["loss"].backward()
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
            gen_loss=out["loss"].item(),
            bpp_loss=out["bpp_loss"].item(),
            mse_loss=out["mse_loss"].item(),
            aux_loss=aux_loss.item(),
        )
def test_epoch(epoch, test_dataloader, generator,criterion):
    generator.eval()

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
            # D_L, G_L,bpp_loss,G_M
            out = criterion.compress_loss(out_generator, d)
        
            aux_loss.update(generator.aux_loss())
            bpp_loss.update(out["bpp_loss"])
            mse_loss.update(out["mse_loss"])
            psnr.update(compute_psnr(d, out_generator["x_hat"]))
            loss.update(out["loss"])
            # 识别精度
            out_labels =classifier_model(out_generator["x_hat"])
            _, predicted = torch.max(out_labels, 1)
            accuracy_current = (predicted == labels).sum().item() / len(labels)
            accuracy.update(accuracy_current)

    print(
        f"Test epoch {epoch}: Average results:"
        f"\tPSNR: {psnr.avg:.3f} |"
        f"\tAccuracy: {accuracy.avg:.2f} |"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tmse loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    return loss.avg

def save_checkpoint(state, is_best, filename="checkpoint_compress.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "checkpoint_best_loss.pth.tar")

def main(argv):
    config.print_config()
    device = config.device
    train_dataloader,test_dataloader = config.get_imagenet_dataloader()
    generator=compress_models.ScaleHyperpriorWithPostProcess(128,192)
    generator.to(device)
    checkpoint = torch.load("/home/user2/LiDexin/ICR/PostProcess.pth.tar", map_location=device)
    generator.post_process.load_state_dict(checkpoint)
    # 冻结post_process模块的参数
    for param in generator.post_process.parameters():
        param.requires_grad = False
    #配置优化器
    gen_optimizer,aux_optimizer = config.get_gen_optimizer(generator)

    criterion = RateDistortionClassificationLoss(metric="mse")

    last_epoch = 0
    if  config.load_checkpoint:  # load from previous checkpoint
        print("Loading", "checkpoint_compress.pth.tar")
        checkpoint = torch.load("checkpoint_compress.pth.tar", map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        generator.load_state_dict(checkpoint["generator"])


    best_loss = float("inf")
    for epoch in range(last_epoch, config.num_epochs):
        print(f"\nGen Learning rate: {gen_optimizer.param_groups[0]['lr']}\n")
        train_one_epoch_tqdm(
            generator,
            criterion,
            train_dataloader,
            gen_optimizer,
            aux_optimizer,
            epoch,
            config.clip_max_norm
        )
        loss= test_epoch(epoch, test_dataloader, generator,criterion)
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        if config.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "generator": generator.state_dict(),
                },
                is_best,
            )

if __name__ == "__main__":
    main(sys.argv[1:])