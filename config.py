import sys
import torch
sys.path.append("models")
sys.path.append("losses")

import torch.optim as optim
from compressai.datasets import ImageFolder
from torch.utils.data import Subset
from compressai.optimizers import net_aux_optimizer
from models import compress_models
from torch.utils.data import Dataset, DataLoader
from torchvision import models, utils, datasets, transforms
import numpy as np
import sys
import os
from PIL import Image
import shutil

# 迭代次数
num_epochs=5000

#批量大小
batch_size=64
# 学习率
lr=1e-4
aux_lr=1e-3
#图像相似度超参
lmbda=1
#bpp权衡超参 0.01~0.1之间
mu=0.05
#图识别像损权衡超参
nu=1
#要分割的图像大小
patch_size=(256,256)
#梯削减度最大范数
clip_max_norm=1
# 是否保存权重
save=True
#是否加载权重
load_checkpoint=True
#对于ImageNet数据集，定义要加载的数据百分比
data_percentage = 0.03
# 使用GPU还是CPU
device="cuda"

def configure_optimizers(net):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": lr},
        "aux": {"type": "Adam", "lr": aux_lr},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]

def get_gen_optimizer(net):
    gen_parameters = set(p for n, p in net.named_parameters() if not n.endswith(".quantiles"))
    aux_parameters = set(p for n, p in net.named_parameters() if n.endswith(".quantiles"))
    gen_optimizer = optim.Adam(gen_parameters, lr=lr)
    aux_optimizer = optim.Adam(aux_parameters, lr=aux_lr)
    return gen_optimizer,aux_optimizer

def get_disc_optimizer(net):
    disc_optimizer = optim.Adam(net.parameters(), lr=lr)
    return disc_optimizer

def get_imagenet_dataloader():
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    train_dir = 'ImageNet/train'
    val_dir = 'ImageNet/val'
    train_transforms = transforms.Compose([
    transforms.Resize(320, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(patch_size),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

    test_transforms = transforms.Compose([
    transforms.Resize(320, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(patch_size),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(val_dir, transform=test_transforms)

    # Reduce the size of the datasets
    train_size = int(data_percentage * len(train_dataset))


    train_indices = np.random.choice(np.arange(len(train_dataset)), size=train_size, replace=False)


    train_dataset = Subset(train_dataset, train_indices)


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


    return train_loader, test_loader

def get_full_imagenet_testloader():


    val_dir = 'ImageNet/val'

    test_transforms = transforms.Compose([
        transforms.Resize(320, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(patch_size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = datasets.ImageFolder(val_dir, transform=test_transforms)


    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return test_loader


def get_caltech101_dataloader():


    train_dir = 'Caltech101/train'
    val_dir = 'Caltech101/val'
    train_transforms = transforms.Compose([
        transforms.Resize(320, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(patch_size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(320, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(patch_size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(val_dir, transform=test_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, test_loader


def copy_images(dataset, indices, target_root):
    # 确保目标根目录存在
    os.makedirs(target_root, exist_ok=True)

    for idx in indices:
        # 获取原始路径和目标路径
        img_path, _ = dataset.samples[idx]
        relative_path = os.path.relpath(img_path, os.path.dirname(dataset.root))
        target_path = os.path.join(target_root, relative_path)

        # 创建目标路径的父目录
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        # 复制图像到目标路径
        shutil.copyfile(img_path, target_path)

def save_images():
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    val_dir = 'ImageNet/val'
    # 加载验证集
    test_dataset_full = datasets.ImageFolder(val_dir)
    # 计算希望保存的图片数量
    test_size = int(data_percentage * 3*len(test_dataset_full))

    # 随机选择指定数量的图片索引
    test_indices = np.random.choice(np.arange(len(test_dataset_full)), size=test_size, replace=False)

    # 复制测试集图像到新的本地目录
    target_root = '/home/user2/LiDexin/ICR/saved_images'  # 目标根目录

    # 请注意，这里我们传递的是完整的数据集和子集的索引
    copy_images(test_dataset_full, test_indices, target_root)

def print_config():
    print(f"""Config:
    Num Epochs = {num_epochs}
    Batch Size = {batch_size}
    Learning Rate = {lr}
    Aux Learning Rate = {aux_lr}
    Lambda = {lmbda}
    Mu = {mu} 
    Nu = {nu}
    Patch Size = {patch_size}
    Clip Max Norm = {clip_max_norm}
    Save = {save}
    Load_checkpoint={load_checkpoint}
    Data Percentage = {data_percentage}
    Device = {device}
             """)


if __name__ == '__main__':
    save_images()