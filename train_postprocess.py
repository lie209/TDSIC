import sys

from torchvision import models
from tqdm import tqdm

sys.path.append("models")
import torch
import torch.nn as nn
import torch.optim as optim
import config
from models import compress_models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 获得训练数据
train_dataloader,test_dataloader = config.get_caltech101_dataloader()
# 2. 加载预训练的ResNet50模型
model = compress_models.PostProcess(192).to(device)
checkpoint = torch.load("/home/user2/LiDexin/ICR/PostProcess.pth.tar", map_location=device)
model.load_state_dict(checkpoint)
compress_model=compress_models.ScaleHyperpriorForPostProcess(128,192)
checkpoint = torch.load("/home/user2/LiDexin/ICR/bmshj2018-hyperprior-4-de1b779c.pth.tar", map_location=device)
compress_model.load_state_dict(checkpoint, strict=False)
compress_model=compress_model.to(device)
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer=optim.Adam(model.parameters(), lr= 0.001)
# 优化学习率策略
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=4)

# 训练
num_epochs = 500
print(f"Training PostProcess started!\n")

def save_checkpoint(state, filename="PostProcess1.pth.tar"):
    torch.save(state, filename)



best_acc = 0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    train_dataloader_tqdm = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} (Train)", ncols=100)
    for images, labels in train_dataloader_tqdm:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        out_compress=compress_model(images)
        y=out_compress["y"]
        y_hat=out_compress["y_hat"]
        y_hat1 = model(y_hat)
        loss = criterion(y, y_hat1)

        loss.backward()
        optimizer.step()
        running_loss = loss.item()
        train_dataloader_tqdm.set_description(f"Epoch {epoch + 1}/{num_epochs} (Train) Loss: {running_loss:.4f}")

    lr_scheduler.step(running_loss)

    # 测试阶段
    correct = 0
    total = 0
    model.eval()

    test_dataloader_tqdm = tqdm(test_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} (Test)", ncols=100)
    with torch.no_grad():
        for images, labels in test_dataloader_tqdm:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out_compress = compress_model(images)
            y = out_compress["y"]
            y_hat = out_compress["y_hat"]
            y_hat1 = model(y_hat)
            loss = criterion(y, y_hat1)
            total += labels.size(0)
            correct += loss
    loss = correct / total
    print(f'Epoch {epoch + 1} Test Accuracy: {loss:.4f}')
    if epoch%20==0:
        save_checkpoint(state=model.state_dict())

print('Finished training!')


