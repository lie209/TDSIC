from torchvision import models
import torch.nn as nn
import torch
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        # model=models.efficientnet_v2(pretrained=True)
        model = models.resnet50(pretrained=True)

        # # 冻结模型参数
        # for param in model.parameters():
        #     param.requires_grad = False

        model.classifier = torch.nn.Sequential()
        model.eval()
        # 修改之处:添加self.model
        self.model = model

    def forward(self, x):
        # 新增forward方法
        return self.model(x)

    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

if __name__ == '__main__':
    model = ResNet50()
    img = torch.randn(1, 3, 320, 320)
    output = model(img)
    print(output.shape)