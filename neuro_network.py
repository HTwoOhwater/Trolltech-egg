import torch
import torch.nn as nn


class Simple(nn.Module):
    def __init__(self):
        super(Simple, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, stride=2, padding=1, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, stride=2, padding=1, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=1),
            nn.AdaptiveAvgPool2d((2, 2))
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=2 * 2 * 16, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=10)
        )

    def forward(self, x):
        x = self.fc1(x)
        x = torch.flatten(x, 1)
        x = self.fc2(x)
        return x


class AlexNet(nn.Module):
    def __init__(self, config):
        super(AlexNet, self).__init__()
        self.config = config
        # 定义卷积层和池化层
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # 自适应层，将上一层的数据转换成6x6大小
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.config),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
