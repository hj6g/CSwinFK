import torch.nn as nn
import torch.nn.functional as F_nn


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CNNBackbone_Res64(nn.Module):
    def __init__(self, use_se: bool = True):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        self.block64 = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
            ResBlock(64)
        )

        self.se = SELayer(64, 16) if use_se else nn.Identity()
        self.upconv = nn.Conv2d(64, 256, 1, bias=False)

    def forward(self, x):
        x = self.stem(x)           
        x = self.block64(x)        
        x = self.se(x)             
        x = self.upconv(x)         
        x = F_nn.avg_pool2d(x, 2)  
        return x
