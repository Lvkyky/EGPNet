import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log


class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class EAM(nn.Module):
    def __init__(self):
        super(EAM, self).__init__()
        # self.reduce1 = Conv1x1(256, 64)
        self.reduce4 = Conv1x1(512, 256)
        self.block = nn.Sequential(
            ConvBNR(256 + 64, 256, 3),
            ConvBNR(256, 256, 3),
            nn.Conv2d(256, 1, 1))

    def forward(self, x2,x4):
        size = x2.size()[2:]
        # x1 = self.reduce1(x1)
        x4 = self.reduce4(x4)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        out = torch.cat((x4, x2), dim=1)
        out = self.block(out)

        return out
class EAMP(nn.Module):
    def __init__(self,wide):
        super(EAMP, self).__init__()
        # self.reduce1 = Conv1x1(256, 64)
        shalowwsize = wide//8
        deepsize = wide
        self.reduce4 = Conv1x1(deepsize, deepsize//2)
        self.block = nn.Sequential(
            ConvBNR(deepsize//2 + shalowwsize, deepsize//2,3),
            ConvBNR(deepsize//2, deepsize//2, 3),
            nn.Conv2d(deepsize//2, 1, 1))

    def forward(self, x2,x4):
        size = x2.size()[2:]
        # x1 = self.reduce1(x1)
        x4 = self.reduce4(x4)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        out = torch.cat((x4, x2), dim=1)
        out = self.block(out)

        return out
class EFM(nn.Module):
    def __init__(self, channel):
        super(EFM, self).__init__()
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d = ConvBNR(channel, channel, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, c, att):
        if c.size() != att.size():
            att = F.interpolate(att, c.size()[2:], mode='bilinear', align_corners=False)
        x = c * att + c
        x = self.conv2d(x)
        wei = self.avg_pool(x)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        x = x * wei

        return x

