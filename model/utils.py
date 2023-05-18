import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import log



#############edge guidance#############
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
############edge guidance#############


############convolution block#########
class EnConvBlock(nn.Module):
    def __init__(self,input):
        super(EnConvBlock, self).__init__()
        if input == 3:
            self.conv1 = nn.Conv2d(in_channels=input, out_channels=32, kernel_size=(3, 3),padding=1)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3),padding=1)
            self.batch1 = nn.BatchNorm2d(32)
            self.batch2 = nn.BatchNorm2d(32)
        else:
            self.unit = nn.Conv2d(in_channels=input, out_channels=input * 2, kernel_size=(1, 1))
            self.conv1=nn.Conv2d(in_channels=input, out_channels=input*2, kernel_size=(3,3),padding=1)
            self.conv2=nn.Conv2d(in_channels=input*2, out_channels=input*2, kernel_size=(3,3),padding=1)
            self.batch1 = nn.BatchNorm2d(input*2)
            self.batch2 = nn.BatchNorm2d(input*2)
        self.relu = nn.ReLU()

    def forward(self,x):
        x=self.conv1(x)
        x=self.batch1(x)
        x=self.relu(x)

        x=self.conv2(x)
        x=self.batch2(x)
        x=self.relu(x)

        return x
class EnConvDiffBlock(nn.Module):
    def __init__(self,input):
        super(EnConvDiffBlock, self).__init__()
        self.flag = 0
        if input == 6:
            self.conv1 = nn.Conv2d(in_channels=input, out_channels=32, kernel_size=(3, 3),padding=1)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3),padding=1)
            self.batch1 = nn.BatchNorm2d(32)
            self.batch2 = nn.BatchNorm2d(32)
        else:
            self.conv1 = nn.Conv2d(in_channels=input, out_channels=input*2, kernel_size=(3,3),padding=1)
            self.conv2 = nn.Conv2d(in_channels=input*2, out_channels=input*2, kernel_size=(3,3),padding=1)
            self.batch1 = nn.BatchNorm2d(input*2)
            self.batch2 = nn.BatchNorm2d(input*2)

        self.relu=nn.ReLU()

    def forward(self,x):

        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu(x)

        return x
class DeConvBlock(nn.Module):
    def __init__(self,input):
        super(DeConvBlock, self).__init__()
        self.unit = nn.Conv2d(in_channels=input, out_channels=int(input / 2), kernel_size=(1, 1))
        self.conv1=nn.Conv2d(in_channels=input, out_channels=int(input/2), kernel_size=(3,3), padding=1)
        self.conv2=nn.Conv2d(in_channels=int(input/2), out_channels=int(input/2), kernel_size=(3,3), padding=1)
        self.batch1 = nn.BatchNorm2d(int(input/2))
        self.batch2 = nn.BatchNorm2d(int(input/2))
        self.relu=nn.ReLU()

    def forward(self,x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu(x)

        return x
###########convolution block###########

##########fuse bi-temporal features and difference features##########
class FuseBlock1(nn.Module):
    def __init__(self, input):
        super(FuseBlock1, self).__init__()
        self.fuse1 = nn.Sequential(nn.Conv2d(input*3, input*2, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(input*2),
                                       nn.ReLU()
                                       )

        self.fuse2 = nn.Sequential(nn.Conv2d(input*2, input, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(input),
                                   nn.ReLU()
                                   )

    def forward(self,input):
        x = self.fuse1(input)
        x = self.fuse2(x)
        return x
##########fuse bi-temporal features and difference features##########

###########encoder-decoder#############
#decoder
class Decoder(nn.Module):
    def __init__(self,wide):
        super(Decoder, self).__init__()
        self.stage4 = DeConvBlock((wide*16))
        self.stage3 = DeConvBlock((wide*8))
        self.stage2 = DeConvBlock((wide*4))
        self.stage1 = DeConvBlock((wide*2))

        self.up1 = torch.nn.ConvTranspose2d(wide*2, wide, kernel_size=(4,4), stride=(2, 2), padding=(1,1))
        self.up2 = torch.nn.ConvTranspose2d(wide*4, wide*2, kernel_size=(4,4), stride=(2, 2), padding=(1,1))
        self.up3 = torch.nn.ConvTranspose2d(wide*8, wide*4, kernel_size=(4,4), stride=(2, 2), padding=(1,1))
        self.up4 = torch.nn.ConvTranspose2d(wide*16, wide*8, kernel_size=(4,4), stride=(2, 2), padding=(1,1))

    def forward(self,x1,x2,x3,x4,x5):
        x = self.up4(x5)#512---->256
        x = self.stage4(torch.cat([x4, x], dim=1))#512--->256
        
        # we find that throwing coarse feture5 leads better performance.
        x = self.up3(x4)#256---->128
        x = self.stage3(torch.cat([x3, x], dim=1))#256--->128

        x = self.up2(x)#128---->64
        x = self.stage2(torch.cat([x2, x], dim=1))#128--->64

        x1_ = self.up1(x)#64---->32
        x = self.stage1(torch.cat([x1, x1_], dim=1))#64--->32

        return x
#bi-temporal encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        self.stage1 = EnConvBlock(3)
        self.stage2 = EnConvBlock(32)
        self.stage3 = EnConvBlock(64)
        self.stage4 = EnConvBlock(128)
        self.stage5 = EnConvBlock(256)

    def forward(self,x):
        x1 = self.stage1(x)
        x2 = self.stage2(self.pool(x1))
        x3 = self.stage3(self.pool(x2))
        x4 = self.stage4(self.pool(x3))
        x5 = self.stage5(self.pool(x4))
        return x1,x2,x3,x4,x5
#difference encoder
class EncoderDiff(nn.Module):
    def __init__(self):
        super(EncoderDiff, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        self.stage1 = EnConvDiffBlock(6)
        self.stage2 = EnConvDiffBlock(32)
        self.stage3 = EnConvDiffBlock(64)
        self.stage4 = EnConvDiffBlock(128)
        self.stage5 = EnConvDiffBlock(256)

    def forward(self,x,x1,x2,x3,x4):
        x1diff = self.stage1(x)
        x2diff = self.stage2((self.pool(x1diff)+self.pool(x1)))#32--->64
        x3diff = self.stage3((self.pool(x2diff)+self.pool(x2)))#64--->128
        x4diff = self.stage4((self.pool(x3diff)+self.pool(x3)))#128--->256
        x5diff = self.stage5((self.pool(x4diff)+self.pool(x4)))#256--->512

        return x1diff,x2diff,x3diff,x4diff,x5diff
###########encoder-decoder#############




