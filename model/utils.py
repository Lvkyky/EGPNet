import torch
import torch.nn as nn
import torch.nn.functional as F


#UP,Down
def DS2(x):
    return F.avg_pool2d(x, 2)
def DS4(x):
    return F.avg_pool2d(x, 4)
def US2(x):
    return F.interpolate(x, scale_factor=2, mode='bilinear')

################Convolution block###############
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


###########module of fusing different features##########
class FuseBlock(nn.Module):
    def __init__(self, input,output):
        super(FuseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input, out_channels=output, kernel_size=(3, 3), padding=1)
        self.batch1 = nn.BatchNorm2d(output)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu(x)

        return x
class FuseBlock0(nn.Module):
    def __init__(self, wide):
        super(FuseBlock0, self).__init__()
        self.fuse2 = nn.Sequential(nn.Conv2d(wide*2, wide, kernel_size=(3,3), stride=(1,1), padding=1, bias=False),
                                   nn.BatchNorm2d(wide),
                                   nn.ReLU()
                                   )
    def forward(self,input):
        x = self.fuse2(input)
        return x
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


#################Encoder-Decoder####################
#Decoder
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

#difference encoder with AS
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

#difference encoder with MS
class EncoderDiff_MS(nn.Module):
    def __init__(self):
        super(EncoderDiff_MS, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        self.stage1 = EnConvDiffBlock(6)
        self.stage2 = EnConvDiffBlock(32)
        self.stage3 = EnConvDiffBlock(64)
        self.stage4 = EnConvDiffBlock(128)
        self.stage5 = EnConvDiffBlock(256)

    def forward(self,x,x1,x2,x3,x4):
        x1diff = self.stage1(x)
        x2diff = self.stage2((self.pool(x1diff)*self.pool(x1)))#32--->64
        x3diff = self.stage3((self.pool(x2diff)*self.pool(x2)))#64--->128
        x4diff = self.stage4((self.pool(x3diff)*self.pool(x3)))#128--->256
        x5diff = self.stage5((self.pool(x4diff)*self.pool(x4)))#256--->512

        return x1diff,x2diff,x3diff,x4diff,x5diff

#difference encoder with CS
class EncoderDiff_CS(nn.Module):
    def __init__(self):
        super(EncoderDiff_CS, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        self.stage1 = EnConvDiffBlock(6)
        self.stage2 = EnConvDiffBlock(32)
        self.stage3 = EnConvDiffBlock(64)
        self.stage4 = EnConvDiffBlock(128)
        self.stage5 = EnConvDiffBlock(256)

        self.reduce1 = Conv1x1(32*2,32)
        self.reduce2 = Conv1x1(64*2,64)
        self.reduce3 = Conv1x1(128*2,128)
        self.reduce4 = Conv1x1(256*2,256)

    def forward(self,x,x1,x2,x3,x4):
        x1diff = self.stage1(x)
        x2diff = self.stage2( self.reduce1(torch.cat([self.pool(x1diff),self.pool(x1)],dim=1)))#32--->64
        x3diff = self.stage3(self.reduce2(torch.cat([self.pool(x2diff),self.pool(x2)],dim=1)))#64--->128
        x4diff = self.stage4(self.reduce3(torch.cat([self.pool(x3diff),self.pool(x3)],dim=1)))#128--->256
        x5diff = self.stage5(self.reduce4(torch.cat([self.pool(x4diff),self.pool(x4)],dim=1)))#256--->512

        return x1diff,x2diff,x3diff,x4diff,x5diff

#difference encoder without SM
#CI
class EncoderDiff_Nsm(nn.Module):
    def __init__(self):
        super(EncoderDiff_Nsm, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)

        self.stage1 = nn.Sequential(ConvBNR(6, 32), ConvBNR(32, 32))
        self.stage2 = EnConvDiffBlock(32)
        self.stage3 = EnConvDiffBlock(64)
        self.stage4 = EnConvDiffBlock(128)
        self.stage5 = EnConvDiffBlock(256)

    def forward(self,x):
        x1diff = self.stage1(x)
        x2diff = self.stage2(self.pool(x1diff)) #32--->64
        x3diff = self.stage3(self.pool(x2diff)) #64--->128
        x4diff = self.stage4(self.pool(x3diff)) #128--->256
        x5diff = self.stage5(self.pool(x4diff)) #256--->512

        return x1diff,x2diff,x3diff,x4diff,x5diff
#DI
class EncoderDiff_Nsm_DI(nn.Module):
    def __init__(self):
        super(EncoderDiff_Nsm_DI, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        self.stage1 = nn.Sequential(ConvBNR(3,32),ConvBNR(32,32))
        self.stage2 = EnConvDiffBlock(32)
        self.stage3 = EnConvDiffBlock(64)
        self.stage4 = EnConvDiffBlock(128)
        self.stage5 = EnConvDiffBlock(256)

    def forward(self,x):
        x1diff = self.stage1(x)
        x2diff = self.stage2(self.pool(x1diff)) #32--->64
        x3diff = self.stage3(self.pool(x2diff)) #64--->128
        x4diff = self.stage4(self.pool(x3diff)) #128--->256
        x5diff = self.stage5(self.pool(x4diff)) #256--->512

        return x1diff,x2diff,x3diff,x4diff,x5diff
#DI+CI
class EncoderDiff_Nsm_DICI(nn.Module):
    def __init__(self):
        super(EncoderDiff_Nsm_DICI, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        self.stage1 = nn.Sequential(ConvBNR(9,32),ConvBNR(32,32))

        self.stage2 = EnConvDiffBlock(32)
        self.stage3 = EnConvDiffBlock(64)
        self.stage4 = EnConvDiffBlock(128)
        self.stage5 = EnConvDiffBlock(256)

    def forward(self,x):
        x1diff = self.stage1(x)
        x2diff = self.stage2(self.pool(x1diff)) #32--->64
        x3diff = self.stage3(self.pool(x2diff)) #64--->128
        x4diff = self.stage4(self.pool(x3diff)) #128--->256
        x5diff = self.stage5(self.pool(x4diff)) #256--->512

        return x1diff,x2diff,x3diff,x4diff,x5diff


###Encoder-Decoder with parameters#####
class EncoderP(nn.Module):
    def __init__(self,wide):
        super(EncoderP, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        self.stage1 = EnConvBlock(3)
        self.stage2 = EnConvBlock(wide)
        self.stage3 = EnConvBlock(wide*2)
        self.stage4 = EnConvBlock(wide*4)
        self.stage5 = EnConvBlock(wide*8)

    def forward(self,x):
        x1 = self.stage1(x)
        x2 = self.stage2(self.pool(x1))
        x3 = self.stage3(self.pool(x2))
        x4 = self.stage4(self.pool(x3))
        x5 = self.stage5(self.pool(x4))

        return x1,x2,x3,x4,x5
class EncoderDiffP(nn.Module):
    def __init__(self,wide):
        super(EncoderDiffP, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        self.stage1 = EnConvDiffBlock(6)
        self.stage2 = EnConvDiffBlock(wide)
        self.stage3 = EnConvDiffBlock(wide*2)
        self.stage4 = EnConvDiffBlock(wide*4)
        self.stage5 = EnConvDiffBlock(wide*8)

    def forward(self,x,x1,x2,x3,x4):
        x1diff = self.stage1(x)
        x2diff = self.stage2((self.pool(x1diff)+self.pool(x1)))#32--->64
        x3diff = self.stage3((self.pool(x2diff)+self.pool(x2)))#64--->128
        x4diff = self.stage4((self.pool(x3diff)+self.pool(x3)))#128--->256
        x5diff = self.stage5((self.pool(x4diff)+self.pool(x4)))#256--->512

        return x1diff,x2diff,x3diff,x4diff,x5diff