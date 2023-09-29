
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.edge_guidance import EAM,EAMP, EFM

from model.utils import Encoder,Decoder,\
    EncoderDiff_Nsm,EncoderDiff_Nsm_DICI,EncoderDiff_Nsm_DI,\
    EncoderDiff,EncoderDiff_MS,EncoderDiff_CS, \
    EncoderP,EncoderDiffP,FuseBlock0,FuseBlock1



##############Ablation of different input forms############
#CI
class Diff_Unet(nn.Module):
    def __init__(self):
        super(Diff_Unet, self).__init__()
        self.encoderdiff = EncoderDiff_Nsm()

        self.decoder = Decoder(32)
        self.classifier = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=(1, 1))
        self.classifier2 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=(1, 1))
        self.classifier3 = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=(1, 1))
        self.classifier4 = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=(1, 1))
        self.classifier5 = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=(1, 1))

        self.siam = 1

    def forward(self,x1,x2):
        self.siam = 1
        x = torch.cat([x1,x2],dim=1)
        x1,x2,x3,x4,x5 = self.encoderdiff(x)
        x = self.decoder(x1,x2,x3,x4,x5)

        result = self.classifier(x)
        result2 = F.interpolate(self.classifier2(x2), scale_factor=2, mode='bilinear')
        result3 = F.interpolate(self.classifier3(x3), scale_factor=4, mode='bilinear')
        result4 = F.interpolate(self.classifier4(x4), scale_factor=8, mode='bilinear')
        result5 = F.interpolate(self.classifier5(x5), scale_factor=16, mode='bilinear')
        return result,result2,result3,result4,result5
#DI
class Diff_Unet_DI(nn.Module):
    def __init__(self):
        super(Diff_Unet_DI, self).__init__()
        self.encoderdiff = EncoderDiff_Nsm_DI()

        self.decoder = Decoder(32)
        self.classifier = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=(1, 1))
        self.classifier2 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=(1, 1))
        self.classifier3 = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=(1, 1))
        self.classifier4 = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=(1, 1))
        self.classifier5 = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=(1, 1))

        self.siam = 1

    def forward(self,x1,x2):
        self.siam = 1
        x = torch.abs(x1-x2)
        x1,x2,x3,x4,x5 = self.encoderdiff(x)
        x = self.decoder(x1,x2,x3,x4,x5)

        result = self.classifier(x)
        result2 = F.interpolate(self.classifier2(x2), scale_factor=2, mode='bilinear')
        result3 = F.interpolate(self.classifier3(x3), scale_factor=4, mode='bilinear')
        result4 = F.interpolate(self.classifier4(x4), scale_factor=8, mode='bilinear')
        result5 = F.interpolate(self.classifier5(x5), scale_factor=16, mode='bilinear')
        return result,result2,result3,result4,result5
#DICI
class Diff_Unet_DICI(nn.Module):
    def __init__(self):
        super(Diff_Unet_DICI, self).__init__()
        self.encoder = Encoder()
        self.encoderdiff = EncoderDiff_Nsm_DICI()

        self.decoder = Decoder(32)
        self.classifier = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=(1, 1))
        self.classifier2 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=(1, 1))
        self.classifier3 = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=(1, 1))
        self.classifier4 = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=(1, 1))
        self.classifier5 = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=(1, 1))
        self.siam = 1

    def forward(self,x1,x2):
        self.siam = 1
        di = torch.abs(x1-x2)
        ci = torch.cat([x1,x2],dim=1)
        dici = torch.cat([di,ci],dim=1)

        x1,x2,x3,x4,x5 = self.encoderdiff(dici)
        x = self.decoder(x1,x2,x3,x4,x5)
        result = self.classifier(x)
        result2 = F.interpolate(self.classifier2(x2), scale_factor=2, mode='bilinear')
        result3 = F.interpolate(self.classifier3(x3), scale_factor=4, mode='bilinear')
        result4 = F.interpolate(self.classifier4(x4), scale_factor=8, mode='bilinear')
        result5 = F.interpolate(self.classifier5(x5), scale_factor=16, mode='bilinear')
        return result,result2,result3,result4,result5
##############输入形式消融############


#############Ablation of parallel encoding framework############
#bi-temporal
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.encoder = Encoder()
        self.fuse1 = FuseBlock0(32)
        self.fuse2 = FuseBlock0(64)
        self.fuse3 = FuseBlock0(128)
        self.fuse4 = FuseBlock0(256)
        self.fuse5 = FuseBlock0(512)

        self.decoder = Decoder(32)
        self.classifier = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=(1, 1))
        self.classifier2 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=(1, 1))
        self.classifier3 = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=(1, 1))
        self.classifier4 = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=(1, 1))
        self.classifier5 = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=(1, 1))

    def forward(self, x1, x2):
        x11, x12, x13, x14, x15 = self.encoder(x1)
        x21, x22, x23, x24, x25 = self.encoder(x2)

        x5 = torch.cat([x15, x25], dim=1)
        x5 = self.fuse5(x5)

        x4 = torch.cat([x14,x24], dim=1)
        x4 = self.fuse4(x4)

        x3 = torch.cat([x13, x23], dim=1)
        x3 = self.fuse3(x3)

        x2= torch.cat([x12, x22], dim=1)
        x2 = self.fuse2(x2)

        x1 = torch.cat([x11, x21], dim=1)
        x1 = self.fuse1(x1)

        x = self.decoder(x1, x2, x3, x4, x5)
        result = self.classifier(x)
        result2 = F.interpolate(self.classifier2(x2), scale_factor=2, mode='bilinear')
        result3 = F.interpolate(self.classifier3(x3), scale_factor=4, mode='bilinear')
        result4 = F.interpolate(self.classifier4(x4), scale_factor=8, mode='bilinear')
        result5 = F.interpolate(self.classifier5(x5), scale_factor=16, mode='bilinear')
        return result,result2,result3,result4,result5
#_AS
class Fuse_Unet(nn.Module):
    def __init__(self):
        super(Fuse_Unet, self).__init__()
        self.encoder = Encoder()
        self.encoderdiff = EncoderDiff()

        self.fuse1 = FuseBlock1(32)
        self.fuse2 = FuseBlock1(64)
        self.fuse3 = FuseBlock1(128)
        self.fuse4 = FuseBlock1(256)
        self.fuse5 = FuseBlock1(512)

        self.decoder = Decoder(32)

        self.classifier = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=(1, 1))
        self.classifier2 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=(1, 1))
        self.classifier3 = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=(1, 1))
        self.classifier4 = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=(1, 1))
        self.classifier5 = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=(1, 1))
        self.siam=1

    def forward(self,x1,x2):
        self.siam = 1
        x11, x12, x13, x14, x15 = self.encoder(x1)
        self.siam = 2
        x21, x22, x23, x24, x25 = self.encoder(x2)

        x = torch.cat([x1,x2],dim=1)
        x4diff = torch.abs(x14-x24)
        x3diff = torch.abs(x13-x23)
        x2diff = torch.abs(x12-x22)
        x1diff = torch.abs(x11-x21)

        x1diff,x2diff,x3diff,x4diff,x5diff = self.encoderdiff(x,x1diff,x2diff,x3diff,x4diff)
        x5 = torch.cat([x15, x25, x5diff], dim=1)
        x5 = self.fuse5(x5)


        x4 = torch.cat([x14, x24, x4diff], dim=1)
        x4 = self.fuse4(x4)


        x3 = torch.cat([x13, x23, x3diff], dim=1)
        x3 = self.fuse3(x3)


        x2 = torch.cat([x12, x22, x2diff], dim=1)
        x2 = self.fuse2(x2)


        x1 = torch.cat([x11, x21, x1diff], dim=1)
        x1 = self.fuse1(x1)


        de1 = self.decoder(x1,x2,x3,x4,x5)
        result2 =  F.interpolate(self.classifier2(x2) , scale_factor=2, mode='bilinear')
        result3 =  F.interpolate(self.classifier3(x3) , scale_factor=4, mode='bilinear')
        result4 =  F.interpolate(self.classifier4(x4) , scale_factor=8, mode='bilinear')
        result5 =  F.interpolate(self.classifier5(x5) , scale_factor=16, mode='bilinear')
        result = self.classifier(de1)
        return result, result2, result3, result4, result5

#############Ablation for different SM strategies############
#_NS
class Fuse_Unet_NS(nn.Module):
    def __init__(self):
        super(Fuse_Unet_NS, self).__init__()
        self.encoder = Encoder()
        self.encoderdiff = EncoderDiff_Nsm()

        self.fuse1 = FuseBlock1(32)
        self.fuse2 = FuseBlock1(64)
        self.fuse3 = FuseBlock1(128)
        self.fuse4 = FuseBlock1(256)
        self.fuse5 = FuseBlock1(512)

        self.decoder = Decoder(32)

        self.classifier = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=(1, 1))
        self.classifier2 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=(1, 1))
        self.classifier3 = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=(1, 1))
        self.classifier4 = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=(1, 1))
        self.classifier5 = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=(1, 1))
        self.siam=1

    def forward(self,x1,x2):
        self.siam = 1
        x11, x12, x13, x14, x15 = self.encoder(x1)
        self.siam = 2
        x21, x22, x23, x24, x25 = self.encoder(x2)

        x = torch.cat([x1,x2],dim=1)
        x1diff,x2diff,x3diff,x4diff,x5diff = self.encoderdiff(x)

        x5 = torch.cat([x15, x25, x5diff], dim=1)
        x5 = self.fuse5(x5)

        x4 = torch.cat([x14, x24, x4diff], dim=1)
        x4 = self.fuse4(x4)

        x3 = torch.cat([x13, x23, x3diff], dim=1)
        x3 = self.fuse3(x3)

        x2 = torch.cat([x12, x22, x2diff], dim=1)
        x2 = self.fuse2(x2)

        x1 = torch.cat([x11, x21, x1diff], dim=1)
        x1 = self.fuse1(x1)

        de1 = self.decoder(x1,x2,x3,x4,x5)
        result2 =  F.interpolate(self.classifier2(x2) , scale_factor=2, mode='bilinear')
        result3 =  F.interpolate(self.classifier3(x3) , scale_factor=4, mode='bilinear')
        result4 =  F.interpolate(self.classifier4(x4) , scale_factor=8, mode='bilinear')
        result5 =  F.interpolate(self.classifier5(x5) , scale_factor=16, mode='bilinear')
        result = self.classifier(de1)
        return result, result2, result3, result4, result5
#_AS
class Fuse_Unet_MS(nn.Module):
    def __init__(self):
        super(Fuse_Unet_MS, self).__init__()
        self.encoder = Encoder()
        self.encoderdiff = EncoderDiff_MS()

        self.fuse1 = FuseBlock1(32)
        self.fuse2 = FuseBlock1(64)
        self.fuse3 = FuseBlock1(128)
        self.fuse4 = FuseBlock1(256)
        self.fuse5 = FuseBlock1(512)

        self.decoder = Decoder(32)

        self.classifier = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=(1, 1))
        self.classifier2 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=(1, 1))
        self.classifier3 = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=(1, 1))
        self.classifier4 = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=(1, 1))
        self.classifier5 = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=(1, 1))
        self.siam=1

    def forward(self,x1,x2):
        self.siam = 1
        x11, x12, x13, x14, x15 = self.encoder(x1)
        self.siam = 2
        x21, x22, x23, x24, x25 = self.encoder(x2)

        x = torch.cat([x1,x2],dim=1)
        x4diff = torch.abs(x14-x24)
        x3diff = torch.abs(x13-x23)
        x2diff = torch.abs(x12-x22)
        x1diff = torch.abs(x11-x21)

        x1diff,x2diff,x3diff,x4diff,x5diff = self.encoderdiff(x,x1diff,x2diff,x3diff,x4diff)
        x5 = torch.cat([x15, x25, x5diff], dim=1)
        x5 = self.fuse5(x5)


        x4 = torch.cat([x14, x24, x4diff], dim=1)
        x4 = self.fuse4(x4)


        x3 = torch.cat([x13, x23, x3diff], dim=1)
        x3 = self.fuse3(x3)


        x2 = torch.cat([x12, x22, x2diff], dim=1)
        x2 = self.fuse2(x2)


        x1 = torch.cat([x11, x21, x1diff], dim=1)
        x1 = self.fuse1(x1)


        de1 = self.decoder(x1,x2,x3,x4,x5)
        result2 =  F.interpolate(self.classifier2(x2) , scale_factor=2, mode='bilinear')
        result3 =  F.interpolate(self.classifier3(x3) , scale_factor=4, mode='bilinear')
        result4 =  F.interpolate(self.classifier4(x4) , scale_factor=8, mode='bilinear')
        result5 =  F.interpolate(self.classifier5(x5) , scale_factor=16, mode='bilinear')
        result = self.classifier(de1)
        return result, result2, result3, result4, result5
#_CS
class Fuse_Unet_CS(nn.Module):
    def __init__(self):
        super(Fuse_Unet_CS, self).__init__()
        self.encoder = Encoder()
        self.encoderdiff = EncoderDiff_CS()

        self.fuse1 = FuseBlock1(32)
        self.fuse2 = FuseBlock1(64)
        self.fuse3 = FuseBlock1(128)
        self.fuse4 = FuseBlock1(256)
        self.fuse5 = FuseBlock1(512)

        self.decoder = Decoder(32)

        self.classifier = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=(1, 1))
        self.classifier2 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=(1, 1))
        self.classifier3 = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=(1, 1))
        self.classifier4 = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=(1, 1))
        self.classifier5 = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=(1, 1))
        self.siam=1

    def forward(self,x1,x2):
        self.siam = 1
        x11, x12, x13, x14, x15 = self.encoder(x1)
        self.siam = 2
        x21, x22, x23, x24, x25 = self.encoder(x2)

        x = torch.cat([x1,x2],dim=1)
        x4diff = torch.abs(x14-x24)
        x3diff = torch.abs(x13-x23)
        x2diff = torch.abs(x12-x22)
        x1diff = torch.abs(x11-x21)

        x1diff,x2diff,x3diff,x4diff,x5diff = self.encoderdiff(x,x1diff,x2diff,x3diff,x4diff)
        x5 = torch.cat([x15, x25, x5diff], dim=1)
        x5 = self.fuse5(x5)


        x4 = torch.cat([x14, x24, x4diff], dim=1)
        x4 = self.fuse4(x4)


        x3 = torch.cat([x13, x23, x3diff], dim=1)
        x3 = self.fuse3(x3)


        x2 = torch.cat([x12, x22, x2diff], dim=1)
        x2 = self.fuse2(x2)


        x1 = torch.cat([x11, x21, x1diff], dim=1)
        x1 = self.fuse1(x1)


        de1 = self.decoder(x1,x2,x3,x4,x5)
        result2 =  F.interpolate(self.classifier2(x2) , scale_factor=2, mode='bilinear')
        result3 =  F.interpolate(self.classifier3(x3) , scale_factor=4, mode='bilinear')
        result4 =  F.interpolate(self.classifier4(x4) , scale_factor=8, mode='bilinear')
        result5 =  F.interpolate(self.classifier5(x5) , scale_factor=16, mode='bilinear')
        result = self.classifier(de1)
        return result, result2, result3, result4, result5

##########Ablation of model copacity#############
class Fuse_Unet_Edge(nn.Module):
    def __init__(self):
        super(Fuse_Unet_Edge, self).__init__()
        self.encoder = Encoder()
        self.encoderdiff = EncoderDiff()

        self.fuse1 = FuseBlock1(32)
        self.fuse2 = FuseBlock1(64)
        self.fuse3 = FuseBlock1(128)
        self.fuse4 = FuseBlock1(256)
        self.fuse5 = FuseBlock1(512)

        self.eam = EAM()
        self.efm2 = EFM(64)
        self.efm3 = EFM(128)
        self.efm4 = EFM(256)
        self.efm5 = EFM(512)

        self.decoder = Decoder(32)
        self.classifier = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=(1, 1))

        self.classifier2 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=(1, 1))
        self.classifier3 = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=(1, 1))
        self.classifier4 = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=(1, 1))
        self.classifier5 = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=(1, 1))
        self.siam=1

    def forward(self,x1,x2):
        self.siam = 1
        x11, x12, x13, x14, x15 = self.encoder(x1)
        self.siam = 2
        x21, x22, x23, x24, x25 = self.encoder(x2)

        x = torch.cat([x1,x2],dim=1)
        x4diff = torch.abs(x14-x24)
        x3diff = torch.abs(x13-x23)
        x2diff = torch.abs(x12-x22)
        x1diff = torch.abs(x11-x21)

        x1diff,x2diff,x3diff,x4diff,x5diff = self.encoderdiff(x,x1diff,x2diff,x3diff,x4diff)
        x5 = torch.cat([x15, x25, x5diff], dim=1)
        x5 = self.fuse5(x5)

        x4 = torch.cat([x14, x24, x4diff], dim=1)
        x4 = self.fuse4(x4)

        x3 = torch.cat([x13, x23, x3diff], dim=1)
        x3 = self.fuse3(x3)

        x2 = torch.cat([x12, x22, x2diff], dim=1)
        x2 = self.fuse2(x2)

        x1 = torch.cat([x11, x21, x1diff], dim=1)
        x1 = self.fuse1(x1)

        #边缘注意力
        edge_att = self.eam(x2,x5)
        edge_att = torch.sigmoid(edge_att)
        edge = F.interpolate(edge_att,scale_factor=2, mode='bilinear')
        x2 = self.efm2(x2,edge_att)
        x3 = self.efm3(x3,edge_att)
        x4 = self.efm4(x4,edge_att)
        x5 = self.efm5(x5,edge_att)

        x  = self.decoder(x1,x2,x3,x4,x5)

        result2 =  F.interpolate(self.classifier2(x2) , scale_factor=2, mode='bilinear')
        result3 =  F.interpolate(self.classifier3(x3) , scale_factor=4, mode='bilinear')
        result4 =  F.interpolate(self.classifier4(x4) , scale_factor=8, mode='bilinear')
        result5 =  F.interpolate(self.classifier5(x5) , scale_factor=16, mode='bilinear')
        result = self.classifier(x)
        return result, result2, result3, result4, result5, edge
class Fuse_Unet_Edge_40(nn.Module):
    def __init__(self):
        super(Fuse_Unet_Edge_40, self).__init__()
        self.encoder = EncoderP(40)
        self.encoderdiff = EncoderDiffP(40)

        self.fuse1 = FuseBlock1(40)
        self.fuse2 = FuseBlock1(80)
        self.fuse3 = FuseBlock1(160)
        self.fuse4 = FuseBlock1(320)
        self.fuse5 = FuseBlock1(640)

        self.eam = EAMP(640)
        self.efm2 = EFM(80)
        self.efm3 = EFM(160)
        self.efm4 = EFM(320)
        self.efm5 = EFM(640)


        self.decoder = Decoder(40)
        self.classifier = nn.Conv2d(in_channels=40, out_channels=2, kernel_size=(1, 1))
        self.classifier2 = nn.Conv2d(in_channels=80, out_channels=2, kernel_size=(1, 1))
        self.classifier3 = nn.Conv2d(in_channels=160, out_channels=2, kernel_size=(1, 1))
        self.classifier4 = nn.Conv2d(in_channels=320, out_channels=2, kernel_size=(1, 1))
        self.classifier5 = nn.Conv2d(in_channels=640, out_channels=2, kernel_size=(1, 1))
        self.siam=1

    def forward(self,x1,x2):
        self.siam = 1
        x11, x12, x13, x14, x15 = self.encoder(x1)
        self.siam = 2
        x21, x22, x23, x24, x25 = self.encoder(x2)

        x = torch.cat([x1,x2],dim=1)
        x4diff = torch.abs(x14-x24)
        x3diff = torch.abs(x13-x23)
        x2diff = torch.abs(x12-x22)
        x1diff = torch.abs(x11-x21)

        x1diff,x2diff,x3diff,x4diff,x5diff = self.encoderdiff(x,x1diff,x2diff,x3diff,x4diff)
        x5 = torch.cat([x15, x25, x5diff], dim=1)
        x5 = self.fuse5(x5)


        x4 = torch.cat([x14, x24, x4diff], dim=1)
        x4 = self.fuse4(x4)


        x3 = torch.cat([x13, x23, x3diff], dim=1)
        x3 = self.fuse3(x3)


        x2 = torch.cat([x12, x22, x2diff], dim=1)
        x2 = self.fuse2(x2)


        x1 = torch.cat([x11, x21, x1diff], dim=1)
        x1 = self.fuse1(x1)

        #边缘注意力
        edge_att = self.eam(x2,x5)
        edge_att = torch.sigmoid(edge_att)
        edge = F.interpolate(edge_att,scale_factor=2, mode='bilinear')

        x2 = self.efm2(x2,edge_att)
        x3 = self.efm3(x3,edge_att)
        x4 = self.efm4(x4,edge_att)
        x5 = self.efm5(x5,edge_att)

        x  = self.decoder(x1,x2,x3,x4,x5)

        result2 =  F.interpolate(self.classifier2(x2) , scale_factor=2, mode='bilinear')
        result3 =  F.interpolate(self.classifier3(x3) , scale_factor=4, mode='bilinear')
        result4 =  F.interpolate(self.classifier4(x4) , scale_factor=8, mode='bilinear')
        result5 =  F.interpolate(self.classifier5(x5) , scale_factor=16,mode='bilinear')
        result = self.classifier(x)
        return result, result2, result3, result4, result5,edge
class Fuse_Unet_Edge_24(nn.Module):
    def __init__(self):
        super(Fuse_Unet_Edge_24, self).__init__()
        self.encoder = EncoderP(24)
        self.encoderdiff = EncoderDiffP(24)

        self.fuse1 = FuseBlock1(24)
        self.fuse2 = FuseBlock1(48)
        self.fuse3 = FuseBlock1(96)
        self.fuse4 = FuseBlock1(192)
        self.fuse5 = FuseBlock1(384)

        self.eam = EAMP(384)
        self.efm2 = EFM(48)
        self.efm3 = EFM(96)
        self.efm4 = EFM(192)
        self.efm5 = EFM(384)

        self.decoder = Decoder(24)
        self.classifier = nn.Conv2d(in_channels=24, out_channels=2, kernel_size=(1, 1))
        self.classifier2 = nn.Conv2d(in_channels=48, out_channels=2, kernel_size=(1, 1))
        self.classifier3 = nn.Conv2d(in_channels=96, out_channels=2, kernel_size=(1, 1))
        self.classifier4 = nn.Conv2d(in_channels=192, out_channels=2, kernel_size=(1, 1))
        self.classifier5 = nn.Conv2d(in_channels=384, out_channels=2, kernel_size=(1, 1))
        self.siam = 1

    def forward(self,x1,x2):
        self.siam = 1
        x11, x12, x13, x14, x15 = self.encoder(x1)
        self.siam = 2
        x21, x22, x23, x24, x25 = self.encoder(x2)

        x = torch.cat([x1,x2],dim=1)
        x4diff = torch.abs(x14-x24)
        x3diff = torch.abs(x13-x23)
        x2diff = torch.abs(x12-x22)
        x1diff = torch.abs(x11-x21)

        x1diff,x2diff,x3diff,x4diff,x5diff = self.encoderdiff(x,x1diff,x2diff,x3diff,x4diff)
        x5 = torch.cat([x15, x25, x5diff], dim=1)
        x5 = self.fuse5(x5)


        x4 = torch.cat([x14, x24, x4diff], dim=1)
        x4 = self.fuse4(x4)

        x3 = torch.cat([x13, x23, x3diff], dim=1)
        x3 = self.fuse3(x3)

        x2 = torch.cat([x12, x22, x2diff], dim=1)
        x2 = self.fuse2(x2)

        x1 = torch.cat([x11, x21, x1diff], dim=1)
        x1 = self.fuse1(x1)

        #边缘注意力
        edge_att = self.eam(x2,x5)
        edge_att = torch.sigmoid(edge_att)
        edge = F.interpolate(edge_att,scale_factor=2, mode='bilinear')

        x2 = self.efm2(x2,edge_att)
        x3 = self.efm3(x3,edge_att)
        x4 = self.efm4(x4,edge_att)
        x5 = self.efm5(x5,edge_att)

        x  = self.decoder(x1,x2,x3,x4,x5)

        result2 =  F.interpolate(self.classifier2(x2), scale_factor=2, mode='bilinear')
        result3 =  F.interpolate(self.classifier3(x3), scale_factor=4, mode='bilinear')
        result4 =  F.interpolate(self.classifier4(x4), scale_factor=8, mode='bilinear')
        result5 =  F.interpolate(self.classifier5(x5), scale_factor=16, mode='bilinear')
        result = self.classifier(x)
        return result, result2, result3, result4, result5,edge
class Fuse_Unet_Edge_16(nn.Module):
    def __init__(self):
        super(Fuse_Unet_Edge_16, self).__init__()
        self.encoder = EncoderP(16)
        self.encoderdiff = EncoderDiffP(16)

        self.fuse1 = FuseBlock1(16)
        self.fuse2 = FuseBlock1(32)
        self.fuse3 = FuseBlock1(64)
        self.fuse4 = FuseBlock1(128)
        self.fuse5 = FuseBlock1(256)

        self.eam = EAMP(256)
        self.efm2 = EFM(32)
        self.efm3 = EFM(64)
        self.efm4 = EFM(128)
        self.efm5 = EFM(256)


        self.decoder = Decoder(16)
        self.classifier = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=(1, 1))
        self.classifier2 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=(1, 1))
        self.classifier3 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=(1, 1))
        self.classifier4 = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=(1, 1))
        self.classifier5 = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=(1, 1))
        self.siam=1

    def forward(self,x1,x2):
        self.siam = 1
        x11, x12, x13, x14, x15 = self.encoder(x1)
        self.siam = 2
        x21, x22, x23, x24, x25 = self.encoder(x2)

        x = torch.cat([x1,x2],dim=1)
        x4diff = torch.abs(x14-x24)
        x3diff = torch.abs(x13-x23)
        x2diff = torch.abs(x12-x22)
        x1diff = torch.abs(x11-x21)

        x1diff,x2diff,x3diff,x4diff,x5diff = self.encoderdiff(x,x1diff,x2diff,x3diff,x4diff)
        x5 = torch.cat([x15, x25, x5diff], dim=1)
        x5 = self.fuse5(x5)


        x4 = torch.cat([x14, x24, x4diff], dim=1)
        x4 = self.fuse4(x4)


        x3 = torch.cat([x13, x23, x3diff], dim=1)
        x3 = self.fuse3(x3)


        x2 = torch.cat([x12, x22, x2diff], dim=1)
        x2 = self.fuse2(x2)


        x1 = torch.cat([x11, x21, x1diff], dim=1)
        x1 = self.fuse1(x1)

        #边缘注意力
        edge_att = self.eam(x2,x5)
        edge_att = torch.sigmoid(edge_att)
        edge = F.interpolate(edge_att,scale_factor=2, mode='bilinear')

        x2 = self.efm2(x2,edge_att)
        x3 = self.efm3(x3,edge_att)
        x4 = self.efm4(x4,edge_att)
        x5 = self.efm5(x5,edge_att)

        x  = self.decoder(x1,x2,x3,x4,x5)

        result2 =  F.interpolate(self.classifier2(x2) , scale_factor=2, mode='bilinear')
        result3 =  F.interpolate(self.classifier3(x3) , scale_factor=4, mode='bilinear')
        result4 =  F.interpolate(self.classifier4(x4) , scale_factor=8, mode='bilinear')
        result5 =  F.interpolate(self.classifier5(x5) , scale_factor=16, mode='bilinear')
        result = self.classifier(x)
        return result, result2, result3, result4, result5,edge
class Fuse_Unet_Edge_8(nn.Module):
    def __init__(self):
        super(Fuse_Unet_Edge_8, self).__init__()
        self.encoder = EncoderP(8)
        self.encoderdiff = EncoderDiffP(8)

        self.fuse1 = FuseBlock1(8)
        self.fuse2 = FuseBlock1(16)
        self.fuse3 = FuseBlock1(32)
        self.fuse4 = FuseBlock1(64)
        self.fuse5 = FuseBlock1(128)

        self.eam = EAMP(128)
        self.efm2 = EFM(16)
        self.efm3 = EFM(32)
        self.efm4 = EFM(64)
        self.efm5 = EFM(128)

        self.decoder = Decoder(8)
        self.classifier = nn.Conv2d(in_channels=8, out_channels=2, kernel_size=(1, 1))
        self.classifier2 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=(1, 1))
        self.classifier3 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=(1, 1))
        self.classifier4 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=(1, 1))
        self.classifier5 = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=(1, 1))
        self.siam=1

    def forward(self,x1,x2):
        self.siam = 1
        x11, x12, x13, x14, x15 = self.encoder(x1)
        self.siam = 2
        x21, x22, x23, x24, x25 = self.encoder(x2)

        x = torch.cat([x1,x2],dim=1)
        x4diff = torch.abs(x14-x24)
        x3diff = torch.abs(x13-x23)
        x2diff = torch.abs(x12-x22)
        x1diff = torch.abs(x11-x21)

        x1diff,x2diff,x3diff,x4diff,x5diff = self.encoderdiff(x,x1diff,x2diff,x3diff,x4diff)
        x5 = torch.cat([x15, x25, x5diff], dim=1)
        x5 = self.fuse5(x5)


        x4 = torch.cat([x14, x24, x4diff], dim=1)
        x4 = self.fuse4(x4)


        x3 = torch.cat([x13, x23, x3diff], dim=1)
        x3 = self.fuse3(x3)


        x2 = torch.cat([x12, x22, x2diff], dim=1)
        x2 = self.fuse2(x2)


        x1 = torch.cat([x11, x21, x1diff], dim=1)
        x1 = self.fuse1(x1)

        #边缘注意力
        edge_att = self.eam(x2,x5)
        edge_att = torch.sigmoid(edge_att)
        edge = F.interpolate(edge_att,scale_factor=2, mode='bilinear')

        x2 = self.efm2(x2,edge_att)
        x3 = self.efm3(x3,edge_att)
        x4 = self.efm4(x4,edge_att)
        x5 = self.efm5(x5,edge_att)

        x  = self.decoder(x1,x2,x3,x4,x5)

        result2 =  F.interpolate(self.classifier2(x2) , scale_factor=2, mode='bilinear')
        result3 =  F.interpolate(self.classifier3(x3) , scale_factor=4, mode='bilinear')
        result4 =  F.interpolate(self.classifier4(x4) , scale_factor=8, mode='bilinear')
        result5 =  F.interpolate(self.classifier5(x5) , scale_factor=16, mode='bilinear')
        result = self.classifier(x)
        return result, result2, result3, result4, result5,edge