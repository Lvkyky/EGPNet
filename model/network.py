import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import Encoder,Decoder,EncoderDiff, FuseBlock1,EAM,EFM



#Final EGPNet
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
