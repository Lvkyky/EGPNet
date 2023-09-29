from thop import profile
from model.network import Fuse_Unet_Edge
import torch


device = torch.device("cuda:0")
model = Fuse_Unet_Edge()
model = model.to(torch.device(device))

input1 = torch.randn(1, 3, 256, 256).to(device)
input2 = torch.randn(1, 3, 256, 256).to(device)
flops, params = profile(model, inputs=(input1,input2))

print("params=", str(params/1e6)+'{}'.format("M"))
print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))