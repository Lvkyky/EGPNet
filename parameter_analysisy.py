
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from model.network import Fuse_Unet_Edge
import torch


model = Fuse_Unet_Edge()
#参数量分析
print(parameter_count_table(model))
#计算量分析
#创建输入网络的tensor
device = torch.device('cuda:0')
model = model.to(device)
tensor1 = (torch.rand(1, 3, 256, 256).to(device),torch.rand(1, 3, 256, 256).to(device))
flops = FlopCountAnalysis(model, tensor1)
print("FLOPs: ", flops.total())