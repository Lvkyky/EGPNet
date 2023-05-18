from model.network import Fuse_Unet_Edge
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.data_utils_LEVIR import CDDataset
import trainer
from torch.optim import lr_scheduler
from torch.nn import init
from model.loss import FocalLoss,dice_loss
from util.visualizer import  visulize

work_Dir = '/home/lkk/PytorchWork/Unet-Fuse'
root_dir = '/home/lkk/PytorchWork/Data-Set/LEVIR-CD-256'


device = torch.device('cuda:0')
model = Fuse_Unet_Edge()
lossFunction  = [FocalLoss(apply_nonlin=nn.Softmax(dim=1)),dice_loss]
dataset_test = CDDataset(root_dir=root_dir, split='test', img_size=256, is_train=False, label_transform='norm',Edge=True)
data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

bestepoch = 84
test = trainer.CDTester_Edge(model=model, lossFunction=lossFunction, workDir=work_Dir, bestEpoch=bestepoch, dataLoader_test=data_loader_test, device=device, case = 0)
score = test.test()
print(score)
