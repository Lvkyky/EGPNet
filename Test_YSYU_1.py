from model.network import Fuse_Unet_Edge
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.data_utils_YSYU import LoadDatasetFromFolder
from model.loss import FocalLoss,dice_loss
import trainer


work_Dir = '/home/lkk/PytorchWork/YSYU'
train_dir1 = '/home/lkk/PytorchWork/Data-Set/YSYU/train/time1'
train_dir2 = '/home/lkk/PytorchWork/Data-Set/YSYU/train/time2'
train_label = '/home/lkk/PytorchWork/Data-Set/YSYU/train/label'
train_edge = '/home/lkk/PytorchWork/Data-Set/YSYU/train/edge'

val_dir1 = '/home/lkk/PytorchWork/Data-Set/YSYU/val/time1'
val_dir2 = '/home/lkk/PytorchWork/Data-Set/YSYU/val/time2'
val_label = '/home/lkk/PytorchWork/Data-Set/YSYU/val/label'
val_edge = '/home/lkk/PytorchWork/Data-Set/YSYU/val/edge'

test_dir1 = '/home/lkk/PytorchWork/Data-Set/YSYU/test/time1'
test_dir2 = '/home/lkk/PytorchWork/Data-Set/YSYU/test/time2'
test_label = '/home/lkk/PytorchWork/Data-Set/YSYU/test/label'
test_edge = '/home/lkk/PytorchWork/Data-Set/YSYU/test/edge'


device = torch.device('cuda:0')
model = Fuse_Unet_Edge()
lossFunction  = [FocalLoss(apply_nonlin=nn.Softmax(dim=1)),dice_loss]


dataset_test = LoadDatasetFromFolder(test_dir1, test_dir2,test_label,test_edge,ori=True,edge=True)
data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

bestepoch = 85
test = trainer.CDTester_Edge(model=model, lossFunction=lossFunction, workDir=work_Dir, bestEpoch=bestepoch, dataLoader_test=data_loader_test, device=device, case = 0)




