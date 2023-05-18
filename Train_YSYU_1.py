from model.network import Fuse_Unet_Edge
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.data_utils_YSYU import LoadDatasetFromFolder
from model.loss import FocalLoss,dice_loss
import trainer
from util.tools import  setup_seed,init_weights, get_scheduler

setup_seed(20)
#work main directory
work_Dir = '/home/lkk/PytorchWork/Unet-Fuse'
#work subdirectory
case = 1
#dataset directory
train_dir1 = '/home/lkk/PytorchWork/Data-Set/YSYU/train/time1'
train_dir2 = '/home/lkk/PytorchWork/Data-Set/YSYU/train/time2'
train_label = '/home/lkk/PytorchWork/Data-Set/YSYU/train/label'
train_edge = '/home/lkk/PytorchWork/Data-Set/YSYU/train/edge'

val_dir1 = '/home/lkk/PytorchWork/Data-Set/YSYU/val/time1'
val_dir2 = '/home/lkk/PytorchWork/Data-Set/YSYU/val/time2'
val_label = '/home/lkk/PytorchWork/Data-Set/YSYU/val/label'
val_edge = '/home/lkk/PytorchWork/Data-Set/YSYU/val/edge'


device = torch.device('cuda:0')
maxEpoch = 100
batchsize = 1
model = Fuse_Unet_Edge()
init_weights(model)
lossFunction  = [FocalLoss(apply_nonlin=nn.Softmax(dim=1)),dice_loss]
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
lr_Scheduler = get_scheduler(optimizer, 'linear', maxEpoch)


# dataset preparation
dataset_train = LoadDatasetFromFolder(train_dir1,train_dir2,train_label,train_edge,edge=True)
dataset_val = LoadDatasetFromFolder(val_dir1, val_dir2,val_label,val_edge,edge=True)
data_loader_train = DataLoader(dataset_train, batch_size=batchsize, shuffle=True)
data_loader_val = DataLoader(dataset_val, batch_size=8, shuffle=False)

train = trainer.CDTrainner_Edge(model = model, loss_function=lossFunction, lr_Scheduler = lr_Scheduler, optimizer = optimizer, max_epoch=maxEpoch, work_dir=work_Dir,
                           data_loader_train=data_loader_train, data_loader_val=data_loader_val, device=device, RESUM=False, refine=False, case = 0)
train.train_model()
