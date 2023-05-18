from model.network import Fuse_Unet_Edge
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.data_utils_LEVIR import CDDataset
import trainer
from model.loss import FocalLoss,dice_loss
from util.tools import  setup_seed,init_weights, get_scheduler

setup_seed(20)
#work main directory
work_Dir = '/home/lkk/PytorchWork/Unet-Fuse'
#work subdirectory
case = 1
#dataset directory
root_dir = '/home/lkk/PytorchWork/Data-Set/LEVIR-CD-256'

device = torch.device('cuda:0')
maxEpoch = 100
batchsize = 1
model = Fuse_Unet_Edge()
init_weights(model)
lossFunction  = [FocalLoss(apply_nonlin=nn.Softmax(dim=1)),dice_loss]
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
lr_Scheduler = get_scheduler(optimizer, 'linear', maxEpoch)


# dataset preparation
dataset_train = CDDataset(root_dir=root_dir, split='train', img_size=256, is_train=False, label_transform='norm',Edge=True)
dataset_val = CDDataset(root_dir=root_dir, split='val', img_size=256, is_train=False, label_transform='norm',Edge=True)
data_loader_train = DataLoader(dataset_train, batch_size=batchsize, shuffle=True, num_workers=4)
data_loader_val = DataLoader(dataset_val, batch_size=8, shuffle=False)


train = trainer.CDTrainner_Edge(model = model, loss_function=lossFunction, lr_Scheduler = lr_Scheduler, optimizer = optimizer, max_epoch=maxEpoch, work_dir=work_Dir,
                           data_loader_train=data_loader_train, data_loader_val=data_loader_val, device=device, RESUM=False, refine=False, case = 0)
train.train_model()
