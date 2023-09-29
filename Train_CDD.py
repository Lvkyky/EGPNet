#EPGNet-24
from model.network import Fuse_Unet_Edge
import torch
import torch.nn as nn
from model.loss import FocalLoss,dice_loss
import trainer
from util.tools import  setup_seed,init_weights, get_scheduler

setup_seed(20)
work_Dir = '/home/u202132803175/jupyterlab/EGPNet/upload'
data_dir =  '/home/u202132803175/jupyterlab/Data-Set/CDD'
project_dir = work_Dir + '/EGPNET_CDD'


device = torch.device('cuda:0')
maxEpoch = 100
batchsize = 8
model = Fuse_Unet_Edge()
init_weights(model)
lossFunction  = [FocalLoss(apply_nonlin=nn.Softmax(dim=1)),dice_loss]
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
lr_Scheduler = get_scheduler(optimizer, 'linear', maxEpoch)


if __name__ == '__main__':
    train = trainer.CDTrainner(project_dir,data_dir,model,lossFunction, optimizer,lr_Scheduler,maxEpoch,device,batchsize,RESUM=False,edge=True,lamda=0.1)
    train.train_model()
