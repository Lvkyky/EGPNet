
from model.network import Fuse_Unet_Edge
import torch
import trainer

#work main directory
work_Dir = '/home/u202132803175/jupyterlab/EGPNet/upload'

#sub-working directory.
data_dir =  '/home/u202132803175/jupyterlab/Data-Set/CDD'

#Path of the dataset. The dataset folder must be named LEVIR, SYSU or CDD.
project_dir = work_Dir + '/EGPNET_CDD'

device = torch.device('cuda:0')
batchsize = 1
model = Fuse_Unet_Edge()

if __name__ == '__main__':
    tester = trainer.CDTester(project_dir,data_dir,batchsize,model,device,edge=True,visual=True)
    tester.test()

