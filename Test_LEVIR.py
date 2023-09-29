
from model.network import Fuse_Unet_Edge
import torch
import trainer

#main working directory
work_Dir = '/home/u202132803175/jupyterlab/EGPNet/upload'

#sub-working directory
project_dir = work_Dir + '/EGPNET_LEVIR'

#Path of the dataset. The dataset folder must be named LEVIR, SYSU or CDD.
data_dir =  '/home/u202132803175/jupyterlab/Data-Set/LEVIR'


device = torch.device('cuda:0')
batchsize = 1
model = Fuse_Unet_Edge()


if __name__ == '__main__':
    tester = trainer.CDTester(project_dir,data_dir,batchsize,model,device,edge=True,visual =True)
    tester.test()
