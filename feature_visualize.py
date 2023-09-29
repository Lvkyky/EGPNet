from model.network import Fuse_Unet_Edge
import torch
from util.visualizer import  visulize
import trainer

# main working directory
work_Dir = '/home/u202132803175/jupyterlab/EGPNet/upload'
# sub-working directory
project_dir = work_Dir + '/EGPNET_LEVIR'
# Path of the dataset. The dataset folder must be named LEVIR, SYSU or CDD.
data_dir = '/home/u202132803175/jupyterlab/Data-Set/LEVIR'


device = torch.device('cuda:0')
batchsize = 1
model = Fuse_Unet_Edge()
test = trainer.CDTester(project_dir,data_dir,batchsize,model,device,edge=True,visual =True)


def hook(module, input, output):
    if test.siam == 1:
        visulize(output, project_dir, test.numcount, 'or1', mode='grid')
        test.siam = 2
    else:
        visulize(output, project_dir, test.numcount, 'or2', mode='grid')
        test.siam = 1

def hook1(module, input, output):
    visulize(output,project_dir,test.numcount,'diff',mode='grid')
def hook2(module, input, output):
    visulize(output,project_dir,test.numcount,'fuse',mode='grid')
def hook3(module, input, output):
    visulize(output,project_dir,test.numcount,'efm',mode='grid')


for name, module in model.named_modules():
    if name == 'encoder.stage3':
        module.register_forward_hook(hook)

    if name == 'encoderdiff.stage3':
        module.register_forward_hook(hook1)

    if name == 'fuse3':
        module.register_forward_hook(hook2)

    if name == 'efm3':
        module.register_forward_hook(hook3)
test.test()

