from model.network import Fuse_Unet_Edge
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.data_utils_LEVIR import CDDataset
import trainer
from model.loss import FocalLoss,dice_loss
from util.visualizer import  visulize

def hook(module, input, output):
    if test.siam == 1:
        print('1')
        visulize(output, work_Dir, 5, test.numcount, 'or1', mode='PCA3')
        test.siam = 2
    else:
        print('6666666666666')
        visulize(output, work_Dir, 5, test.numcount, 'or2', mode='PCA3')
        test.siam = 1
def hook1(module, input, output):
    visulize(output,work_Dir,5,test.numcount,'diff',mode='PCA3')
def hook2(module, input, output):
    visulize(output,work_Dir,5,test.numcount,'fuse',mode='PCA3')
def hook3(module, input, output):
    visulize(output,work_Dir,5,test.numcount,'efm',mode='PCA3')


work_Dir = '/home/lkk/PytorchWork/Unet-Fuse'
root_dir = '/home/lkk/PytorchWork/Data-Set/LEVIR-CD-256'
savePath = work_Dir + '/hisNet' + '/Net'
logPath = work_Dir + '/log.txt'
checkpointPath = work_Dir + '/check.pth'
device = torch.device('cuda:0')

model = Fuse_Unet_Edge()
for name, module in model.named_modules():
    print(name)
    if name == 'encoder.stage3':
        module.register_forward_hook(hook)

    if name == 'encoderdiff.stage3':
        module.register_forward_hook(hook1)

    if name == 'fuse3':
        module.register_forward_hook(hook2)

    if name == 'efm3':
        module.register_forward_hook(hook3)


lossFunction  = [FocalLoss(apply_nonlin=nn.Softmax(dim=1)),dice_loss]
dataset_test = CDDataset(root_dir=root_dir, split='test', img_size=256, is_train=False, label_transform='norm',Edge=True)
data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)


bestepoch = 84
test = trainer.CDTester_Edge(model=model, lossFunction=lossFunction, workDir=work_Dir, bestEpoch=bestepoch, dataLoader_test=data_loader_test, device=device, case = 5,find=False,mode=3)
score = test.test()
print(score)

