import os
from util.metrics import RunningMetrics, AverageMeter
import numpy as np
import logging
import torch
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from data.data_utils_LEVIR import LEVIRDataset
from data.data_utils_SYSU import  SYSUDataset
from data.data_utils_CDD import  CDDDataset
from torch.utils.data import DataLoader


#获取数据集
def getdataloder(data_dir,batchsize):

    name  = data_dir.split('/')

    if name[-1] == 'LEVIR':
        dataset_train = LEVIRDataset(data_dir, split='train',edge=True)
        dataset_val = LEVIRDataset(data_dir, split='val',edge=True)
        dataset_test = LEVIRDataset(data_dir, split='test',edge=True)

        return DataLoader(dataset_train, batch_size=batchsize, shuffle=True, num_workers=4),\
        DataLoader(dataset_val, batch_size=batchsize, shuffle=False, num_workers=4),\
        DataLoader(dataset_test, batch_size=batchsize, shuffle=False, num_workers=4)

    elif name[-1] == 'SYSU':
        dataset_train = SYSUDataset(data_dir, split='train',edge=True)
        dataset_val = SYSUDataset(data_dir, split='val',edge=True)
        dataset_test = SYSUDataset(data_dir, split='test',edge=True)
        
        return DataLoader(dataset_train, batch_size=batchsize, shuffle=True, num_workers=4),\
        DataLoader(dataset_val, batch_size=batchsize, shuffle=False, num_workers=4),\
        DataLoader(dataset_test, batch_size=batchsize, shuffle=False, num_workers=4)

    elif name[-1] == 'CDD':
        dataset_train = CDDDataset(data_dir, split='train',edge=True)
        dataset_val = CDDDataset(data_dir, split='val',edge=True)
        dataset_test = CDDDataset(data_dir, split='test',edge=True)
        
        return DataLoader(dataset_train, batch_size=batchsize, shuffle=True, num_workers=4),\
        DataLoader(dataset_val, batch_size=batchsize, shuffle=False, num_workers=4),\
        DataLoader(dataset_test, batch_size=batchsize, shuffle=False, num_workers=4)

'''
曲线显示
'''
class PloterVal:
    def __init__(self, project_dir, split):
        self.OA_list = []
        self.Mean_Iou = []
        self.Pre_list = []
        self.Recall_list = []
        self.F1_list = []
        self.Loss_list = []

        self.x = None
        cure_dir = project_dir + '/' + 'curve'

        if os.path.exists(cure_dir) == False:
            os.mkdir(cure_dir)
        self.save_cure = cure_dir + '/'+ split +'_cure.png'
        self.save_dict = cure_dir + '/'+ split+ '_dict.npy'

    def update(self, scores, epoch, loss):
        self.OA_list.append(np.float((scores['Overall_Acc'])))
        self.Mean_Iou.append(np.float((scores['Mean_IoU'])))
        self.Pre_list.append(np.float((scores['precision_1'])))
        self.Recall_list.append(np.float((scores['recall_1'])))
        self.F1_list.append(np.float((scores['F1_1'])))
        self.Loss_list.append(np.float(loss))
        self.x = np.linspace(1, epoch, epoch)

    def show(self):
        plt.plot(self.x, self.OA_list, linewidth=1, label='OA')
        plt.plot(self.x, self.Mean_Iou, linewidth=1, label='Iou')
        plt.plot(self.x, self.Pre_list, linewidth=1, label='Pre')
        plt.plot(self.x, self.Recall_list, linewidth=1, label='Recall')
        plt.plot(self.x, self.F1_list, linewidth=1, label='F1')
        plt.plot(self.x, self.Loss_list, linewidth=1, label='Loss')

        plt.title("Validation Curve", fontsize=12)
        plt.xlabel("Epochs", fontsize=12)
        plt.legend()
        plt.show()

    def save(self):
        plt.plot(self.x, self.OA_list, linewidth=1, label='OA')
        plt.plot(self.x, self.Mean_Iou, linewidth=1, label='Iou')
        plt.plot(self.x, self.Pre_list, linewidth=1, label='Pre')
        plt.plot(self.x, self.Recall_list, linewidth=1, label='Recall')
        plt.plot(self.x, self.F1_list, linewidth=1, label='F1')
        plt.plot(self.x, self.Loss_list, linewidth=1, label='Loss')

        plt.title("Validation Curve", fontsize=12)
        plt.xlabel("Epochs", fontsize=12)
        plt.legend()

        dict = {
                "OA_list": self.OA_list,
                'Mean_Iou': self.Mean_Iou,
                'Pre_list': self.Pre_list,
                'Recall_list': self.Recall_list,
                "F1_list": self.F1_list,
                "Loss_list": self.Loss_list,
            }
        plt.savefig(self.save_cure)
        np.save(self.save_dict, dict)

class PloterTrain:
    def __init__(self, project_dir, split):
        self.Loss_list = []

        self.x = None
        cure_dir = project_dir + '/' + 'curve'
        if os.path.exists(cure_dir) == False:
            os.mkdir(cure_dir)
        self.save_cure = cure_dir + '/'+ split +'_cure.png'
        self.save_dict = cure_dir + '/'+ split+ '_dict.npy'
        print(self.save_cure)

    def update(self, epoch, loss):
        self.Loss_list.append(np.float(loss))
        self.x = np.linspace(1, epoch, epoch)

    def show(self):
        plt.plot(self.x, self.Loss_list, linewidth=1, label='Loss')

        plt.title("Train Loss Curve", fontsize=12)
        plt.xlabel("Epochs", fontsize=12)
        plt.legend()
        plt.show()

    def save(self):
        plt.plot(self.x, self.Loss_list, linewidth=1, label='Loss')
        plt.title("Validation Curve", fontsize=12)
        plt.xlabel("Epochs", fontsize=12)
        plt.legend()

        dict = {
                "Loss_list": self.Loss_list,
            }
        plt.savefig(self.save_cure)
        np.save(self.save_dict, dict)


'''
模型训练器
'''
class CDTrainner:
    def __init__(self, project_name, data_dir, model, loss_function, optimizer, lr_Scheduler, max_epoch, device, batchsize, RESUM=False, edge= False, lamda=0.1):
        #建立相关目录
        if os.path.exists(project_name) == False:
            os.mkdir(project_name)

        self.best_path = project_name + '/bestNet'
        if os.path.exists(self.best_path) == False:
            os.mkdir(self.best_path)

        self.log_path = project_name + '/log'
        if os.path.exists(self.log_path) == False:
            os.mkdir(self.log_path)

        self.checkpoint_path = project_name + '/check'
        if os.path.exists(self.checkpoint_path) == False:
            os.mkdir(self.checkpoint_path)

        self.checkpoint_path = self.checkpoint_path + '/check' + '.pth'

        #模型相关
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.lr_Scheduler = lr_Scheduler
        self.loss_function = loss_function

        #数据集
        #dataset preparation
        self.dataloder_train, self.dataloder_val, self.dataloder_test = getdataloder(data_dir,batchsize)

        #训练状态记录
        self.start_epoch = 0
        self.currentEpoch = 0
        self.max_epoch = max_epoch
        self.bestF1 = 0

        #曲线
        self.Train_ploter = PloterTrain(project_name, 'train')
        self.Validation_ploter = PloterVal(project_name, 'validation')

        if RESUM:
            self.load_Checkpoint()

        #日志目录
        self.log_path = self.log_path + '/log.txt'
        logging.basicConfig(level=logging.CRITICAL,
                            filename=self.log_path,
                            filemode='w',
                            format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

        self.edge = edge
        self.lamda = lamda

    def save_Checkponit(self):
        if self.currentEpoch % 1 == 0:  # 每隔1个epoch保存一次模型断点
            print('learning rate:', self.optimizer.state_dict()['param_groups'][0]['lr'])
            # 自定义要保存的参数信息
            check_Train_curve = [self.Train_ploter.Loss_list]

            check_Validation_curve = [self.Validation_ploter.OA_list, self.Validation_ploter.Mean_Iou,
                                      self.Validation_ploter.Pre_list,
                                      self.Validation_ploter.Recall_list, self.Validation_ploter.F1_list,
                                      self.Validation_ploter.Loss_list]
            checkpoint = {
                "net": self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_schedule': self.lr_Scheduler.state_dict(),
                'curv_Train': check_Train_curve,
                'curv_Validation': check_Validation_curve,
                "epoch": self.currentEpoch
            }
            torch.save(checkpoint, self.checkpoint_path)

    def load_Checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path)  # 加载断点

        # if self.refine == False:
        self.model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
        self.optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        self.lr_Scheduler.load_state_dict(checkpoint['lr_schedule'])  # 加载lr_scheduler
        self.start_epoch = checkpoint['epoch']  #设置开始的epoch

# #         else:
#         self.model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
#         self.start_epoch = checkpoint['epoch']

        self.Train_ploter.Loss_list = checkpoint['curv_Train'][0]
        self.Validation_ploter.OA_list = checkpoint['curv_Validation'][0]
        self.Validation_ploter.Mean_Iou = checkpoint['curv_Validation'][1]
        self.Validation_ploter.Pre_list = checkpoint['curv_Validation'][2]
        self.Validation_ploter.Recall_list = checkpoint['curv_Validation'][3]
        self.Validation_ploter.F1_list = checkpoint['curv_Validation'][4]
        self.Validation_ploter.Loss_list = checkpoint['curv_Validation'][5]

    def train(self):
        sum_loss = 0
        numcount = 0
        self.model.train()
        # 训练
        tq_data_loader_train = tqdm(self.dataloder_train)
        for img1, img2, label, edge in tq_data_loader_train:
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            label = label.to(self.device).view(label.shape[0], 256, 256)
            edge = edge.to(self.device).view(edge.shape[0], 256, 256)

            if self.edge == True:
                pre1, pre2, pre3, pre4, pre5, predge = self.model(img1, img2)
                #变化损失
                l1 = self.loss_function[0](pre1, label)
                l2 = self.loss_function[0](pre2, label)
                l3 = self.loss_function[0](pre3, label)
                l4 = self.loss_function[0](pre4, label)
                l5 = self.loss_function[0](pre5, label)
                ltotal = l1 + (l2 + l3 + l4 + l5) / 4
                #边缘损失
                ledge = self.loss_function[1](predge, edge)
                if self.lamda == 'adjustable':
                    ltotal = ltotal + self.model.lamda * ledge
                else:
                    ltotal = ltotal+self.lamda*ledge
                ltotal.backward()

                self.optimizer.step()  # 更新一次参数
                self.optimizer.zero_grad()

            else:
                pre1, pre2, pre3, pre4, pre5 = self.model(img1, img2)
                # 变化损失
                l1 = self.loss_function[0](pre1, label)
                l2 = self.loss_function[0](pre2, label)
                l3 = self.loss_function[0](pre3, label)
                l4 = self.loss_function[0](pre4, label)
                l5 = self.loss_function[0](pre5, label)
                ltotal = l1 + (l2 + l3 + l4 + l5) / 4
                ltotal.backward()

                self.optimizer.step()  # 更新一次参数
                self.optimizer.zero_grad()

            sum_loss = sum_loss + ltotal.item()
            numcount = numcount + 1

        self.Train_ploter.update(self.currentEpoch, sum_loss / numcount)
        self.lr_Scheduler.step()

    def eval(self):
        with torch.no_grad():
            metric = RunningMetrics(2)
            edge_metric = RunningMetrics(2)

            running_metrics_change = AverageMeter()
            running_metrics_edge = AverageMeter()

            change_sum_loss = 0
            edge_sum_loss = 0
            numcount = 0

            self.model.eval()
            for img1, img2, label, edge in self.dataloder_val:
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)

                label = label.to(self.device).view(label.shape[0], 256, 256)
                edge = edge.to(self.device).view(edge.shape[0], 256, 256)

                if self.edge == True:
                    pre1, pre2, pre3, pre4, pre5, predge = self.model(img1, img2)

                    #change loss
                    l1 = self.loss_function[0](pre1, label)
                    l2 = self.loss_function[0](pre2, label)
                    l3 = self.loss_function[0](pre3, label)
                    l4 = self.loss_function[0](pre4, label)
                    l5 = self.loss_function[0](pre5, label)
                    lchange = l1 + (l2 + l3 + l4 + l5) / 4

                    result_change = torch.argmax(pre1, dim=1).long()
                    metric.update(label.detach().cpu().numpy(), result_change.detach().cpu().numpy())
                    scores = metric.get_cm()
                    running_metrics_change.update(scores)
                    metric.reset()
                    change_sum_loss = change_sum_loss + lchange

                    #edge loss
                    ledge = self.loss_function[1](predge, edge)
                    result_edge = torch.where(predge > 0.5, 1, 0).long()
                    edge_metric.update(edge.detach().cpu().numpy(), result_edge.detach().cpu().numpy())
                    scores = edge_metric.get_cm()
                    running_metrics_edge.update(scores)
                    edge_metric.reset()

                else:
                    pre1, pre2, pre3, pre4, pre5  = self.model(img1, img2)

                    # change loss
                    l1 = self.loss_function[0](pre1, label)
                    l2 = self.loss_function[0](pre2, label)
                    l3 = self.loss_function[0](pre3, label)
                    l4 = self.loss_function[0](pre4, label)
                    l5 = self.loss_function[0](pre5, label)
                    lchange = l1 + (l2 + l3 + l4 + l5) / 4

                    #change loss
                    lchange = self.loss_function[0](pre1, edge)
                    result_change = torch.argmax(pre1, dim=1).long()
                    metric.update(label.detach().cpu().numpy(), result_change.detach().cpu().numpy())
                    scores = metric.get_cm()
                    running_metrics_change.update(scores)
                    metric.reset()

                    #edge loss
                    ledge = self.loss_function[1](predge, edge)
                    result_edge = torch.where(predge > 0.5, 1, 0).long()
                    edge_metric.update(edge.detach().cpu().numpy(), result_edge.detach().cpu().numpy())
                    scores = edge_metric.get_cm()
                    running_metrics_edge.update(scores)
                    edge_metric.reset()

                change_sum_loss = change_sum_loss + lchange.item()
                edge_sum_loss = edge_sum_loss + ledge.item()
                numcount = numcount + 1

            #获取分数
            score_change = running_metrics_change.get_scores()
            logging.critical(score_change)
            print(score_change)

            score_edge = running_metrics_edge.get_scores()
            logging.critical(score_edge)
            print(score_edge)

            #寻找最优模型
            if score_change['F1_1'] > self.bestF1:
                self.bestF1 = score_change['F1_1']
                nameList = os.listdir(self.best_path)
                for each in nameList:
                    path = self.best_path + '/' + each
                    os.remove(path)

                torch.save(self.model.state_dict(),self.best_path + '/epoch:' + str(self.currentEpoch) + '_' + str(self.bestF1))
                torch.save(self.model.state_dict(), self.best_path + '/'+str(self.currentEpoch))

            #画验证曲线
            self.Validation_ploter.update(score_change, self.currentEpoch, (change_sum_loss+edge_sum_loss)/numcount)

    def train_model(self):
        for epoch in range(self.start_epoch + 1, self.max_epoch):
            print(str(epoch) + '次训练')
            self.currentEpoch = epoch
            self.train()
            self.eval()
            self.save_Checkponit()

        self.Train_ploter.save()
        self.Validation_ploter.save()

        
        
   


'''
模型测试器
'''
class CDTester:
    def __init__(self,project_dir,data_dir,batchsize,model, device, edge, visual = False):
        folder_path = project_dir + '/bestNet'
        file_list = os.listdir(folder_path)
        print(file_list)
        bestPath = project_dir + '/bestNet'  + '/'+ file_list[0]
        self.imgPath = project_dir + '/visual'
        if os.path.exists(self.imgPath) == False:
            os.mkdir(self.imgPath)

        self.model = model
        self.dataloder_train, self.dataloder_val, self.dataloder_test = getdataloder(data_dir,batchsize)
      
        self.device = device
        self.model = self.model.to(device)
        dict = torch.load(bestPath)
        self.model.load_state_dict(dict)
        self.edge = edge
        self.visual = visual

        self.numcount = 0
        self.siam = 1

    def test(self):
        with torch.no_grad():
            self.model.eval()
            metric_change = RunningMetrics(2)
            running_metrics_change = AverageMeter()

            metric_edge = RunningMetrics(2)
            running_metrics_edge = AverageMeter()

            for img1_or,img2_or,img1, img2, label, edge in self.dataloder_test:
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                label = label.to(self.device)
                edge = edge.to(self.device)
                
                label = label.to(self.device).view(label.shape[0],256, 256)
                edge = edge.to(self.device).view(label.shape[0],256, 256)

                if self.edge == True:
                    pre1, pre2, pre3, pre4, pre5, preedge = self.model(img1, img2)
                 
                    #计算变化性能
                    result_change = torch.argmax(pre1, dim=1).long()
                    metric_change.update(label.detach().cpu().numpy(), result_change.detach().cpu().numpy())
                    scores = metric_change.get_cm()
                    running_metrics_change.update(scores)
                    metric_change.reset()

                    #计算边缘性能
                    result_edge = torch.where(preedge>0.5, 1, 0).long()
                    metric_change.update(edge.detach().cpu().numpy(), result_edge.detach().cpu().numpy())
                    scores = metric_change.get_cm()
                    running_metrics_edge.update(scores)
                    metric_change.reset()
                    
                    if self.visual:
                        self.visual1(self.numcount,img1_or, img2_or, result_change, label,result_edge,edge)

                else:
                    pre1, pre2, pre3, pre4, pre5,_ = self.model(img1, img2)

                    # 计算变化性能
                    result_change = torch.argmax(pre1, dim=1).long()
                    metric_change.update(label.detach().cpu().numpy(), result_change.detach().cpu().numpy())
                    scores = metric_change.get_cm()
                    running_metrics_change.update(scores)
                    metric_change.reset()
                    
                    if self.visual:
                        self.visual2(self.numcount, img1_or, img2_or, result_change, label)
                        
                self.numcount = self.numcount + 1

            score_change = running_metrics_change.get_scores()
            score_edge = running_metrics_edge.get_scores()
            print(score_change)
            print(score_edge)

    def visual1(self, numcount,img1_or, img2_or, result_change, label, result_edge, edge):
        imgdir = self.imgPath + '/' + str(numcount)
        if not os.path.exists(imgdir):
            os.mkdir(imgdir)

        img1_or = img1_or[0].to(torch.device('cpu')).detach().numpy().reshape(256, 256, 3)
        img2_or = img2_or[0].to(torch.device('cpu')).detach().numpy().reshape(256, 256, 3)

        result_change = result_change[0].to(torch.device('cpu')).detach().numpy().reshape(256, 256, 1)
        result_edge = result_edge[0].to(torch.device('cpu')).detach().numpy().reshape(256, 256, 1)
        label = label[0].to(torch.device('cpu')).detach().numpy().reshape(256, 256, 1)
        edge = edge[0].to(torch.device('cpu')).detach().numpy().reshape(256, 256, 1)

        #生成分析图片
        TP = result_change*label*255         #白色
        TP = np.concatenate((TP, TP, TP), axis=2)

        FP = result_change*(1-label)*255     #红色
        zero = np.zeros((256,256,1))
        FP = np.concatenate((zero, zero, FP), axis=2)

        FN = (1-result_change)*label*255     #绿色
        FN = np.concatenate((zero, FN, zero), axis=2)

        annalysis = TP + FP + FN
        result_change = result_change*255
        result_edge = result_edge*255
        label = label*255
        edge = edge*255

        path_Img1 = imgdir + '/1.jpg'
        path_Img2 = imgdir + '/2.jpg'
        path_changeresult  = imgdir + '/pre.jpg'
        path_edgeresult = imgdir + '/pre.jpg'

        path_Label = imgdir + '/label.jpg'
        path_Edge = imgdir + '/edge.jpg'
        path_analysis = imgdir + '/analysis.jpg'

        # cv2.imwrite(path_Img1, img1_or)
        # cv2.imwrite(path_Img2, img2_or)
        cv2.imwrite(path_changeresult, result_change)
        cv2.imwrite(path_edgeresult, result_edge)
        cv2.imwrite(path_Label, label)
        cv2.imwrite(path_Edge, edge)
        cv2.imwrite(path_analysis, annalysis)

    def visual2(self, numcount, img1_or, img2_or, result_change, label):
        imgdir = self.imgPath + '/' + str(numcount)
        if not os.path.exists(imgdir):
            os.mkdir(imgdir)

        img1_or = img1_or[0].to(torch.device('cpu')).detach().numpy().reshape(256, 256, 3)
        img2_or = img2_or[0].to(torch.device('cpu')).detach().numpy().reshape(256, 256, 3)

        result_change = result_change[0].to(torch.device('cpu')).detach().numpy()
        label = label[0].to(torch.device('cpu')).detach().numpy().reshape(256, 256, 1)

        # 生成分析图片
        TP = result_change * label * 255  # 白色
        TP = np.concatenate((TP, TP, TP), axis=2)

        FP = result_change * (1 - label) * 255  # 红色
        zero = np.zeros((256, 256, 1))
        FP = np.concatenate((zero, zero, FP), axis=2)

        FN = (1 - result_change) * label * 255  # 绿色
        FN = np.concatenate((zero, FN, zero), axis=2)

        annalysis = TP + FP + FN
        result_change = result_change * 255
        label = label * 255

        path_Img1 = imgdir + '/1.jpg'
        path_Img2 = imgdir + '/2.jpg'
        path_resultchange = imgdir + '/pre.jpg'
        path_Label = imgdir + '/label.jpg'
        path_analysis = imgdir + '/analysis.jpg'

        # cv2.imwrite(path_Img1, img1_or)
        # cv2.imwrite(path_Img2, img2_or)
        cv2.imwrite(path_resultchange, result_change)
        cv2.imwrite(path_Label, label)
        cv2.imwrite(path_analysis, annalysis)

