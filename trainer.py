import os
from util.metrics import RunningMetrics, AverageMeter
import numpy as np
import logging
import torch
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch.nn.functional as F
def findmax(path):
    img_name_list = np.loadtxt(path, dtype=np.str)
    list = []
    for each in img_name_list:
        tem = (each[-1][0:-1])
        list.append(float(tem))
    print(list)
    m = max(list)
    index = list.index(m)
    return m,index


'''
曲线显示
'''
class PloterVal:
    def __init__(self, WorkDir, model, case):
        self.OA_list = []
        self.Mean_Iou = []
        self.Pre_list = []
        self.Recall_list = []
        self.F1_list = []
        self.Loss_list = []

        self.x = None
        cure_dir = WorkDir + '/' + 'curve' + str(case)
        if os.path.exists(cure_dir) == False:
            os.mkdir(cure_dir)
        self.save_cure = cure_dir + '/'+ model +'_cure.png'
        self.save_dict = cure_dir + '/'+ model+ '_dict.npy'
        print(self.save_cure)

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
    def __init__(self, WorkDir, model, case):
        self.Loss_list = []

        self.x = None
        cure_dir = WorkDir + '/' + 'curve' + str(case)
        if os.path.exists(cure_dir) == False:
            os.mkdir(cure_dir)
        self.save_cure = cure_dir + '/'+ model +'_cure.png'
        self.save_dict = cure_dir + '/'+ model+ '_dict.npy'
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
class CDTrainner_Edge:
    def __init__(self, work_dir, model, loss_function, optimizer, lr_Scheduler, data_loader_train, data_loader_val,max_epoch, device, mode=2, RESUM=False, refine=False, case=0):
        if os.path.exists(work_dir) == False:
            os.mkdir(work_dir)

        self.save_path = work_dir + '/hisNet' + str(case)
        if os.path.exists(self.save_path) == False:
            os.mkdir(self.save_path)

        self.save_path = self.save_path + '/'

        #######################################################3
        self.log_path = work_dir + '/log' + str(case)
        if os.path.exists(self.log_path) == False:
            os.mkdir(self.log_path)

        ############################################################33
        self.checkpoint_path = work_dir + '/check' + str(case)
        if os.path.exists(self.checkpoint_path) == False:
            os.mkdir(self.checkpoint_path)
        self.checkpoint_path = self.checkpoint_path + '/check' + '.pth'

        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.lr_Scheduler = lr_Scheduler
        self.loss_function = loss_function
        self.data_loader_train = data_loader_train
        self.data_loader_val = data_loader_val

        self.start_epoch = 0
        self.currentEpoch = 0
        self.max_epoch = max_epoch

        self.Train_ploter = PloterTrain(work_dir, 'train',  case)
        self.Validation_ploter = PloterVal(work_dir, 'validation', case)

        self.mode = mode
        self.refine = refine
        self.bestF1 = 0

        if RESUM:
            self.load_Checkpoint()

        self.log_path = self.log_path + '/log.txt'
        logging.basicConfig(level=logging.CRITICAL,
                            filename=self.log_path,
                            filemode='w',
                            format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

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

        if self.refine == False:
            self.model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
            self.optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
            self.lr_Scheduler.load_state_dict(checkpoint['lr_schedule'])  # 加载lr_scheduler
            self.start_epoch = checkpoint['epoch']  #设置开始的epoch

        else:
            self.model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
            self.start_epoch = checkpoint['epoch']

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
        tq_data_loader_train = tqdm(self.data_loader_train)
        for img1, img2, label, edge in tq_data_loader_train:
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            label = label.to(self.device).view(label.shape[0], 256, 256)
            edge = edge.to(self.device).view(edge.shape[0], 256, 256)

            pre, pre2, pre3, pre4, pre5, predge = self.model(img1, img2)

            #变化损失
            l = self.loss_function[0](pre, label)
            l2 = self.loss_function[0](pre2, label)
            l3 = self.loss_function[0](pre3, label)
            l4 = self.loss_function[0](pre4, label)
            l5 = self.loss_function[0](pre5, label)
            ltotal = l + (l2 + l3 + l4 + l5) / 4

            #边缘损失
            ledge = self.loss_function[1](predge, edge)
            ltotal = ltotal+0.1*ledge
            ltotal.backward()

            self.optimizer.step()  # 更新一次参数
            self.optimizer.zero_grad()

            sum_loss = sum_loss + ltotal
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
            for img1, img2, label, edge in self.data_loader_val:
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)

                label = label.to(self.device).view(label.shape[0], 256, 256)
                edge = edge.to(self.device).view(edge.shape[0], 256, 256)

                pre, pre2, pre3, pre4, pre5, predge = self.model(img1, img2)

                #change loss
                l = self.loss_function[0](pre, label)
                l2 = self.loss_function[0](pre2, label)
                l3 = self.loss_function[0](pre3, label)
                l4 = self.loss_function[0](pre4, label)
                l5 = self.loss_function[0](pre5, label)
                lchange = l + (l2 + l3 + l4 + l5) / 4

                pre_l = torch.argmax(pre, dim=1)
                pre_l = pre_l.long()
                metric.update(label.detach().cpu().numpy(), pre_l.detach().cpu().numpy())
                scores = metric.get_cm()
                running_metrics_change.update(scores)
                metric.reset()
                change_sum_loss = change_sum_loss + lchange

                #edge loss
                ledge = self.loss_function[1](predge, edge)
                pedge_l = torch.where(predge > 0.5, 1, 0)
                pedge_l = pedge_l.long()
                edge_metric.update(edge.detach().cpu().numpy(), pedge_l.detach().cpu().numpy())
                scores = edge_metric.get_cm()
                running_metrics_edge.update(scores)
                edge_metric.reset()
                edge_sum_loss = edge_sum_loss + ledge
                numcount = numcount + 1

            score_change = running_metrics_change.get_scores()
            logging.critical(score_change)
            print(score_change)

            score_edge = running_metrics_edge.get_scores()
            logging.critical(score_edge)

            if score_change['F1_1'] > self.bestF1:
                self.bestF1 = score_change['F1_1']
                nameList = os.listdir(self.save_path)
                for each in nameList:
                    path = self.save_path + each
                    os.remove(path)
                torch.save(self.model.state_dict(),self.save_path + 'epoch:' + str(self.currentEpoch) + '_' + str(self.bestF1))
                torch.save(self.model.state_dict(), self.save_path + str(self.currentEpoch))
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
        self.Train_ploter.save()
        self.Validation_ploter.save()



'''
模型测试器
'''
class CDTester_Edge:
    def __init__(self, model, lossFunction, workDir, bestEpoch, dataLoader_test, device, case):
        bestPath = workDir + '/hisNet' + str(case) + '/'+ str(bestEpoch)
        self.imgPath = workDir + '/visual' + str(case)

        if os.path.exists(self.imgPath) == False:
            os.mkdir(self.imgPath)

        self.model = model
        self.lossFunction = lossFunction
        self.dataLoader_test = dataLoader_test
        self.device = device
        self.model = self.model.to(device)
        dict = torch.load(bestPath)
        self.model.load_state_dict(dict)

    def test(self):
        self.model.eval()
        with torch.no_grad():
            metric_change = RunningMetrics(2)
            metric_edge = RunningMetrics(2)

            running_metrics_change = AverageMeter()
            running_metrics_edge = AverageMeter()

            numcount = 0
            change_sum_loss = 0
            edge_sum_loss = 0

            for img1_or, img2_or, img1, img2, label, edge in self.dataLoader_test:
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                label = label.to(self.device)
                label = torch.where(label>0.5,1,0)
                edge = edge.to(self.device)

                pre, pre2, pre3, pre4, pre5, predge = self.model(img1, img2)
                label = label.to(self.device).view(label.shape[0],1,256,256)
                edge = edge.to(self.device).view(label.shape[0],1,256,256)

                #change loss
                l = self.lossFunction[0](pre, label)
                l2 = self.lossFunction[0](pre2, label)
                l3 = self.lossFunction[0](pre3, label)
                l4 = self.lossFunction[0](pre4, label)
                l5 = self.lossFunction[0](pre5, label)
                lchange = l + (l2 + l3 + l4 + l5) / 4

                pre_l = torch.argmax(pre, dim=1)
                pre_l = pre_l.long()
                metric_change.update(label.detach().cpu().numpy(), pre_l.detach().cpu().numpy())
                scores = metric_change.get_cm()
                running_metrics_change.update(scores)
                metric_change.reset()
                change_sum_loss = change_sum_loss + lchange

                #change edge loss
                ledge = self.lossFunction[1](predge, edge)
                pedge_l = torch.where(predge > 0.5, 1, 0)
                pedge_l = pedge_l.long()
                metric_edge.update(edge.detach().cpu().numpy(), pedge_l.detach().cpu().numpy())
                scores = metric_edge.get_cm()
                running_metrics_edge.update(scores)
                metric_edge.reset()
                edge_sum_loss = edge_sum_loss + ledge

                #visualize bi-temporal images and the results predicted by model
                self.visual(img1_or, img2_or, pre, label, predge, edge, numcount)
                numcount = numcount + 1

            score1 = running_metrics_change.get_scores()
            score2 = running_metrics_edge.get_scores()
            return score1,score2

    def visual(self, img1_or, img2_or, pre, label,preedge,edge,numcount):
        if numcount < 1 :
            pass
        else:
            imgdir = self.imgPath + '/' + str(numcount)
            if not os.path.exists(imgdir):
                os.mkdir(imgdir)

            img1_or = img1_or[0].to(torch.device('cpu')).detach().numpy().reshape(256, 256, 3)
            img2_or = img2_or[0].to(torch.device('cpu')).detach().numpy().reshape(256, 256, 3)


            pre = pre[0].to(torch.device('cpu')).detach().numpy()
            pre = np.argmax(pre, axis=0).reshape(256, 256, 1)

            preedge = preedge[0].to(torch.device('cpu')).detach().numpy()
            preedge = np.where(preedge > 0.5, 1, 0).reshape(256, 256, 1)

            label = label[0].to(torch.device('cpu')).detach().numpy().reshape(256, 256, 1)
            edge = edge[0].to(torch.device('cpu')).detach().numpy().reshape(256, 256, 1)

            #生成分析图片
            TP = pre*label*255         #白色
            TP = np.concatenate((TP, TP, TP), axis=2)

            FP = pre*(1-label)*255     #红色
            zero = np.zeros((256,256,1))
            FP = np.concatenate((zero, zero, FP), axis=2)

            FN = (1-pre)*label*255     #绿色
            FN = np.concatenate((zero, FN, zero), axis=2)

            annalysis = TP + FP + FN
            pre = pre*255
            label = label*255
            preedge = preedge*255
            edge = edge*255

            path_Img1 = imgdir + '/1.jpg'
            path_Img2 = imgdir + '/2.jpg'
            path_Pre  = imgdir + '/pre.jpg'
            path_Label = imgdir + '/label.jpg'
            path_PreEdge = imgdir + '/preedge.jpg'
            path_Edge = imgdir + '/edge.jpg'
            path_analysis = imgdir + '/analysis.jpg'

            cv2.imwrite(path_Img1, img1_or)
            cv2.imwrite(path_Img2, img2_or)
            cv2.imwrite(path_Pre, pre)
            cv2.imwrite(path_Label, label)
            cv2.imwrite(path_PreEdge, preedge)
            cv2.imwrite(path_Edge, edge)

            cv2.imwrite(path_analysis, annalysis)
