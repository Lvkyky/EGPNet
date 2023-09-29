import os
import torchvision.utils as vutil
from sklearn.decomposition import PCA
import cv2
import torch.nn.functional as F
import numpy as np
import torch


def draw_in_pic(img,count):
    '''
    todo:   draw text info on picture
    img:    numpy
    '''
    str_context = str(count)
    font = cv2.FONT_HERSHEY_COMPLEX # 字体类型
    color = (255,255,255) # 颜色选择，单通道只有黑白
    start_point = (10,20) # 文字起始点
    print_size = 1 # 字体大小
    thickness = 1 # 文字粗细
    cv2.putText(img, str_context, start_point,font,print_size,color,thickness)
    return img


#可视化特征图函数
def visulize(features,wordir,numcount,name,mode):
    imgdir = wordir+'/'+'feature_visual'
    if not os.path.exists(imgdir):
        os.mkdir(imgdir)

    imgdir = wordir + '/' + 'feature_visual' + '/' + str(numcount)
    if not os.path.exists(imgdir):
        os.mkdir(imgdir)

    image_name = imgdir+'/'+name+'.jpg'
    numpy_name = imgdir+'/'+name+'.npy'

    data = features
    if mode == 'PCA3':
        data = features.permute(2, 3, 1, 0)#w,h,c,b
        data = data.view(data.shape[0], data.shape[1], data.shape[2])
        w = data.shape[0]
        data = data.view(data.shape[0] * data.shape[1], data.shape[2])
        data = data.cpu().numpy()

        pca = PCA(n_components=3)  # 选取4个主成分
        data_low = pca.fit_transform(data)  # 对原数据进行pca处理
        picdata = data_low.reshape(w, w, 3)

    elif mode == 'PCA1':
        data = data.permute(2, 3, 1, 0)#w,h,c,b
        data = data.view(data.shape[0], data.shape[1], data.shape[2])
        w = data.shape[0]
        data = data.view(data.shape[0] * data.shape[1], data.shape[2])
        data = data.cpu().numpy()

        pca = PCA(n_components=1)  # 选取4个主成分
        data_low = pca.fit_transform(data)  # 对原数据进行pca处理
        picdata = data_low.reshape(w, w, 1)

    elif mode == 'avg':
        data = torch.mean(data, dim=1, keepdim=True)#[B, 1, H, W], average
        picdata = data.cpu().numpy()
        picdata = picdata.reshape(256,256,1)
        print(picdata.shape)

    elif mode == 'max':
        data, _ = torch.max(data, dim=1, keepdim=True)#[B, 1, H, W], max
        picdata = data.cpu().numpy()
        picdata = picdata.reshape(256, 256, 1)

    elif mode == 'grid':
        data = data.cpu().numpy()
        count = data.shape[1]
        width = data.shape[2]
        data = data.reshape(count,width,width)
        stack = np.ones((width,width))

        #########################热力图#####################
        # heatmap = data[3]
        # min = np.min(heatmap)
        # max = np.max(heatmap)
        #
        # heatmap = (heatmap-min)/(max-min)
        # heatmap = heatmap*255
        #
        # heatmap = heatmap.astype(np.uint8)
        # heatmap = cv2.applyColorMap(heatmap,2)
        # print(heatmap.shape)
        # cv2.imwrite(image_name,heatmap)
        np.save(numpy_name, data)
        ########################热力图#####################
        # print(heatmap.shape)

        for each in range(0,count):
            tem = data[each]*5
            tem = draw_in_pic(tem,each)
            stack = np.vstack((stack,tem))

        # picdata = stack.reshape(256,32512,1)
        picdata = stack

    # pic = cv2.resize(picdata, (0, 0), fx=2, fy=2,interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(image_name, picdata*25+150)



