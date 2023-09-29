# coding=utf-8
from data.data_utils import CDDataAugmentation
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np
import glob
from torch.utils.data import DataLoader
import torch
import os
import cv2
import numpy as np



class CDDDataset(Dataset):
    """ChangeDataset Numpy Pickle Dataset"""
    def __init__(self, rootdir, split, edge=False):
        super(CDDDataset, self).__init__()
        files = rootdir +'/'+split
        image_path1 = glob.glob(files + '/A' + '/*.jpg')
        image_path1.sort()
        self.image_path1 = image_path1

        image_path2 = glob.glob(files + '/B' + '/*.jpg')
        image_path2.sort()
        self.image_path2 = image_path2

        label_path = glob.glob(files + '/OUT' + '/*.jpg')
        label_path.sort()
        self.label_path = label_path

        edge_path = glob.glob(files + '/EDGE' + '/*.jpg')
        edge_path.sort()
        self.edge_path = edge_path

        self.split = split
        self.edge = edge
        if split == 'train':
            #The common data augmentation strategies are not used which may lead more accurate results.
            self.augm = CDDataAugmentation(
                img_size=256
                # with_random_hflip=True,
                # with_random_vflip=True,
                # with_scale_random_crop=True,
                # with_random_blur=True,
                # random_color_tf=True
            )
        else:
            self.augm = CDDataAugmentation(
                img_size=256
            )

    def __len__(self):
        assert len(self.image_path1) == len(self.image_path2)
        return len(self.image_path1)

    def __getitem__(self, idx):
        img1_path = self.image_path1[idx]
        img2_path = self.image_path2[idx]
        label_path = self.label_path[idx]
        edge_path = self.edge_path[idx]

        img1 = np.asarray(Image.open(img1_path).convert('RGB'))
        img2 = np.asarray(Image.open(img2_path).convert('RGB'))
        label = np.array(Image.open(label_path), dtype=np.uint8)
        edge = np.array(Image.open(edge_path), dtype=np.uint8)

        #标签归一化（CDD数据集需要控制阈值抵抗高斯模糊）
        label = label//200
        edge = edge//255

        img1_original = img1
        img2_original = img2

        #数据增强
        [img1, img2], [label,edge] = self.augm.transform([img1, img2], [label,edge], to_tensor=True, nomalization=None)


        if self.split == 'test':
            if self.edge == True:
                return img1_original, img2_original, img1, img2, label,edge
            else:
                return img1_original, img2_original, img1, img2, label

        else:
            if self.edge == True:
                return img1, img2, label,edge
            else:
                return img1, img2, label








#边缘提取代码
# ori_dir = '/home/lkk/PytorchWork/Data-Set/CDD/test/OUT'
# des_dir = '/home/lkk/PytorchWork/Data-Set/CDD/test/EDGE'
#
# dirs = os.listdir(ori_dir)
# for file in dirs:
#     path_ori = os.path.join(ori_dir, file)
#     img = cv2.imread(path_ori,0)
#     img = np.where(img > 200,255,0)
#
#     canny = cv2.Canny(np.uint8(img),20,200 )
#
#     path_des = os.path.join(des_dir, file)
#     cv2.imwrite(path_des ,canny)
#
#
























# root_dir = '/home/lkk/PytorchWork/Data-Set/CDD'
# dataset_train = CDDDataset(root_dir,"train")
# data_loader_train = DataLoader(dataset_train, batch_size=10000, shuffle=False,num_workers=1)
# for img1,img2,label in data_loader_train:
#     cat = torch.cat([img1,img2],dim=0)
#     print(cat.shape)
#     print(cat[:,0,:,:].mean())
#     print(cat[:,1,:,:].mean())
#     print(cat[:,2,:,:].mean())
#     print('方差')
#     print(cat[:,0,:,:].std())
#     print(cat[:,1,:,:].std())
#     print(cat[:,2,:,:].std())
#
# dataset_val = CDDDataset(root_dir,"val")
# data_loader_val = DataLoader(dataset_val, batch_size=2998, shuffle=False,num_workers=1)
# for img1,img2,label in data_loader_val:
#     cat = torch.cat([img1,img2],dim=0)
#     print(cat.shape)
#     print(cat[:,0,:,:].mean())
#     print(cat[:,1,:,:].mean())
#     print(cat[:,2,:,:].mean())
#     print('方差')
#     print(cat[:,0,:,:].std())
#     print(cat[:,1,:,:].std())
#     print(cat[:,2,:,:].std())
#
# dataset_test = CDDDataset(root_dir,"test")
# data_loader_test = DataLoader(dataset_test, batch_size=3000, shuffle=False,num_workers=1)
# for o1,o2,img1,img2,label in data_loader_test:
#     cat = torch.cat([img1,img2],dim=0)
#     print(cat.shape)
#     print(cat[:,0,:,:].mean())
#     print(cat[:,1,:,:].mean())
#     print(cat[:,2,:,:].mean())
#     print('方差')
#     print(cat[:,0,:,:].std())
#     print(cat[:,1,:,:].std())
#     print(cat[:,2,:,:].std())

