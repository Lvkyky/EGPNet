# coding=utf-8
from data.data_utils import CDDataAugmentation
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np
import glob
import os
import cv2
from torch.utils.data import DataLoader


class SYSUDataset(Dataset):
    """ChangeDataset Numpy Pickle Dataset"""
    def __init__(self, root_dir, split, edge):
        super(SYSUDataset, self).__init__()
        files = root_dir + '/' + split
        image_path1 = glob.glob(files + '/time1' + '/*.png')
        image_path1.sort()
        self.image_path1 = image_path1

        image_path2 = glob.glob(files + '/time2' + '/*.png')
        image_path2.sort()
        self.image_path2 = image_path2

        label_path = glob.glob(files + '/label' + '/*.png')
        label_path.sort()
        self.label_path = label_path

        edge_path = glob.glob(files + '/edge' + '/*.png')
        edge_path.sort()
        self.edge_path = edge_path

        self.split = split
        self.edge = edge

        if split == 'train':
            # The common data augmentation strategies are not used which may lead more accurate results.
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
        # return len(self.data_dict)
        assert len(self.image_path1) == len(self.image_path2)
        return len(self.image_path1)

    def __getitem__(self, idx):
        img1_path  = self.image_path1[idx]
        img2_path  = self.image_path2[idx]
        label_path = self.label_path[idx]
        edge_path  = self.edge_path[idx]

        img1 = np.asarray(Image.open(img1_path).convert('RGB'))
        img2 = np.asarray(Image.open(img2_path).convert('RGB'))
        label = np.array(Image.open(label_path), dtype=np.uint8)
        edge = np.array(Image.open(edge_path), dtype=np.uint8)

        #标签归一化
        label = label//255
        edge = edge//255

        img1_original = img1
        img2_original = img2

        #数据增强
        [img1, img2], [label,edge] = self.augm.transform([img1, img2], [label,edge],to_tensor=True,nomalization=[[0.5,0.5,0.5],[0.5,0.5,0.5]])


        if self.edge == True:
            if self.split == 'test':
                return img1_original,img2_original,img1,img2,label,edge
            else:
                return img1, img2, label,edge

        else:
            if self.split == 'test':
                return img1_original,img2_original,img1,img2,label
            else:
                return img1, img2, label



