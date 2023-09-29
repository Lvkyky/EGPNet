"""
变化检测数据集
"""
import torch
import os
from PIL import Image
import numpy as np

from torch.utils import data

from data.data_utils import CDDataAugmentation
from torch.utils.data import DataLoader

"""
CD data set with pixel-level labels；
├─image
├─image_post
├─label
└─list
"""
IMG_FOLDER_NAME = "A"
IMG_POST_FOLDER_NAME = 'B'
LIST_FOLDER_NAME = 'list'
ANNOT_FOLDER_NAME = "label"


IGNORE = 255

label_suffix='.png' # jpg for gan dataset, others : png

def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=str)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list
def load_image_label_list_from_npy(npy_path, img_name_list):
    cls_labels_dict = np.load(npy_path, allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]
def get_img_post_path(root_dir,img_name):
    return os.path.join(root_dir, IMG_POST_FOLDER_NAME, img_name)
def get_img_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_FOLDER_NAME, img_name)
def get_label_path(root_dir, img_name):
    return os.path.join(root_dir, ANNOT_FOLDER_NAME, img_name.replace('.jpg', label_suffix))
def get_edge_path(root_dir, img_name):
    return os.path.join(root_dir, 'edge', img_name.replace('.jpg', label_suffix))
def get_label1_path(root_dir, img_name):
    return os.path.join(root_dir, 'label1', img_name.replace('.jpg', label_suffix))


class ImageDataset(data.Dataset):
    """VOCdataloder"""
    def __init__(self, root_dir, split='train'):
        super(ImageDataset, self).__init__()
        self.root_dir = root_dir
        self.split = split
        self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, self.split+'.txt')
        self.img_name_list = load_img_name_list(self.list_path)

        self.A_size = len(self.img_name_list)  # get the size of dataset A
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
            self.augm = CDDataAugmentation(img_size=256)
                    
    def __getitem__(self, index):
        pass
       
    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_size

class LEVIRDataset(ImageDataset):
    def __init__(self, root_dir, split, edge):
        super(LEVIRDataset, self).__init__(root_dir, split=split)
        self.edge = edge

    def __getitem__(self, index):
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.A_size])
        L_path = get_label_path(self.root_dir, self.img_name_list[index % self.A_size])
        E_path = get_edge_path(self.root_dir, self.img_name_list[index % self.A_size])

        img1 = np.asarray(Image.open(A_path).convert('RGB'))
        img2 = np.asarray(Image.open(B_path).convert('RGB'))
        label = np.array(Image.open(L_path), dtype=np.uint8)
        edge = np.array(Image.open(E_path), dtype=np.uint8)

       
        img1_original = img1
        img2_original = img2

      
        label = label//255
        edge = edge//255


        [img1, img2], [label,edge] = self.augm.transform([img1, img2], [label,edge], to_tensor=True, nomalization = [[0.5,0.5,0.5],[0.5,0.5,0.5]])
       

        if self.edge == True:
            if self.split == 'test':
                return img1_original, img2_original, img1, img2, label, edge
            else:
                return img1, img2, label, edge

        else:
            if self.split == 'test':
                return img1_original, img2_original, img1, img2, label
            else:
                return img1, img2, label