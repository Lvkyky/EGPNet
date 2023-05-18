# coding=utf-8
from os import listdir
from os.path import join
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import numpy as np
import torchvision.transforms as transforms
import os

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.tif', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])
def calMetric_iou(predict, label):
    tp = np.sum(np.logical_and(predict == 1, label == 1))
    fp = np.sum(predict == 1)
    fn = np.sum(label == 1)
    return tp, fp + fn - tp
def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)
def display_transform():
    return Compose([
        ToPILImage(),
        Resize(64),
        CenterCrop(64),
        ToTensor()
    ])
def getSampleLabel(img_path):
    img_name = img_path.split('\\')[-1]
    return torch.from_numpy(np.array([int(img_name[0] == 'i')], dtype=np.float32))
def getDataList(img_path):
    dataline = open(img_path, 'r').readlines()
    datalist = []
    for line in dataline:
        temp = line.strip('\n')
        datalist.append(temp)
    return datalist
def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)
    return result
def get_transform(convert=True, normalize=False):
    transform_list = []
    if convert:
        transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

class LoadDatasetFromFolder(Dataset):
    def __init__(self, hr1_path, hr2_path, lab_path, edge_path, ori=False,edge=False):
        super(LoadDatasetFromFolder, self).__init__()
        # 获取图片列表
        datalist = [name for name in os.listdir(hr1_path)]

        self.hr1_filenames = [join(hr1_path, x) for x in datalist if is_image_file(x)]
        self.hr2_filenames = [join(hr2_path, x) for x in datalist if is_image_file(x)]
        self.lab_filenames = [join(lab_path, x) for x in datalist if is_image_file(x)]
        self.edge_filenames = [join(edge_path, x) for x in datalist if is_image_file(x)]

        self.transform = get_transform(convert=True, normalize=True)  # convert to tensor and normalize to [-1,1]
        self.label_transform = get_transform()  # only convert to tensor
        self.ori = ori
        self.edge = edge

    def __getitem__(self, index):
        ori1 = np.asarray(Image.open(self.hr1_filenames[index]).convert('RGB'))
        ori2 = np.asarray(Image.open(self.hr2_filenames[index]).convert('RGB'))
        hr1_img = self.transform(ori1)
        hr2_img = self.transform(ori2)
        label = self.label_transform(Image.open(self.lab_filenames[index]))
        edge = self.label_transform(Image.open(self.edge_filenames[index]))

        if self.edge == False:

            if self.ori == True:
                return ori1,ori2, hr1_img, hr2_img, label

            else:
                return hr1_img, hr2_img, label

        else:
            if self.ori == True:
                return ori1, ori2, hr1_img, hr2_img, label,edge
            else:
                return hr1_img, hr2_img, label,edge


    def __len__(self):
        return len(self.hr1_filenames)
class TestDatasetFromFolder(Dataset):
    def __init__(self, Time1_dir, Time2_dir, Label_dir, edge_dir, image_sets):
        super(TestDatasetFromFolder, self).__init__()
        self.image1_filenames = [join(Time1_dir, x) for x in image_sets if is_image_file(x)]
        self.image2_filenames = [join(Time2_dir, x) for x in image_sets if is_image_file(x)]
        self.image3_filenames = [join(Label_dir, x) for x in image_sets if is_image_file(x)]
        self.image4_filenames = [join(edge_dir, x) for x in image_sets if is_image_file(x)]

        self.transform = get_transform(convert=True, normalize=True)  # convert to tensor and normalize to [-1,1]
        self.label_transform = get_transform()

    def __getitem__(self, index):
        image1 = self.transform(Image.open(self.image1_filenames[index]).convert('RGB'))
        image2 = self.transform(Image.open(self.image2_filenames[index]).convert('RGB'))
        label = self.label_transform(Image.open(self.image3_filenames[index]))
        edge = self.label_transform(Image.open(self.image4_filenames[index]))

        return image1, image2, label, edge

    def __len__(self):
        return len(self.image1_filenames)











