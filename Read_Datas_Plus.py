# -*- coding=utf-8 -*-
'''
**----------项目开发：山石---------------------------**
**----------开发时间：{2024/7/1}------------------------**
**----------单位：重庆大学----------------------------**
**----------地址：重庆市沙坪坝区沙正街174号重庆大学-------**
版权所有归开发人员
'''
from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):

    def __init__(self,root_dir,data_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.data_dir = data_dir
        self.data_path = os.path.join(self.root_dir, self.data_dir)
        self.label_path = os.path.join(self.root_dir, self.label_dir)
        self.imgs = os.listdir(self.data_path)
        self.labels = os.listdir(self.label_path)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        label_name = self.labels[idx]
        img_item_path = os.path.join(self.root_dir, self.data_dir, img_name)
        label_item_path = os.path.join(self.root_dir, self.label_dir, label_name)
        img = Image.open(img_item_path)
        with open(label_item_path, 'r') as f:
            label = f.read()
        return img, label

    def __len__(self):

       #return len(self.imgs), len(self.labels)
        return len(self.imgs) #为了避免无法合并的问题，只返回一个长度，一般来说 数据和label是对应的，二者一样大

root_dir = 'datasets/dataset_prac/train'
ants_data_dir = 'ants_image'
ants_label_dir = 'ants_label'
bees_data_dir = 'bees_image'
bees_label_dir = 'bees_label'

ants_dataset = MyData(root_dir, ants_data_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_data_dir, bees_label_dir)

train_datasets = ants_dataset + bees_dataset
# 如果函数具有多个返回值时，python的自动打包机制会将其打包成元组输出，此时交互赋值是没问题的，
# return len(self.imgs), len(self.labels)
# 但是调用Dataset.ConcatDataset连接（也就是 + ）时候，就会出现问题
# 在 PyTorch 的 Dataset 类中，__len__ 方法应该返回一个整数，
# 表示数据集的长度。返回元组会导致使用 ConcatDataset 时出错，
# 因为 ConcatDataset 期望 __len__ 方法返回一个整数。

ants_img, ants_label = ants_dataset[1]
ants_img.show()
print(ants_label)

bees_img, bees_label = bees_dataset[0]
bees_img.show()
print(bees_label)