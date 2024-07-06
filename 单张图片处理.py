# -*- coding=utf-8 -*-
'''
**----------项目开发：山石---------------------------**
**----------开发时间：{2024/7/4}------------------------**
**----------单位：重庆大学----------------------------**
**----------地址：重庆市沙坪坝区沙正街174号重庆大学-------**
版权所有归开发人员
'''
import torch
import torchvision.transforms
from PIL import Image
from torch import nn
from torch.nn import Linear

from torch.utils.tensorboard import SummaryWriter

img_path = '/home/zy/pycharm/pytorch_learning/datasets/hymenoptera_data/train/bees/132826773_dbbcb117b9.jpg'
img = Image.open(img_path)
print(type(img))

img = torchvision.transforms.Resize((300,300) )(img)      #将图片变换为方形300*300  此处的输入只能是PIL图片
img_tensor = torchvision.transforms.ToTensor()(img)



img_ = img_tensor.unsqueeze(0)   # 这个函数用来给tensor类型的数据的某一具体位置添加维度 ，例如这里就是添加batch_size维度

print(img_.shape)
img_ = torch.flatten(img_)   #查看多少个数字
print(img_.shape)

class Linear_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(270000,3072)

    def forward(self,input):
        output = self.linear1(input)
        return output

my_net = Linear_Net()

out = my_net(img_)
writer = SummaryWriter("log_test")
out = torch.reshape(out,[1,3,32,32])

writer.add_images("single_img",out,1)

writer.close()