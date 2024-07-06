# -*- coding=utf-8 -*-
'''
**----------项目开发：山石---------------------------**
**----------开发时间：{2024/7/4}------------------------**
**----------单位：重庆大学----------------------------**
**----------地址：重庆市沙坪坝区沙正街174号重庆大学-------**
版权所有归开发人员
'''
import torch
import torch.nn as nn
import torchvision
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

pic_data = torchvision.datasets.CIFAR10('dataset',train=False,transform=torchvision.transforms.ToTensor(),download=True)

pic_loader = DataLoader(pic_data,batch_size=1,drop_last=True)


class NN_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(196608,4800)   #in_features:输入量，out_features:输出量
    def forward(self,input):
        output = self.linear1(input)
        return output



nn_net = NN_Net()

writer = SummaryWriter('logs_linear')
step = 0
for data in pic_loader:
    img, target = data
    #print(img.shape)
    #图像不满足线性层的使用规则，为了压缩图片，需要将其铺平然后线性层压缩
    writer.add_images("orgin_pic",img,step)
    img = torch.flatten(img)  #将图片拉平的函数，与torch.reshape(img, (1,1,1,-1)）作用一样
    # print(img.shape)
    output = nn_net(img)
    #print(output.shape)
    output = torch.reshape(output,[1,3,5,5])
    writer.add_images("output_img",output,step)
    step += 1

writer.close()








