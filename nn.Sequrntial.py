# -*- coding=utf-8 -*-
'''
**----------项目开发：山石---------------------------**
**----------开发时间：{2024/7/5}------------------------**
**----------单位：重庆大学----------------------------**
**----------地址：重庆市沙坪坝区沙正街174号重庆大学-------**
版权所有归开发人员
'''
from collections import OrderedDict

from torch.nn import Flatten
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torchvision.transforms

'''sequential 是 torch容器的一种，跟nn.Moudeul是同一级别，是把多个模型整合在一起的容器
例如：
Cifar10_model = nn.Sequential(
                nn.Conv2d(3,32,5, padding=2),  #卷积3@32*32 -> 32@32*32
                # padding = (kernel_size - 1) // 2 实现卷积/前后大小不变，此时kernel_size只能是奇数
                # padding = (kernel_size * 2 + 1) // 2 则不限制奇偶数
                nn.MaxPool2d(2,stride=2),                            #最大池化32@32*32 -> 32@16*16
                nn.Conv2d(32,32,5, padding=2), #卷积32@16*16 -> 32@16*16
                nn.MaxPool2d(2,stride=2),                            #最大池化32@16*16 -> 32@8*8
                nn.Conv2d(32,64,5, padding=2),  #卷积32@8*8 -> 64@8*8
                nn.MaxPool2d(2,stride=2),                             #最大池化64@8*8 -> 64@4*4
                )
'''

''' 与OrderedDict函数一起使用可以是这样的形式，上下两个模型是一样的'''
'''
Cifar10_model = nn.Sequential(OrderedDict([
    ('conv1',nn.Conv2d(3,32,5, padding=2)),
    ('max_pool',nn.MaxPool2d(2,stride=2)),
    ('conv2',nn.Conv2d(32,32,5, padding=2)),
    ('max_pool2',nn.MaxPool2d(2,stride=2)),
    ('conv3',nn.Conv2d(32,64,5, padding=2)),
    ('max_pool3', nn.MaxPool2d(2,stride=2))
    ]))
'''


'''   ------------用法-------------'''

#如果不用sequrntel网络搭建比较繁琐，不易阅读
class Cifar_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,5, padding=2)
        self.maxpool1 = nn.MaxPool2d(2)  #默认stride = kernel_size = 2
        self.conv2 = nn.Conv2d(32,32,5, padding=2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32,64,5, padding=2)
        self.maxpool3 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1024,64)
        self.linear2 = nn.Linear(64, 10)

    def forward(self,input):
        layer1 = self.conv1(input)
        layer2 = self.maxpool1(layer1)
        layer3 = self.conv2(layer2)
        layer4 = self.maxpool2(layer3)
        layer5 = self.conv3(layer4)
        layer6 = self.maxpool3(layer5)
        flatten = self.flatten(layer6)              #如果不知道，可以就写到这里，自动计算拉伸后多长
        layer7 = self.linear1(flatten)
        output = self.linear2(layer7)
        return output



cifar = Cifar_net()
print(cifar)

# 测试数据，手动编写，torch库可以随机生成
test_data = torch.ones([64, 3, 32, 32],dtype=torch.float32)  # batch_size = 64, data = 3 @ 32 * 32

output_test = cifar(test_data)
#print(output_test)
print(output_test.shape)

class Cifar_net_new(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(OrderedDict([
                    ('conv1',nn.Conv2d(3,32,5, padding=2)),
                    ('max_pool',nn.MaxPool2d(2,stride=2)),
                    ('conv2',nn.Conv2d(32,32,5, padding=2)),
                    ('max_pool2',nn.MaxPool2d(2,stride=2)),
                    ('conv3',nn.Conv2d(32,64,5, padding=2)),
                    ('max_pool3', nn.MaxPool2d(2,stride=2)),
                    ('flatten',Flatten()),
                    ('linear1', nn.Linear(1024, 64)),
                    ('linear2', nn.Linear(64, 10))
                    ]))

    def forward(self,input):
        output = self.model(input)
        return output

cifar_new = Cifar_net_new()
print(cifar_new)

output_test_new = cifar_new(test_data)
#print(output_test_new)
print(output_test_new.shape)

#  可以用tensorboard查看计算图了解网络结构

writer = SummaryWriter('log_seq')
writer.add_graph(cifar_new,test_data)
writer.close()
