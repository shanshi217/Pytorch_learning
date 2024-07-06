# -*- coding=utf-8 -*-
'''
**----------项目开发：山石---------------------------**
**----------开发时间：{2024/7/2}------------------------**
**----------单位：重庆大学----------------------------**
**----------地址：重庆市沙坪坝区沙正街174号重庆大学-------**
版权所有归开发人员
'''
from torch.utils.tensorboard import SummaryWriter

'''
--------------------------------------常用的transform函数-----------------------------------------
'''

import torchvision

import torchvision.transforms

import torchvision.utils   #使用tensorboard


'''为了后续神经网络使用，需要将PIL图片转换为tensor格式，使用transform工具转化'''
dataset_transform = torchvision.transforms.Compose([    #compose函数，集成多个操作，输入为[]
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=dataset_transform, download=True)

# print(train_set[0])
# print(test_set[0])     #两个返回值，一个是图片格式和大小，另一个是目标分类，给的是label的数字表示
#
#
# img, target = test_set[0]
# print(img)
# print(target)
#
# print(test_set.classes[target])   #查看具体的分类 即label的具体内容
# img.show()

writer = SummaryWriter("Datasets_using")

for i in range(15):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close() #这个不能放在循环里，否则只能记录一次