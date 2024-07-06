# -*- coding=utf-8 -*-
'''
**----------项目开发：山石---------------------------**
**----------开发时间：{2024/7/3}------------------------**
**----------单位：重庆大学----------------------------**
**----------地址：重庆市沙坪坝区沙正街174号重庆大学-------**
版权所有归开发人员
'''

#dataloader是数据集导入到神经网络前的载入器
#作用是指定选择的数据集、batch_size,
# 是否随机打乱数据shuffle = True 每一个回合打乱数据，
# 是否多线程num_workers = 0 则只使用主进程，
# 以及数据集不能按照batch_size取整时是否载入余下的数据 drop_last = True 就是不载入剩下不足一个batch_size的数据

import torchvision

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10('./dataset',False,transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(test_data,64,True,num_workers=0,drop_last=True)


'''此时，test_data返回值是两个，一个是numpy array类型的图片 10000张*32*32*3，另一个是标签数字 '''
test0 = test_data[0]  #具体调用的时候才会执行totensor函数

img, target = test_data[1]
print(test_data.classes[target])

writer = SummaryWriter('dataloader')

#为了体现每回合随机打乱shuffle，可以跑多个回合epchs

for epoch in range(2):

    time_steps = 0
    for data  in  test_loader:     #DataLoader可以迭代是因为他将数据分为批次，每一个批次可以迭代一次
                                   # 迭代625次是因为 10000//16 = 625
                                   # 重新改为batch_size = 64  10000//64 = 156
        img_tensor, target = data
        writer.add_images("第{}回合".format(epoch), img_tensor, time_steps)
                                   #批次图片需要合并为一张，可以使用torchvision.utils.make_grid()手动转换
                                   #可以以使用add_images来自动合并，一定要注意批次图片要用images!!! 加s!!
        time_steps += 1

writer.close()