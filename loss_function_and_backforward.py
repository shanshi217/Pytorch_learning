# -*- coding=utf-8 -*-
'''
**----------项目开发：山石---------------------------**
**----------开发时间：{2024/7/5}------------------------**
**----------单位：重庆大学----------------------------**
**----------地址：重庆市沙坪坝区沙正街174号重庆大学-------**
版权所有归开发人员
'''
from collections import OrderedDict
import torch
import torchvision
from torch import nn
from torch.nn import Flatten
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter



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


dataset = torchvision.datasets.CIFAR10('dataset',train=False,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,64)

MyNet = Cifar_net_new()
loss = nn.CrossEntropyLoss()  #定义交叉熵损失函数
step = 0

# for data in dataloader:
#     img , target = data
#     #print(img.shape)
#     print(target)
#     output = MyNet(img)
#     result_loss = loss(output,target)       #计算二者的交叉熵损失函数
#     print(result_loss)
#     #为了使用反向传播机制，需要对计算出的损失函数给出backward属性，才会附加梯度属性，用于后续的反向传播
#     result_loss.backward()

'''--------------优化器使用，根据反向传播+梯度优化------------------------'''

optim = torch.optim.SGD(MyNet.parameters(), lr= 1e-3,)       #随机下降优化器,第一个参数：网络模型参数

'''--------------使用下面for循环时候把上面那个可以注释掉---------'''
writer = SummaryWriter('log_loss')

for episode in range(100):
    #running_loss = 0.0  为了方便起见，也可以是使用每回合中的损失函数之和作为表征参数
    for data in dataloader:
        img , target = data
        #print(img.shape)
        #print(target)
        output = MyNet(img)
        result_loss = loss(output,target)       #计算二者的交叉熵损失函数
        #print(result_loss)
        '''这一步是必须的，因为为了迭代优化，每次优化完后需要重新计算梯度，这一步就是将上一步的梯度先清零'''
        optim.zero_grad()
        '''为了使用反向传播机制，需要对计算出的损失函数给出backward属性，才会附加梯度属性，用于后续的反向传播'''
        result_loss.backward()
        optim.step()                            #调用优化器优化参数
        # running_loss += result_loss           #同上running_loss定义哪里可以看到

        print(f"这是第{episode}回合的损失:",result_loss)
        writer.add_scalar("Loss",result_loss,episode)

writer.close()
