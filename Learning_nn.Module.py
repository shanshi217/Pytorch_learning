# -*- coding=utf-8 -*-
'''
**----------项目开发：山石---------------------------**
**----------开发时间：{2024/7/3}------------------------**
**----------单位：重庆大学----------------------------**
**----------地址：重庆市沙坪坝区沙正街174号重庆大学-------**
版权所有归开发人员
'''
import torch
from torch.nn import MaxPool2d, Linear
from torch.utils.tensorboard import SummaryWriter
# torch.nn as nn 是常用写法
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReLU, Sigmoid, Softmax


'''------------------------------------网络骨架 nn.Module---------------------------------------'''

class MyNet(nn.Module):          #每次搞自己的网络，nn.Module是必须被继承的,不然不能调用，会报错
    def __init__(self):
        super().__init__()        #初始化父类  #此处必须是super().__init__()

    def forward(self, input):       #前向传播方法每次必须重写
        layer1 = F.relu(input)      #定义每一层
        output = F.sigmoid(layer1)
        return output

mynet = MyNet()
input = torch.tensor(15.0)
output = mynet(input)
print(output)

'''-------------------------------------二维卷积函数-------------------------------------'''



#输入数据
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])


#卷积核
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])



'''
------------------------------------------------------------------------------------------------------------------------
                            layer1 = F.conv2d(input, kernel , stride=1)
        卷积操作需要的参数:
        1.input : 形状（minibatch, in_channels , iH, iW）: batch, 通道， 高， 宽
        2.weight/kernel： 形状 （out_channels, in_channels/groups, kH, kW）: 输出通道，输入通道（一般groups=1）, 高， 宽
        3.bias: 形状：（out_channels），需要tensor类型，大小和输出通道一致
        4.stride： 步进值，可以是(dH,dW)高方向，宽方向步进值元组，或者一个单独的数字，默认为1
------------------------------------------------------------------------------------------------------------------------
'''


print(input.shape)    #torch.Size([5, 5])  #与2D卷积函数输入要求不符，他需要四维数据
print(kernel.shape)   #torch.Size([3, 3])  #卷积核形状同样不符合要求

input = torch.reshape(input, (1, 1, 5, 5) )  #利用reshape方法重新搞形状， 这里形状（），[]都可以
print(input.shape)
kernel = torch.reshape(kernel, shape=[1, 1, 3, 3])
print(kernel.shape)

output1 = F.conv2d(input, kernel, stride= 1)
print(output1)
print(output1.shape)        #卷积后大小计算：N * N , N = ( ( W - F + 2*P) / stride ) + 1 )
                            # N:输出后矩阵维数， W 输入矩阵维数， F： 卷积核维数，
                            # P：padding:填充像素 ， 边缘填充后 相当于 W+2P * W+2P


output2 = F.conv2d(input, kernel, stride=2)
print(output2)
print(output2.shape)

output3 = F.conv2d(input, kernel, stride=1, padding=1)
print(output3)
print(output3.shape)


'''-------------------------------------卷积神经网络-------------------------------------'''
from torch.utils.data import DataLoader
import torchvision
test_set = torchvision.datasets.CIFAR10('dataset', train=False,
                                        transform=torchvision.transforms.ToTensor(),download=True)

test_set = DataLoader(test_set,batch_size=64,shuffle=True,num_workers=0,drop_last=False)

class MyCovnNet(nn.Module):
    def __init__(self):
        super(MyCovnNet,self).__init__()
        self.conv2d = nn.Conv2d(in_channels=3, out_channels=3,kernel_size=(3,3),stride=1,padding=0)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv2d(input)
        return output

writer = SummaryWriter('logs')
Myconvnet = MyCovnNet()
step = 0
for echo in range(2):
    for data in test_set:
        img, target = data
        output_conv = Myconvnet(img)
        writer.add_images("Conv_out", output_conv,step)
        step += 1
    print(f"这是第{echo}回合")

writer.close()

'''-------------------------------------最大池化---------------------------------'''

input_1 = torch.tensor([[1, 2, 0],
                             [0, 1, 2],
                             [1, 2, 1],
                             [5, 2, 3],
                             [1, 2, 1],
                             [1, 3, 1],
                             [5, 2, 8],
                             [4, 6, 9],
                             [6, 2, 3],],dtype=torch.float32)
#RuntimeError: Input type (long int) and bias type (float) should be the same
# 这里默认是long int类型，需要指定类型


'''最大池化input必须是（N，C，H，W)  所以需要reshape一下'''
input_1 = input_1.reshape(-1, 3, 3, 3)  #-1表示默认计算，也就是根据后面的数字自动计算大小


output_1 = F.max_pool2d(input_1,kernel_size=3, ceil_mode=False)
#采用Floor模式，池化不足池化层大小仍然保留已有的最大值

print(output_1)

output_2 = F.max_pool2d(input_1,kernel_size=3, ceil_mode=True)
#采用Ceil模式，池化不足池化层大小则不计算
print(output_2)

class Maxpool_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = MaxPool2d(kernel_size=2,ceil_mode=False)
        self.conv2d = nn.Conv2d(3, 1, 3, padding=1)

    def forward(self, input):
        input = self.conv2d(input)
        output = self.maxpool(input)
        return output

maxpool_net = Maxpool_Net()

print(maxpool_net(input_1))

'''----------------------------------------非线性激活 ReLu-----------------------------------------------'''

input_non_liner = torchvision.datasets.CIFAR10("dataset",train=False,
                                               transform=torchvision.transforms.ToTensor(),download=True)

input_non_liner = DataLoader(input_non_liner,batch_size=64, shuffle=True)

class Non_Liner_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = ReLU(inplace=False)
        self.sigmod = Sigmoid()
        self.softmax = Softmax()

    def forward(self,idx):
        output_non_liner = self.sigmod(idx)
        return output_non_liner


non_liner_net = Non_Liner_Net()

writer = SummaryWriter('Log_sigmod')
step = 0
for data in input_non_liner:
    img, target = data                  #这里需要data赋值
    writer.add_images("Orgin_pic", img, global_step=step)
    output_non_liner = non_liner_net(img)
    writer.add_images("Output_pic",output_non_liner,step)
    step += 1

writer.close()

'''--------------------------------------正则化---------------------------------------------'''
'''-----------------------正则化的目的是为了加快神经网络训练的速度，实现快速的训练---------------------'''
'''------具体用法可以查看pytorch官网文档：https://pytorch.org/docs/stable/nn.html#normalization-layers ---'''


'''-------------------------------------RNN/LSTM 循环层-------------------------------------------'''
'''------------------------------------用于构建循环神经网络-----------------------------------------'''
'''-----------具体用法：https://pytorch.org/docs/stable/nn.html#recurrent-layers ------------------'''

'''-------------------------------------Transformer层---------------------------------------------'''
'''------------------- https://pytorch.org/docs/stable/nn.html#transformer-layers ----------------'''

'''---------------------------------------线性层 nn.Leaner---------------------------------------'''
'''--------------------------------线性层是对输入做一个线性变换---------------------------------------'''


'''-----------------------------------Dropout Layers nn.Dropout----------------------------------'''
'''-------------------------训练过程中个随机按照概率p将一部分输入元素的tensor变为0------------------------'''
'''----------------------------作用像是让神经网络中的部分神经元失活，来防止过拟合--------------------------'''

'''-----------------------------Distance Function: 计算两值之间的距离（误差）---------------------------'''

'''---------------------------------Loss Function: 损失函数  例如MSE等----------------------------------'''













