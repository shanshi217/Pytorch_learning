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
transform一般用于torchvision中，进行图像变换，最常用的是将图片格式转换为Tensor格式
'''
from torchvision import transforms
from PIL import Image

img_path = 'datasets/hymenoptera_data/train/bees/198508668_97d818b6c4.jpg'
img = Image.open(img_path)
print(img)

'''
ToTenser方法使用方法:
1. 创建transforms里面的类实体 (也就是类里面的__init__方法进行初始化)
2. 把对应的参数赋值给类实体 调用对应的方法(也就是类里面的call方法)
'''
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

print(tensor_img.shape)     #CHW
print(tensor_img)

'''
#针对PIL图片也可以使用下面函数转换
tensor_trans1 = transforms.PILToTensor()
tensor_img1 = tensor_trans1(img)
print(tensor_img1)
print(tensor_img1.shape)
'''

'''
为什么需要用tensor数据类型？
通过控制台可以看到，tensor类型为数据赋予了很多神经网络需要的性质，例如梯度方法，反向传播机制等等，而原始的RGB图像是不具备这些特点的
'''
# opencv库读取图片
import cv2
cv_img = cv2.imread(img_path)
print(cv_img)
print(type(cv_img))   #<class 'numpy.ndarray'>  直接将图片读取后转换为numpy数组
print(cv_img.shape)   #HWC


writer = SummaryWriter("logs")

writer.add_image("Tensor_img", cv_img, 1, dataformats="HWC")   #单张图片使用add_image即可

writer.close()