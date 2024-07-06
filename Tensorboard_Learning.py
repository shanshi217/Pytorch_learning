# -*- coding=utf-8 -*-
'''
**----------项目开发：山石---------------------------**
**----------开发时间：{2024/7/2}------------------------**
**----------单位：重庆大学----------------------------**
**----------地址：重庆市沙坪坝区沙正街174号重庆大学-------**
版权所有归开发人员
'''
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
# writer.add_image() 添加图像的函数
#绘制y = 2 * x
for i in range(101):
    writer.add_scalar("y = 2x",  2 * i, i)     #添加标量数据，参数：标题，标量值(y轴)，运行步数（x轴）

'''
需要注意的是，tensorboard是根据标题来区分不同的图像的，所以每次改变参数一定要重新写标题才能分开绘制，
否则就会绘制同名图像最新的那个，旧的不会绘制，而且运行一次生成一次log文件
'''

writer.close()                                    #关闭记录板

ants_img_path = 'datasets/hymenoptera_data/train/ants/9715481_b3cb4114ff.jpg'
ants_img = Image.open(ants_img_path)
print(ants_img.size)     #(宽，高)

ants_img_array = np.array(ants_img)    #将图片转换为numpy数组格式
print(ants_img_array.shape)       # (高，宽，通道数)

bees_img_path = 'datasets/hymenoptera_data/train/bees/90179376_abc234e5f4.jpg'
bees_img = Image.open(bees_img_path)
bees_img_array = np.array(bees_img)
'''
writer.add_image() 输入的图像格式是：img_tensor  -> (torch.Tensor, numpy.ndarray, or string/blobname)
img_tensor: Default is :math:`(3, H, W)`. (默认3通道，高，宽)这样的格式
            如果不一样就要修改参数dataformats(C:chanal通道，H：height高，W：weight：宽)
            convert a batch of tensor into 3xHxW format or call ``add_images`` and let us do the job.
            Tensor with :math:`(1, H, W)`, :math:`(H, W)`, :math:`(H, W, 3)` is also suitable as long as
            corresponding ``dataformats`` argument is passed, e.g. ``CHW``, ``HWC``, ``HW``.
'''
writer.add_image(tag="ants_img_test", img_tensor=ants_img_array, global_step=1, dataformats="HWC")

writer.add_image("bees_img_test", bees_img_array,2,dataformats="HWC")  #最后一个参数必须指定，不指定就会按照默认顺序，给错变量
writer.close()