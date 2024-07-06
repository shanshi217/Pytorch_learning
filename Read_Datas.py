'''
**----------项目开发：山石---------------------------**
**----------开发时间：{2024/7/1}------------------------**
**----------单位：重庆大学----------------------------**
**----------地址：重庆市沙坪坝区沙正街174号重庆大学-------**
版权所有归开发人员
'''
from torch.utils.data import Dataset
# torch.utils是torch基本工具库，里面的Data是数据相关
# import cv2  也是调用查看图片的库，此处没有安装
from PIL import Image   #调用查看图片的库里的函数
import os               #系统包，一般用于导入文件过程中文件路径使用

# img_path = '/home/zy/xiaotuidui/datasets/hymenoptera_data/train/ants/0013035.jpg'
# img = Image.open(img_path)
# img.size    #查看图片属性中的图片大小
# img.show()  #查看图片

class MyData(Dataset):   #Dataset使用方法中明确规定必须被继承

    def __init__(self,root_dir,lable_dir):  #初始化变量，为下面方法做准备 一般会有根目录，标签目录，使用join函数连接方便跨平台使用
        #super().__init__(MyData)
        self.root_dir = root_dir
        self.lable_dir = lable_dir
        self.path = os.path.join(self.root_dir, self.lable_dir)  #利用这个函数连接构成完整的数据地址
        self.img_path = os.listdir(self.path)                    #将全部地址获取后构成列表

    def __getitem__(self, idx):  #此处一般会把Dataset类中方法重写，获取数据索引
        img_name = self.img_path[idx]
        img_itme_path = os.path.join(self.root_dir, self.lable_dir, img_name)
        img = Image.open(img_itme_path)
        lable = self.lable_dir
        return img, lable

    def __len__(self):

        return len(self.img_path)


root_dir = 'datasets/hymenoptera_data/train'
ants_label_dir = 'ants'
bees_label_dir = 'bees'

ants_dataset = MyData(root_dir,ants_label_dir)
bee_dataset = MyData(root_dir,bees_label_dir)

train_dataset = ants_dataset + bee_dataset   #将两个数据集合并，得到一个并集，但是内部二者相互独立

print(ants_dataset.__len__())
print(bee_dataset.__len__())
  #  上面两个相加就是下面这个
print(train_dataset.__len__())

img, label = train_dataset[0]
img.show()


img, label = train_dataset[123]
img.show()                              #按照顺序，前124个事ants,后面是bees

img, label = train_dataset[124]
img.show()

