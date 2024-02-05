# from torchvision.transforms import ToTensor #用于把图片转化为张量
import transforms as transforms
from data import MyDataset
import numpy as np #用于将张量转化为数组，进行除法

Dataroot = 'data/TACE2_LS_jpg'
# Dataroot = 'data/TACE2_HZF_jpg74'
# Dataroot = 'data/TACE2_ZJUF_jpg'
# Dataroot = 'data/TACE2_ZJUF_jpg189'
# Dataroot = 'heatmap_model/plt_img'

# Dataroot = 'data/LIHC-MRI-jpg-label'

# #初始化均值和方差 RGB图
# means = [0.0, 0.0, 0.0]
# stds = [0.0, 0.0, 0.0]
# #可将图片类型转化为张量，并把0~255的像素值缩小到0~1之间
# transform_test = transforms.Compose([
# #    transforms.Resize((72, 72)),
#     transforms.CenterCrop(64),
#     transforms.ToTensor()])
# #导入数据集的图片，并且转化为张量
# dataset = MyDataset(split = 'train', data_dir=Dataroot, transform=transform_test, ratio=1.0)
# num_imgs=len(dataset)
# #遍历数据集的张量和标签
# for img, lab in dataset:
#     # 计算每一个通道的均值和标准差
#     # print("img:",img.shape)
#     # exit(0)
#     means[0] += img[0, :, :].mean()
#     stds[0] += img[0, :, :].std()
#     means[1] += img[1, :, :].mean()
#     stds[1] += img[1, :, :].std()
#     means[2] += img[2, :, :].mean()
#     stds[2] += img[2, :, :].std()
# #要使数据集归一化，均值和方差需除以总图片数量
# mean = np.array(means)/num_imgs
# std = np.array(stds)/num_imgs
# print("mean, std:",mean, std)

#灰度图
#初始化均值和方差
means = 0.0
stds = 0.0
#可将图片类型转化为张量，并把0~255的像素值缩小到0~1之间
transform_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    # transforms.Resize((136, 136)),
    # transforms.CenterCrop(128),
    transforms.ToTensor()])
#导入数据集的图片，并且转化为张量
dataset = MyDataset(split = 'train', data_dir=Dataroot, transform=transform_test, ratio=1.0)
num_imgs=len(dataset)
#遍历数据集的张量和标签
for img, lab in dataset:
    # 计算每一个通道的均值和标准差
    # print("img:",img.shape)
    means += img[:, :].mean()
    stds += img[:, :].std()
#要使数据集归一化，均值和方差需除以总图片数量
mean = np.array(means)/num_imgs
std = np.array(stds)/num_imgs
print("mean, std:",mean, std)

