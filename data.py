# -*- coding:utf-8 -*-
import numpy as np
import os
import torch
from PIL import Image
import torch.utils.data as data

# Custom dataset
class MyDataset(data.Dataset):
    def __init__(self, split='train', data_dir=None, transform=None, ratio=0.8): # 0.7
        self.split = split
        img_list = os.listdir(data_dir)
        # img_name_list = [img_name.split('.')[0][:-2] for img_name in img_list]
        # img_list = list(set(img_name_list))
        # img_list.sort(key=img_name_list.index)
        trainset_count = int(len(img_list) * ratio)
        np.random.seed(3) # 0
        np.random.shuffle(img_list)

        if self.split == 'train':
            train_img_list = img_list[:trainset_count]
            train_img_label = [int(img_name.split('_')[1]) for img_name in train_img_list]
            imgs = zip(train_img_list, train_img_label)

        elif self.split == 'test':
            test_img_list = img_list[trainset_count:]
            test_img_label = [int(img_name.split('_')[1]) for img_name in test_img_list]
            imgs = zip(test_img_list, test_img_label)
        elif self.split == 'val':
            val_img_label = [int(img_name.split('_')[1]) for img_name in img_list]
            imgs = zip(img_list, val_img_label)
            # with open('data/val_set_cnn_TCGA.csv', 'w') as f_val: f_val.write('patient_name \n')
            # with open('data/val_set_cnn_TCGA.csv', 'a') as f_val:
            #     val_img_name = [img_name.split('_')[0] for img_name in img_list]
            #     val_patient = zip(val_img_name, val_img_label)
            #     for val_pat in val_patient: f_val.write('%s,%s\n' %(val_pat))
        else:
            print('Error input, if shuld be train or test!')
            exit(0)

        self.imgs = list(imgs)
        self.data_dir = data_dir
        self.transforms = transform

    def __getitem__(self, index):
        img_name, label = self.imgs[index]
        img = Image.open(self.data_dir + '/' + img_name).convert('RGB')  # convert('RGB')
        img = self.transforms(img)
        # img_A = Image.open(self.data_dir + '/' + img_name + '_A.jpg').convert('L')  # convert('RGB')
        # img_T = Image.open(self.data_dir + '/' + img_name + '_T.jpg').convert('L')
        # img_A = self.transforms(img_A).squeeze()
        # img_T = self.transforms(img_T).squeeze()
        # img = torch.stack([img_A, img_A, img_T], dim=0)
        # torch.set_printoptions(profile="full")
        # img = Image.open(self.data_dir + '/' + img_name).convert('L')  # .convert('RGB'),.convert('L')

        return img, label, img_name.split('.')[0]

    def __len__(self):
        return len(self.imgs)
