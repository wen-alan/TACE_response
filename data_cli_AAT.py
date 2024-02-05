# -*- coding:utf-8 -*-
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torch.utils.data as data
from sklearn.preprocessing import scale, StandardScaler

train_file_name = "data/TACE2_LS_210.csv"
# val_file_name = "data/TACE2_HZF_94.csv"
val_file_name = "data/TACE2_HZF_86.csv"
# val_file_name = "data/TACE2_ZJU_292.csv"
# val_file_name = "data/TACE2_ZJU_180.csv"

# Custom dataset
class MyDataset(data.Dataset):
    def __init__(self, split='train', data_dir=None, transform=None, ratio=0.8):
        self.split = split
        img_list = os.listdir(data_dir)
        img_name_list = [img_name.split('.')[0][:-2] for img_name in img_list]
        img_list = list(set(img_name_list))
        img_list.sort(key=img_name_list.index)
        trainset_count = int(len(img_list) * ratio)
        np.random.seed(3) #3 5 6 1234
        np.random.shuffle(img_list)
        #clinical data
        train_set = pd.read_csv(train_file_name)
        val_set = pd.read_csv(val_file_name)
        train_set_data = train_set.drop(['Patients', 'Name', 'Response'], axis=1)
        val_set_data = val_set.drop(['Patients', 'Name', 'Response'], axis=1)
        train_set_name = list(train_set['Patients'])
        val_set_name = list(val_set['Patients'])
        # z-score
        # scaler = StandardScaler().fit(train_set_data)
        # train_set_data = scaler.transform(train_set_data)
        # val_set_data = scaler.transform(val_set_data)
        train_set_data = scale(train_set_data)
        val_set_data = scale(val_set_data)
        train_set_data = torch.Tensor(train_set_data)
        val_set_data = torch.Tensor(val_set_data)

        if self.split == 'train':
            train_img_list = img_list[:trainset_count]
            train_img_label = [int(img_name.split('_')[1]) for img_name in train_img_list]
            imgs = zip(train_img_list, train_img_label)
            cli_patient_name = train_set_name
            cli_patient_data = train_set_data
            # with open('data/train_set_name.csv', 'w') as f_train: f_train.write('patient_name, label \n')
            # with open('data/train_set_name.csv', 'a') as f_train:
            #     train_img_name = [img_name.split('_')[0] for img_name in train_img_list]
            #     train_patient = zip(train_img_name, train_img_label)
            #     for train_pat in train_patient: f_train.write('%s,%s\n' %(train_pat))
        elif self.split == 'test':
            test_img_list = img_list[trainset_count:]
            test_img_label = [int(img_name.split('_')[1]) for img_name in test_img_list]
            imgs = zip(test_img_list, test_img_label)
            cli_patient_name = train_set_name
            cli_patient_data = train_set_data
            # with open('data/test_set_name.csv', 'w') as f_test: f_test.write('patient_name, label \n')
            # with open('data/test_set_name.csv', 'a') as f_test:
            #     test_img_name = [img_name.split('_')[0] for img_name in test_img_list]
            #     test_patient = zip(test_img_name, test_img_label)
            #     for test_pat in test_patient: f_test.write('%s,%s\n' %(test_pat))
        elif self.split == 'val':
            val_img_label = [int(img_name.split('_')[1]) for img_name in img_list]
            imgs = zip(img_list, val_img_label)
            # with open('data/val_set_name_ZJ189.csv', 'w') as f_val: f_val.write('patient_name, label \n')
            # with open('data/val_set_name_ZJ189.csv', 'a') as f_val:
            # with open('data/val_name_HZF74.csv', 'w') as f_val: f_val.write('patient_name, label \n')
            # with open('data/val_name_HZF74.csv', 'a') as f_val:
            #     val_img_name = [img_name.split('_')[0] for img_name in img_list]
            #     val_patient = zip(val_img_name, val_img_label)
            #     for val_pat in val_patient: f_val.write('%s,%s\n' %(val_pat))
            cli_patient_name = val_set_name
            cli_patient_data = val_set_data
        else:
            print('Error input, if shuld be train or test!')
            exit(0)

        self.imgs = list(imgs)
        self.data_dir = data_dir
        self.transforms = transform
        self.cli_patient_name = cli_patient_name
        self.cli_patient_data = cli_patient_data

    def __getitem__(self, index):
        img_name, label = self.imgs[index]
        img_A = Image.open(self.data_dir + '/' + img_name + '_A.jpg').convert('L') # convert('RGB')
        img_T = Image.open(self.data_dir + '/' + img_name + '_T.jpg').convert('L')
        img_A = self.transforms(img_A).squeeze()
        img_T = self.transforms(img_T).squeeze()
        img = torch.stack([img_A, img_A, img_T], dim=0)
        # img = 0
        cli_name = (img_name.split('_')[0])
        cli_data = self.cli_patient_data[self.cli_patient_name.index(cli_name)]
        return img, label, cli_data

    def __len__(self):
        return len(self.imgs)
