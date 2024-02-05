'''Train with PyTorch.'''
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from __future__ import print_function
import os, argparse, utils, random, torch, torchvision
from models import *
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import torch.nn.functional as F
import transforms as transforms
# from BC_data import MyDataset
from data_cli_AAT import MyDataset
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--model', type=str, default='Resnet18', help='CNN architecture, Resnet18')
parser.add_argument('--dataset', type=str, default='TACE', help='CNN architecture')
parser.add_argument('--bs', default=32, type=int, help='batch size')
parser.add_argument('--lr', default=0.0005, type=float, help='learning rate') # 0.01, 0.005
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--save_dir', type=str, default='result_model', help='save log and model')
opt = parser.parse_args()

Dataroot = 'data/TACE2_LS_jpg'
# Dataroot_val = 'data/TACE2_ZJUF_jpg'
# Dataroot_val = 'data/TACE2_ZJUF_jpg189'
Dataroot_val = 'data/TACE2_HZF_jpg74'

use_cuda = torch.cuda.is_available()
best_Test_auc_epoch = 0; best_Test_acc = 0; best_Test_auc = 0  # best PublicTest accuracy
best_val_auc_epoch = 0; best_val_acc = 0; best_val_auc = 0  # best Private accuracy
prob_all_tr = []; label_all_tr = []
prob_all_test = []; label_all_test = []

start_epoch = 0  # start from epoch 0 or last checkpoint epoch
total_epoch = 20

path = os.path.join(opt.save_dir)
if not os.path.isdir(path): os.mkdir(path)
results_log_csv_name = 'log_CNNRes50_S5_9_DaS3lr5e4Adam_ABriCt_im128_HZF74.csv' #Da表示data，RF表示随机翻转
valset_pre_score = 'data/preScoreS5_9_CNNRes50_DaS3lr5e4Adam_ABriCt_im128_HZF74.csv'
# results_log_csv_name = 'log_CNN_DaS3lr5e4Adam_ABriCt_im128_ZJ292.csv' #Da表示data，RF表示随机翻转
# valset_pre_score = 'data/preScoreSx_CNN_DaS3lr5e4Adam_ABriCt_im128_ZJ292.csv'

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize((136, 136)),
    transforms.RandomCrop(128),
    transforms.ColorJitter(brightness=0.5,contrast=0.5),
    # transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    transforms.ToTensor(), #AAT [0.22649573,0.30327198,0.30260414], std=[0.11754133,0.12226411,0.12168134]
    transforms.Normalize(mean=[0.2701], std=[0.1390])]) #rs128 resize72 jpg [0.2651], std=[0.1188] LS L

transform_test = transforms.Compose([
    transforms.Resize((136, 136)),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2701], std=[0.1390])])  #rs128 resize72 [0.2716], std=[0.1216] LS_HZF

transform_val = transforms.Compose([# HZF [0.3011], std=[0.1644])]) #72
    transforms.Resize((136, 136)),#im128 [0.2494], std=[0.1329])]) #ZJ189
    transforms.CenterCrop(128),# [0.2475], std=[0.1316])]) #ZJ292
    transforms.ToTensor(), #ZJUF 190 [0.2498], std=[0.1305] im128 [0.3026], std=[0.1656])]) #78 HZF
    transforms.Normalize(mean=[0.2985], std=[0.1650])]) # 74 HZF  86 HZF [0.2990], std=[0.1630]

#train and test data
train_data = MyDataset(split = 'train', data_dir=Dataroot, transform=transform_train)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=opt.bs, num_workers=1, shuffle=True, drop_last=True)
test_data = MyDataset(split = 'test', data_dir=Dataroot, transform=transform_test)
testloader = torch.utils.data.DataLoader(test_data, batch_size=16, num_workers=1, shuffle=False)
#external val data
exter_val_data = MyDataset(split = 'val', data_dir=Dataroot_val, transform=transform_val)
exter_valloader = torch.utils.data.DataLoader(exter_val_data, batch_size=16, num_workers=1, shuffle=False)
# exit(0)
#create new model
class Densenet_Cli(nn.Module):
    def __init__(self, num_classes=2):
        super(Densenet_Cli, self).__init__()
        # self.net = torchvision.models.resnet18(pretrained=False) #resnet18
        # self.net.load_state_dict(torch.load('./result_model/resnet18.pth'))
        # self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)
        self.net = torchvision.models.resnet50(pretrained=False) #resnet50
        self.net.load_state_dict(torch.load('./result_model/resnet50.pth'))
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)
        # FC_size = 64
        # self.net = torchvision.models.densenet169(pretrained=True) #densenet169
        # self.net.classifier = nn.Linear(self.net.classifier.in_features, num_classes)
        # self.net.classifier = nn.Sequential(nn.Dropout(p=0.1, inplace=False), nn.Linear(self.net.classifier.in_features, 2))

    def forward(self, img, cli):
        out = self.net(img)
        return out

softmax_func = nn.Softmax(dim=1)

if opt.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(path,'Test_model.t7'))
    net.load_state_dict(checkpoint['net'])
    best_Test_acc = checkpoint['acc']
    best_Test_auc_epoch = checkpoint['epoch']
    start_epoch = checkpoint['epoch'] + 1
    for x in range(start_epoch): scheduler.step()
else: print('==> Building model..')
if use_cuda: net = net.cuda()

def initialize_torch_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)  # Python random module.
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    print('seed:', torch.initial_seed())


def print_roc(fpr,tpr,thresholds):
    print('fpr:')
    for value in fpr: print(value)
    print('tpr:')
    for value in tpr: print(value)
    print('thresholds:')
    for value in thresholds: print(value)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    global Train_acc, Train_auc, prob_all_tr, label_all_tr
    net.train()
    train_loss, correct, total = 0, 0, 0
    prob_all_tr = [];label_all_tr = []
    print('learning_rate: %s' % str(scheduler.get_lr()))
    for batch_idx, (inputs, targets, Cli_data) in enumerate(trainloader):
        if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
        # inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs, Cli_data)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        #compute AUC
        prob_all_tr.extend(softmax_func(outputs)[:, 1].detach().numpy())
        label_all_tr.extend(targets)
        utils.progress_bar(batch_idx, len(trainloader), 'Train: | Loss: %.3f | Acc: %.3f (%d/%d)'
            % (train_loss/(batch_idx+1), float(correct)/total, correct, total))

    scheduler.step()
    Train_acc = float(correct)/total #*100.
    Train_auc = roc_auc_score(label_all_tr, prob_all_tr)
    print('Train_auc:%0.3f, Train_acc:%0.3f'% (Train_auc, Train_acc))


def val(epoch):
    global val_acc, best_val_acc, best_val_auc, best_val_auc_epoch, prob_all_test, label_all_test,\
    best_val_auc_test, best_val_acc_test, best_val_auc_tr, best_val_acc_tr,\
    tr_sens, tr_spec, test_sens, test_spec, val_sens, val_spec, prob_all_val, label_all_val
    net.eval()
    val_loss = 0; correct = 0; total = 0
    prob_all_val, label_all_val = [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets, Cli_data) in enumerate(exter_valloader):
            if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
            # inputs, targets = Variable(inputs), Variable(targets)
            # inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = net(inputs, Cli_data)
            loss = criterion(outputs, targets)
            val_loss += loss.data
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            # compute AUC
            prob_all_val.extend(softmax_func(outputs)[:, 1].detach().numpy())
            label_all_val.extend(targets)
            utils.progress_bar(batch_idx, len(exter_valloader), 'Val: | Loss: %.3f | Acc: %.3f (%d/%d)'
                               % (val_loss/(batch_idx+1), float(correct)/total, correct, total))
    # Save checkpoint.
    best_val_acc = float(correct)/total #*100.
    best_val_auc = roc_auc_score(label_all_val, prob_all_val)
    print('val_auc: %0.3f, acc:%0.3f' % (best_val_auc, best_val_acc))
    # if val_acc > best_val_acc: best_val_acc = val_acc
    # if val_auc > best_val_auc:
    #     print('Saving model..')
    #     print("best_val_auc: %0.3f" % val_auc)
    #     state = {'net': net.state_dict() if use_cuda else net, 'acc': val_acc, 'epoch': epoch}
    #     if not os.path.isdir(path): os.mkdir(path)
    #     torch.save(state, os.path.join(path, 'Val_model.t7'))
    # record best result
    best_val_auc_epoch = epoch
    best_val_auc_test, best_val_acc_test = Test_auc, Test_acc
    best_val_auc_tr, best_val_acc_tr = Train_auc, Train_acc
    # Compute Sensitivity and Specificity
    fpr, tpr, thresholds = roc_curve(label_all_tr, prob_all_tr, pos_label=1)
    senspe = tpr + (1 - fpr)
    tr_sens = tpr[np.argmax(senspe)]
    tr_spec = (1 - fpr)[np.argmax(senspe)]
    # print('Train Sensitivity:{:.3f}, Specificity:{:.3f}'.format(tpr[np.argmax(senspe)], (1 - fpr)[np.argmax(senspe)]))
    # print_roc(fpr, tpr, thresholds)
    fpr, tpr, thresholds = roc_curve(label_all_test, prob_all_test, pos_label=1)
    senspe = tpr + (1 - fpr)
    test_sens = tpr[np.argmax(senspe)]
    test_spec = (1 - fpr)[np.argmax(senspe)]
    # print('Test Sensitivity:{:.3f}, Specificity:{:.3f}'.format(tpr[np.argmax(senspe)], (1 - fpr)[np.argmax(senspe)]))
    # print_roc(fpr, tpr, thresholds)
    fpr, tpr, thresholds = roc_curve(label_all_val, prob_all_val, pos_label=1)
    senspe = tpr + (1 - fpr)
    val_sens = tpr[np.argmax(senspe)]
    val_spec = (1 - fpr)[np.argmax(senspe)]
    # print('Val Sensitivity:{:.3f}, Specificity:{:.3f}'.format(tpr[np.argmax(senspe)], (1 - fpr)[np.argmax(senspe)]))
    # print_roc(fpr, tpr, thresholds)

def test(epoch):
    global Test_acc, Test_auc, best_Test_acc, best_Test_auc, best_Test_auc_epoch, prob_all_test, label_all_test
    net.eval()
    Test_loss = 0; correct = 0; total = 0
    prob_all_test = [];label_all_test = []
    with torch.no_grad():
        for batch_idx, (inputs, targets, Cli_data) in enumerate(testloader):
            if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
            # inputs, targets = Variable(inputs), Variable(targets)
            # inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = net(inputs, Cli_data)
            loss = criterion(outputs, targets)
            Test_loss += loss.data
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            # compute AUC
            prob_all_test.extend(softmax_func(outputs)[:, 1].detach().numpy())
            label_all_test.extend(targets)
            utils.progress_bar(batch_idx, len(testloader), 'Test: | Loss: %.3f | Acc: %.3f (%d/%d)'
                               % (Test_loss/(batch_idx+1), float(correct)/total, correct, total))
        Test_acc = float(correct)/total #*100.
        # Save checkpoint.
        # if Test_acc > best_Test_acc: best_Test_acc = Test_acc
        Test_auc = roc_auc_score(label_all_test, prob_all_test)
        print("Test_auc: %0.3f" % Test_auc)
        if Test_auc > best_Test_auc:
            if not os.path.isdir(path): os.mkdir(path)
            # state = {'net': net.state_dict() if use_cuda else net, 'acc': Test_acc, 'epoch': epoch}
            # print('Saving model..')
            # torch.save(state, os.path.join(path, 'Test_model.t7'))
            best_Test_auc = Test_auc; best_Test_auc_epoch = epoch; best_Test_acc = Test_acc
            val(epoch)
    print("best_Test_auc: %0.3f,acc: %0.3f, epoch: %d" % (best_Test_auc, best_Test_acc, best_Test_auc_epoch))

if __name__ == '__main__':
    # record train log
    with open(os.path.join(path, results_log_csv_name), 'w') as f: f.write(' , AUC, ACC, sens, spec \n')
    with open(valset_pre_score, 'w') as f_val: f_val.write('val score Sx: \n')
    # AUC, ACC, sens, spec
    val_all_auc, val_all_acc, val_all_sens, val_all_spec = [], [], [], []
    test_all_auc, test_all_acc, test_all_sens, test_all_spec = [], [], [], []
    tr_all_auc, tr_all_acc, tr_all_sens, tr_all_spec = [], [], [], []

    for fold in range(5,10):
        # initialize seed
        print('fold: ', fold)
        initialize_torch_seed(fold)
        # Global variable
        best_Test_auc_epoch, best_Test_acc, best_Test_auc = 0, 0, 0 # best PublicTest accuracy
        best_val_auc_epoch, best_val_acc, best_val_auc = 0, 0, 0 # best Private accuracy
        best_val_auc_test, best_val_acc_test, best_val_auc_tr, best_val_acc_tr = 0,0,0,0
        tr_sens, tr_spec, test_sens, test_spec, val_sens, val_spec = 0,0,0,0,0,0
        prob_all_tr, label_all_tr = [], []
        prob_all_test, label_all_test = [], []
        prob_all_val, label_all_val = [], []
        # Model
        if opt.model == 'VGG19': net = VGG('VGG19')
        elif opt.model == 'Resnet18':  net = Densenet_Cli()
        # Optimizer
        criterion = nn.CrossEntropyLoss() # weight=torch.tensor([1.0, 1.2])
        # optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
        optimizer = torch.optim.Adam(net.parameters(), opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch, eta_min=1e-8)
        if use_cuda: net = net.cuda()
        #start train
        for epoch in range(start_epoch, total_epoch):
            print('time:',datetime.now().strftime('%b%d-%H:%M:%S'),'(',opt.save_dir,')')
            train(epoch)
            test(epoch) # val(epoch)
        # output val auc acc
        print("best_Test_acc: %0.3f, auc: %0.3f, epoch: %d" % (best_Test_acc, best_Test_auc, best_Test_auc_epoch))
        print("best_val_acc: %0.3f, auc: %0.3f, epoch: %d" % (best_val_acc, best_val_auc, best_val_auc_epoch))
        # record val pre score
        with open(valset_pre_score, 'a') as f_val:
            f_val.write('val score S' + str(fold) + ': \n')
            for prob in prob_all_val: f_val.write('%0.6f\n' % (prob))
        # record AUC, ACC, sens, spec
        val_all_auc.append(round(best_val_auc, 3)), val_all_acc.append(round(best_val_acc, 3)), \
        val_all_sens.append(round(val_sens, 3)), val_all_spec.append(round(val_spec, 3))
        test_all_auc.append(round(best_val_auc_test, 3)), test_all_acc.append(round(best_val_acc_test, 3)), \
        test_all_sens.append(round(test_sens, 3)), test_all_spec.append(round(test_spec, 3))
        tr_all_auc.append(round(best_val_auc_tr, 3)), tr_all_acc.append(round(best_val_acc_tr, 3)), \
        tr_all_sens.append(round(tr_sens, 3)), tr_all_spec.append(round(tr_spec, 3))
        # Log results
        with open(os.path.join(path, results_log_csv_name), 'a') as f:
            f.write('fold:,%d\n' % (fold))
            f.write(' ,AUC, ACC, sens, spec, best_val_epoch: %03d\n' % (best_val_auc_epoch))
            f.write('val,%0.3f,%0.3f,%0.3f,%0.3f,\n' % (best_val_auc,best_val_acc,val_sens,val_spec))
            f.write('te,%0.3f,%0.3f,%0.3f,%0.3f,\n' % (best_val_auc_test, best_val_acc_test, test_sens, test_spec))
            f.write('tr,%0.3f,%0.3f,%0.3f,%0.3f,\n' % (best_val_auc_tr, best_val_acc_tr, tr_sens, tr_spec))
            f.write('val prod:,%s\n' % (str(prob_all_val)))
            f.write('te prod:,%s\n' % (str(prob_all_test)))
            f.write('tr prod:,%s\n' % (str(prob_all_tr)))

    val_all_result = zip(val_all_auc, val_all_acc, val_all_sens, val_all_spec)
    val_all_result_avg = (np.mean(val_all_auc), np.mean(val_all_acc), np.mean(val_all_sens), np.mean(val_all_spec))
    test_all_result = zip(test_all_auc, test_all_acc, test_all_sens, test_all_spec)
    test_all_result_avg = (np.mean(test_all_auc), np.mean(test_all_acc), np.mean(test_all_sens), np.mean(test_all_spec))
    tr_all_result = zip(tr_all_auc, tr_all_acc, tr_all_sens, tr_all_spec)
    tr_all_result_avg = (np.mean(tr_all_auc), np.mean(tr_all_acc), np.mean(tr_all_sens), np.mean(tr_all_spec))
    # print result ,The object is destroyed after output
    print('val_all_res:', list(zip(val_all_auc, val_all_acc, val_all_sens, val_all_spec)))
    print('val_all_res_avg:', val_all_result_avg)
    print('te_all_res:', list(zip(test_all_auc, test_all_acc, test_all_sens, test_all_spec)))
    print('te_all_res_avg:', test_all_result_avg)
    print('tr_all_res:', list(zip(tr_all_auc, tr_all_acc, tr_all_sens, tr_all_spec)))
    print('tr_all_res_avg:', tr_all_result_avg)
    with open(os.path.join(path, results_log_csv_name), 'a') as f:
        for res in val_all_result: f.write('val,%s,%s,%s,%s\n' % (res))
        f.write('val_avg,%s,%s,%s,%s\n' % (val_all_result_avg))
        for res in test_all_result: f.write('te,%s,%s,%s,%s\n' % (res))
        f.write('te_avg,%s,%s,%s,%s\n' % (test_all_result_avg))
        for res in tr_all_result: f.write('tr,%s,%s,%s,%s\n' % (res))
        f.write('tr_avg,%s,%s,%s,%s\n' % (tr_all_result_avg))

