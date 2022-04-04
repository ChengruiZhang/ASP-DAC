import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchviz import make_dot
from matplotlib import pyplot as plt
import pdb
import sys
import datetime
import os
import datetime
import pydevd_pycharm

pydevd_pycharm.settrace('localhost', port=12233,
                        stdoutToServer=True, stderrToServer=True)

from self_models import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

## Load DataSet

def Load_Dataset(dataset, batch_size=64):
    if dataset == 'CIFAR100':
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        labels = 100
    elif dataset == 'CIFAR10':
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        labels = 10
    elif dataset == 'MNIST':
        labels = 10

    if dataset == 'CIFAR10' or dataset == 'CIFAR100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    if dataset == 'CIFAR100':
        train_dataset = datasets.CIFAR100(root='~/Datasets/cifar_data', train=True, download=True,
                                          transform=transform_train)
        test_dataset = datasets.CIFAR100(root='~/Datasets/cifar_data', train=False, download=True,
                                         transform=transform_test)

    elif dataset == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root='~/Datasets/cifar_data', train=True, download=True,
                                         transform=transform_train)
        test_dataset = datasets.CIFAR10(root='~/Datasets/cifar_data', train=False, download=True,
                                        transform=transform_test)

    elif dataset == 'MNIST':
        train_dataset = datasets.MNIST(root='~/Datasets/mnist/', train=True, download=True,
                                       transform=transforms.ToTensor()
                                       )
        test_dataset = datasets.MNIST(root='~/Datasets/mnist/', train=False, download=True,
                                      transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


train_loader, test_loader = Load_Dataset('CIFAR10')

## Load ANN Model

model = VGG(vgg_name='VGG5')

# load model: accuracy, epoch, model, state_dict, optimizer
state = torch.load('/root/SNN_Thermal/hybrid-snn-conversion/trained_models/ann_vgg5_cifar10.pth'
                   , map_location='cpu')

for k in state.keys():
    print(k)
print(state['model'])
dict = state['state_dict']
optimizer = state['optimizer']
print(optimizer.keys())
print(dict.keys())
print(dict['module.features.6.weight'].shape)

## Load SNN model

state_snn = torch.load('/root/SNN_Thermal/hybrid-snn-conversion/trained_models/snn_vgg5_cifar10.pth'
                   , map_location='cpu')
for k in state_snn.keys():
    print(k)
dict_snn = state_snn['state_dict']
optimizer_snn = state_snn['optimizer']
print(optimizer_snn.keys())
print(dict_snn.keys())
print(dict_snn['module.features.6.weight'].shape)