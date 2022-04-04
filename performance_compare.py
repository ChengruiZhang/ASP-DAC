import torch
import numpy as np
from matplotlib import pyplot as plt
import copy


# compare the performance of our methods and previous methods
if __name__ == '__main__':

    # parameters
    rram_size = 128
    resolution = 8
    bit = 2


    # name = 'snn_vgg11_cifar100_100_final-2.pth'
    # Killnum = 10
    name = 'snn_vgg9_cifar10_100_node-306.pth'
    Killnum = 9
    # load remap location
    bias_weight_test_ori = np.load('./result/' + name + '%sbit' % bit + 'weight_test_ori.npy', allow_pickle=True).item()
    bias_weight_test = np.load('./result/' + name + '%sbit' % bit + 'weight_test.npy', allow_pickle=True).item()
    # bias_weight_test = np.load('./result/' + name + 'test' + '%sbit' % bit + 'weight_test.npy'
    #                            , allow_pickle=True).item()
    # bias_weight_test = np.load('./result/' + name + '%sbit' % bit + 'weight_KAIST_test.npy'
    #                            , allow_pickle=True).item()
    size = rram_size / resolution * bit
    result = {}
    result_ori = {}
    for i in bias_weight_test.keys():
        if i == 0 or i == Killnum:
            continue
        shape = bias_weight_test[i].shape
        result[i] = np.zeros((int(shape[0]/size), int(shape[1]/size)))
        result_ori[i] = np.zeros((int(shape[0]/size), int(shape[1]/size)))
        for j in range(int(shape[0]/size)):
            for k in range(int(shape[1]/size)):
                result[i][j, k] = bias_weight_test[i][int(j*size):int((j+1)*size), int(k*size):int((k+1)*size)].sum()
                result_ori[i][j, k] = bias_weight_test_ori[i][int(j*size):int((j+1)*size), int(k*size):int((k+1)*size)].sum()
        result[i] = result[i] / 1e4 / 100
        result_ori[i] = result_ori[i] / 1e4 / 100

    compare = {}
    compare_ori = {}
    for i in result.keys():
        compare[i] = (result[i].max() - result[i].min())/(result_ori[i].max() - result_ori[i].min())
    print(1)
