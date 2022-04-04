import torch
import numpy as np
from matplotlib import pyplot as plt
import copy


# compute the total power caused by the NN accelerator
if __name__ == '__main__':

    # parameters
    rram_size = 128
    resolution = 8
    bit = 2
    
    # load remap location
    # name = 'snn_vgg11_cifar100_100_final-2.pth'
    name = 'snn_vgg9_cifar10_100_node-306.pth'
    bias_weight_test_ori = np.load('./result/' + name + '%sbit' % bit + 'weight_test_ori.npy', allow_pickle=True).item()
    bias_weight_test = np.load('./result/' + name + '%sbit' % bit + 'weight_test.npy', allow_pickle=True).item()
    # bias_weight_test = np.load('./result/' + name + 'test' + '%sbit' % bit + 'weight_test.npy'
    #                            , allow_pickle=True).item()
    # bias_weight_test = np.load('./result/' + name + '%sbit' % bit + 'weight_KAIST_test.npy'
    #                            , allow_pickle=True).item()

    size = rram_size / resolution * bit
    max_result_ori = []
    max_result_test = []
    min_result_ori = []
    min_result_test = []

    result = {}
    result_ori = {}
    for i in bias_weight_test_ori.keys():
        # bias_weight_test_ori[i] = bias_weight_test_ori[i].transpose()
        shape = bias_weight_test_ori[i].shape
        ori_max, test_max = 0, 0
        ori_min, test_min = 1e100, 1e100
        result[i] = np.zeros((4, 16))
        result_ori[i] = np.zeros((4, 16))
        for j in range(int(shape[0]/4/size)):
            for k in range(int(shape[1]/16/size)):
                tmp = bias_weight_test_ori[i][int(j * 4 * size):int((j + 1) * 4 * size),
                      int(k * 16 * size):int((k + 1) * 16 * size)]
                if tmp.sum() >= ori_max:
                    ori_max = copy.deepcopy(tmp.sum())
                    tmp = tmp / 100 / 1e4
                    for x1 in range(4):
                        for y1 in range(16):
                            result_ori[i][x1, y1] = tmp[int(x1*size):int((x1+1)*size), int(y1*size):int((y1+1)*size)].sum()
                if tmp.sum() <= ori_min:
                    ori_min = copy.deepcopy(tmp.sum())
                tmp = bias_weight_test[i][int(j * 4 * size):int((j + 1) * 4 * size),
                      int(k * 16 * size):int((k + 1) * 16 * size)]
                if tmp.sum() >= test_max:
                    test_max = copy.deepcopy(tmp.sum())
                    tmp = tmp / 100 / 1e4
                    for x1 in range(4):
                        for y1 in range(16):
                            result[i][x1, y1] = tmp[int(x1 * size):int((x1 + 1) * size),
                                                    int(y1 * size):int((y1 + 1) * size)].sum()
                if tmp.sum() <= test_min:
                    test_min = copy.deepcopy(tmp.sum())
        max_result_ori.append(ori_max)
        max_result_test.append(test_max)
        min_result_ori.append(ori_min)
        min_result_test.append(test_min)

        # load npy
    input_spike = {}
    input_percentage = {}
    total_times = 100
    for i in range(6):
        name = 'Spike_test_input_10_' + str(3 * i) + '.npy'
    # name = 'Spike_train_input_' + str(total_times) + '_' + str(3 * i) + '.npy'
    input_spike[i] = np.load(name, allow_pickle=True)
    if input_spike[i].max() != 0:
        input_percentage[i] = input_spike[i] / input_spike[i].max()
    else:
        input_percentage[i] = input_spike[i]

    # parameter
    G_off = 0.1 * 1e-6
    G_on = 5 * 1e-6
    Voltage = 0.5

    # load weight
    model = torch.load('./trained_models/snn/snn_vgg5_cifar10_100_final.pth', map_location='cpu')
    weight_ori = {}
    G_ori = {}
    for i, j in enumerate(model['state_dict'].keys()):
        weight_ori[i] = model['state_dict'][j].numpy()
    n = weight_ori[i].shape
    G_ori[i] = (G_on - G_off) / (weight_ori[i].max() - weight_ori[i].min()) * (
            weight_ori[i] - weight_ori[i].min()) + G_off
    if len(n) == 4:
        G_ori[i] = G_ori[i].reshape(n[0], n[1] * n[2] * n[3])
        G_ori[i] = G_ori[i].transpose()
    else:
        G_ori[i] = G_ori[i].transpose()

    # calculate power
    power_ori = {}
    for i in range(6):
        power_ori[i] = copy.deepcopy(G_ori[i])
    if i <= 2:
        for j in range(len(input_spike[i])):
            power_ori[i][j,:] = power_ori[i][j,:] * input_spike[i][j] * Voltage * Voltage / 1000
    else:
        for j in range(len(input_spike[i])):
            power_ori[i][j,:] = power_ori[i][j,:] * input_spike[i][j] * Voltage * Voltage / 100

    # get the total_power_ori
    total_power_ori = {}
    array_size = [(5, 1), (9, 1), (64, 32), (32, 32)]
    for i in range(3):
        i = i + 2
    total_power_ori[i] = np.zeros(array_size[i-1])
    for j in range(array_size[i-1][1]):
        for k in range(array_size[i-1][0]):
            total_power_ori[i][k, j] = np.sum(power_ori[i][128 * k:128 * (k + 1), 128 * j:128 * (j + 1)])
            total_power_ori[1] = np.zeros(array_size[0])
            total_power_ori[1][0, 0] = np.sum(power_ori[1][0:128, 0:128])
            total_power_ori[1][1, 0] = np.sum(power_ori[1][128:256, 0:128])
            total_power_ori[1][2, 0] = np.sum(power_ori[1][256:384, 0:128])
            total_power_ori[1][3, 0] = np.sum(power_ori[1][384:512, 0:128])
            total_power_ori[1][4, 0] = np.sum(power_ori[1][512:576, 0:128])

    # get the power_remap
    power_remap = copy.deepcopy(power_ori)

    for i in range(6):
        if i == 1:
            continue
            n1, n2 = power_ori[i].shape
        print(i)
    for j in range(n1):
        print(j)
        for k in range(n2):
            tmp = location_ori_remap[i][j, k] - 1
            tmp_x = int(tmp // n2)
            tmp_y = int(tmp % n2)
            power_remap[i][j,k] = power_ori[i][tmp_x, tmp_y]

for i in range(1):
    i = 1
    power_remap[i] = np.zeros((640, 128))
    n1, n2 = power_remap[i].shape
    for j in range(n1):
        for k in range(n2):
            tmp = location_ori_remap[i][j, k] - 1
            tmp_x = int(tmp // n2)
            tmp_y = int(tmp % n2)
            if tmp_x >= 576 or tmp_y >= 128:
                power_remap[i][j, k] = 0
            else:
                power_remap[i][j, k] = power_ori[i][tmp_x, tmp_y]

# get the total_power_remap
total_power_remap = {}
array_size = [(5, 1), (9, 1), (64, 32), (32, 32)]
for i in range(4):
    i = i + 1
    total_power_remap[i] = np.zeros(array_size[i - 1])
    for j in range(array_size[i - 1][1]):
        for k in range(array_size[i - 1][0]):
            total_power_remap[i][k, j] = np.sum(power_remap[i][128 * k:128 * (k + 1), 128 * j:128 * (j + 1)])

np.save('total_power_KAIST.npy', total_power_remap)
np.save('power_KAIST.npy', power_remap)
print(1)
