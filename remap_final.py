## location needs -1 because it starts at 1

import torch
import numpy as np
from matplotlib import pyplot as plt
import copy


# input: weight_bias, location, type (row:1 or col:0), divide_num Output: remap_weight, remap_location
# output: re-ordered matrix for origin and test data; re-ordered location
# the goal of this function is to allocate the weight to the proper location
def remap(input_mat, input_mat_test, location, tp, divide_num, divide_len=128):

    if tp == 0:
        input_mat = input_mat.transpose()
        input_mat_test = input_mat_test.transpose()
        location = location.transpose()

    input_sum = input_mat.sum(axis = 1)
    loc_tmp = np.argsort(input_sum)

    # get the exchange list
    exchange_list = np.zeros(divide_len*divide_num)
    order = 1
    for i in range(divide_len):
        tmp = loc_tmp[int(i*divide_num) : int((i+1)*divide_num)]
        if order == 1:
            for j in range(divide_num):
                exchange_list[i+divide_len*j] = tmp[j]
            order *= -1
        else:
            for j in range(divide_num):
                exchange_list[i+divide_len*j] = tmp[-(j+1)]
            order *= -1

    # do exchange
    out_mat = np.zeros(input_mat.shape)
    out_mat_test = np.zeros(input_mat_test.shape)
    out_location = np.zeros(location.shape)
    for i in range(len(exchange_list)):
        out_mat[i, :] = input_mat[int(exchange_list[i]), :]
        out_mat_test[i, :] = input_mat_test[int(exchange_list[i]), :]
        out_location[i, :] = location[int(exchange_list[i]), :]

    if tp == 0: # transpose the result to match
        out_mat = out_mat.transpose()
        out_mat_test = out_mat_test.transpose()
        out_location = out_location.transpose()

    return out_mat, out_mat_test, out_location


# input: exchange_list, [min, max] of this part, input_sum, part_sum (sum of each row/col), minimum division number
# output: exchange_list, part_sum (after exchange)
# find 2 elements in exchange_list whose subtraction is diff.
def finer_exchange(exchange_list, part_min, part_max, input_sum, part_sum, diff, divide_len=128):
    P_min, P_max = np.zeros(divide_len), np.zeros(divide_len)

    for i in range(divide_len):
        P_min[int(-(i+1))] = exchange_list[int(divide_len * part_min + i)] # P_min, P_max zengxu
        P_max[int(-(i+1))] = exchange_list[int(divide_len * part_max + i)]
    res_i, res_j = -1, -1

    tmp = 1e10
    for i in range(divide_len): # i for min, j for max
        tmp2 = 0
        if input_sum[int(P_max[-1])] <= input_sum[int(P_min[0])]:
            break
        for j in range(divide_len):
            if tmp2 == 2:
                break
            if abs(diff-(input_sum[int(P_max[j])] - input_sum[int(P_min[i])])) <= tmp:
                res_i, res_j = i, j
                tmp = abs(diff-(input_sum[int(P_max[j])] - input_sum[int(P_min[i])]))
            if input_sum[int(P_max[j])] - input_sum[int(P_min[i])] > diff:
                tmp2 += 1
    tmp3 = exchange_list[int(divide_len * part_min + res_i)]
    if res_i != -1:
        exchange_list[int(divide_len * part_min + res_i)] = exchange_list[int(divide_len * part_max + res_j)]
        exchange_list[int(divide_len * part_max + res_j)] = tmp3
        part_sum[part_min] += input_sum[int(P_max[res_j])] - input_sum[int(P_min[res_i])]
        part_sum[part_max] -= input_sum[int(P_max[res_j])] - input_sum[int(P_min[res_i])]
    return exchange_list, part_sum


# input: weight_bias, location, type (row:1 or col:0), divide_num 
# Output: remap_weight, remap_location
# the goal of this function is to allocate the weight to the proper location
def remap_opt(input_mat, input_mat_test, location, tp, divide_num, divide_len=128, iter=200):

    if tp == 0:
        input_mat = input_mat.transpose()
        input_mat_test = input_mat_test.transpose()
        location = location.transpose()

    input_sum = input_mat.sum(axis = 1)
    loc_tmp = np.argsort(input_sum)
    input_sum_tmp = copy.deepcopy(input_sum)
    # get the exchange list
    exchange_list = np.zeros(divide_len*divide_num)
    for i in range(len(exchange_list)):
        exchange_list[i] = i

    order = 1
    for i in range(divide_len):
        tmp = loc_tmp[int(i*divide_num) : int((i+1)*divide_num)]
        if order == 1:
            for j in range(divide_num):
                exchange_list[i+divide_len*j] = tmp[j]
            order *= -1
        else:
            for j in range(divide_num):
                exchange_list[i+divide_len*j] = tmp[-(j+1)]
            order *= -1

    for i in range(len(exchange_list)):
        input_sum_tmp[i] = input_sum[int(exchange_list[i])]

    part_num = int(len(input_sum_tmp)/divide_len)
    part_sum = np.zeros(part_num)
    for i in range(part_num):
        part_sum[i] = input_sum_tmp[int(i*128):int((i+1)*128)].sum()
    part_mean = copy.deepcopy(part_sum.mean())
    part_mm = copy.deepcopy(part_sum.max() - part_sum.min())

    mm = part_sum.max() - part_sum.min()
    for i in range(iter):
        if mm <= part_mm / 200:
            break
        part_min, part_max = part_sum.argmin(), part_sum.argmax()
        # exchange_list[int(divide_num*part_min):int(divide_num*(part_min+1))],
        exchange_list, part_sum = finer_exchange(exchange_list, part_min, part_max, input_sum, part_sum, diff=mm/2, divide_len=divide_len)
        mm = part_sum.max() - part_sum.min()

    # do exchange
    out_mat = np.zeros(input_mat.shape)
    out_mat_test = np.zeros(input_mat_test.shape)
    out_location = np.zeros(location.shape)
    for i in range(len(exchange_list)):
        out_mat[i, :] = input_mat[int(exchange_list[i]), :]
        out_mat_test[i, :] = input_mat_test[int(exchange_list[i]), :]
        out_location[i, :] = location[int(exchange_list[i]), :]

    if tp == 0: # transpose the result to match
        out_mat = out_mat.transpose()
        out_mat_test = out_mat_test.transpose()
        out_location = out_location.transpose()

    return out_mat, out_mat_test, out_location


# remap and calculate power
if __name__ == '__main__':
    # parameter
    G_off = 0.1 * 1e-6
    G_on = 5 * 1e-6

    # load the origin location and weight information

    # location = np.load('./location_ori.npy', allow_pickle=True).item()
    # input_spike = np.load('./spike/VGG5_10_Spike_power_all.npy', allow_pickle=True).item()
    # input_spike_test = np.load('./spike/VGG5_10_Spike_power_test_32.npy', allow_pickle=True).item()
    #
    # name = 'snn_vgg5_cifar10_100_final.pth'

    # location = np.load('./spike/loc_snn_vgg9_10_ori.npy', allow_pickle=True).item()
    # input_spike = np.load('./spike/VGG9_10_Spike_test_input_all.npy', allow_pickle=True).item()
    # input_spike_test = np.load('./spike/VGG9_10_Spike_power_32.npy', allow_pickle=True).item()
    #
    # name = 'snn_vgg9_cifar10_100_node-306.pth'

    location = np.load('./spike/loc_snn_vgg11_10_ori.npy', allow_pickle=True).item()
    input_spike = np.load('./spike/VGG11_100_Spike_test_input_all.npy', allow_pickle=True).item()
    input_spike_test = np.load('./spike/VGG11_100_Spike_power_32.npy', allow_pickle=True).item()

    name = 'snn_vgg11_cifar100_100_final-2.pth'



    model = torch.load('./trained_models/snn/' + name, map_location='cpu')
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

    # bias the weight
    bias_weight = {}
    bias_weight_test = {}
    input_percentage = {}
    input_percentage_test = {}

    for i in location.keys():
        bias_weight[i] = copy.deepcopy(G_ori[i])
        bias_weight_test[i] = copy.deepcopy(G_ori[i])
        if input_spike[3*i].max() != 0:
            # input_percentage[i] = input_spike[3*i] / input_spike[3*i].max()
            input_percentage[i] = input_spike[3 * i] * 1e5 * 0.25
            input_percentage_test[i] = input_spike_test[3 * i] * 1e5 * 0.25
        else:
            # input_percentage[i] = input_spike[3*i]
            input_percentage[i] = input_spike[3 * i] * 1e5 * 0.25
            input_percentage_test[i] = input_spike_test[3 * i] * 1e5 * 0.25
        for j in range(len(input_percentage[i])):
            bias_weight[i][j, :] = input_percentage[i][j] * G_ori[i][j, :]  # col to col mul
            bias_weight_test[i][j, :] = input_percentage_test[i][j] * G_ori[i][j, :]  # col to col mul

    # now, bias_weight, bias_weight_test represents the power information of general case and test case (32 pics)

    bias_weight_ori = copy.deepcopy(bias_weight)
    bias_weight_test_ori = copy.deepcopy(bias_weight_test)
    location_ori = copy.deepcopy(location)

    # for each layer, fill up the weight matrix with 0 tensor

    for key in bias_weight_ori.keys():
        print(key)
        if key == 0 or key == 1:
            continue
        # get tmp weight and location
        weight_tmp, location_tmp = bias_weight[key], location[key]
        H, W = weight_tmp.shape
        W_num = int(W / 128)
        if W_num == 0:
            continue
        H_num = int(H / 128)
        Tile_num = int(np.ceil(H_num * W_num / 64))

        # for each layer, do remap several times
        bias_weight[key], bias_weight_test[key], location[key] = remap_opt(bias_weight[key], bias_weight_test[key],
                                                                               location[key], tp=1,
                                                                               divide_num=int(H_num), divide_len=128, iter=1000)

        tmp = int(64 / W_num)  # 1 Tile contains how many 128 rows
        if tmp > H_num:
            tmp = H_num
        tmp = 1

        for i in range(H_num):
            print('Step: %f%%' % (i/H_num*100))
            index_1, index_2 = int(128 * i * tmp), int(128 * tmp * (i + 1))
            if i != H_num - 1:
                bias_weight[key][index_1:index_2], bias_weight_test[key][index_1:index_2], location[key][
                                                                                           index_1:index_2] = remap_opt(
                    bias_weight[key][index_1:index_2], bias_weight_test[key][index_1:index_2],
                    location[key][index_1:index_2], tp=0, divide_num=W_num, divide_len=128, iter=1000)
            else:
                bias_weight[key][index_1:H], bias_weight_test[key][index_1:H], location[key][index_1:H] = remap_opt(
                    bias_weight[key][index_1:H], bias_weight_test[key][index_1:H], location[key][index_1:H], tp=0,
                    divide_num=W_num, divide_len=128, iter=1000)

        
    np.save(name + 'weight_ori.npy', bias_weight_ori)
    np.save(name + 'weight_test_ori.npy', bias_weight_test_ori)
    np.save(name + 'location_ori.npy', location_ori)
    np.save(name + 'weight.npy', bias_weight)
    np.save(name + 'weight_test.npy', bias_weight_test)
    np.save(name + 'location.npy', location)

    ## KAIST's method
    bias_weight_KAIST = copy.deepcopy(G_ori)
    # bias_weight_KAIST = copy.deepcopy(bias_weight_ori)
    bias_weight_KAIST_test = copy.deepcopy(bias_weight_test_ori)
    location_KAIST = copy.deepcopy(location_ori)
    
    # for each layer, do remap
    for key in bias_weight_ori.keys():
        print(key)
        if key == 0 or key == 1:
            continue
        # get tmp weight and location
        weight_tmp, location_tmp = bias_weight_KAIST[key], location_KAIST[key]
        H, W = weight_tmp.shape
        W_num = int(W / 128)
        if W_num == 0:
            continue
        H_num = int(H / 128)
        Tile_num = int(np.ceil(H_num * W_num / 64))
    
        # for each layer, do remap several times
        bias_weight_KAIST[key], bias_weight_KAIST_test[key], location_KAIST[key] = remap(bias_weight_KAIST[key],
                                                                                         bias_weight_KAIST_test[key],
                                                                                         location_KAIST[key],
                                                                                         tp=0, divide_num=W_num,
                                                                                         divide_len=128)
    
    np.save(name + 'weight_KAIST_test.npy', bias_weight_KAIST_test)
    np.save(name + 'location_KAIST.npy', location_KAIST)
    
    ## Lin Bin's method
    bias_weight_LB = copy.deepcopy(G_ori)
    # bias_weight_LB = copy.deepcopy(bias_weight_ori)
    bias_weight_LB_test = copy.deepcopy(bias_weight_test_ori)
    location_LB = copy.deepcopy(location_ori)
    
    # for each layer, do remap
    for key in bias_weight_ori.keys():
        print(key)
        if key == 0 or key == 1:
            continue
        # get tmp weight and location
        weight_tmp, location_tmp = bias_weight_LB[key], location_LB[key]
        H, W = weight_tmp.shape
        W_num = int(W / 128)
        if W_num == 0:
            continue
        H_num = int(H / 128)
        Tile_num = int(np.ceil(H_num * W_num / 64))
    
        # for each layer, do remap several times
        bias_weight_LB[key], bias_weight_LB_test[key], location_LB[key] = remap(bias_weight_LB[key],
                                                                                bias_weight_LB_test[key],
                                                                                location_LB[key], tp=0,
                                                                                divide_num=W_num, divide_len=128)
    
        bias_weight_LB[key], bias_weight_LB_test[key], location_LB[key] = remap(bias_weight_LB[key],
                                                                                bias_weight_LB_test[key],
                                                                                location_LB[key], tp=1,
                                                                                divide_num=H_num, divide_len=128)
    
    np.save(name + 'weight_LB_test.npy', bias_weight_LB_test)
    np.save(name + 'location_LB.npy', location_LB)
