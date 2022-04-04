import torch
import numpy as np
from matplotlib import pyplot as plt
import copy


if __name__ == '__main__':

    power_ori = np.load('power_ori.npy', allow_pickle=True).item()
    total_power_ori = np.load('total_power_ori.npy', allow_pickle=True).item()
    for i in range(2):
        total_power_ori[i+3] = total_power_ori[i+3]

    power_remap = np.load('power_remap.npy', allow_pickle=True).item()
    total_power_remap = np.load('total_power_remap.npy', allow_pickle=True).item()
    for i in range(2):
        total_power_remap[i + 3] = total_power_remap[i + 3]

    power_KAIST = np.load('power_KAIST.npy', allow_pickle=True).item()
    total_power_KAIST = np.load('total_power_KAIST.npy', allow_pickle=True).item()
    for i in range(2):
        total_power_KAIST[i+3] = total_power_KAIST[i+3]

    total_power_ori_diff = np.zeros((4, 1))
    total_power_remap_diff = np.zeros((4, 1))
    total_power_KAIST_diff = np.zeros((4, 1))
    for i in total_power_remap.keys():
        total_power_ori_diff[i - 1] = total_power_ori[i].max() - total_power_ori[i].min()
        total_power_remap_diff[i - 1] = total_power_remap[i].max() - total_power_remap[i].min()
        total_power_KAIST_diff[i - 1] = total_power_KAIST[i].max() - total_power_KAIST[i].min()


    draw_fig = True
    if draw_fig == True:
        x = np.arange(4)
        y = total_power_ori_diff
        y2 = total_power_remap_diff
        y1 = total_power_KAIST_diff

        bar_width = 0.3
        tick_label = ["CONV-2", "CONV-3", "FC-1", "FC-2"]

        plt.bar(x, y, bar_width, align="center", color="white", label="Baseline", alpha=0.33, hatch = '\\', edgecolor='black')
        plt.bar(x + bar_width, y1, bar_width, color="white", align="center", label="TOPAR-C", alpha=0.33,hatch='-', edgecolor='black')
        plt.bar(x + 2 * bar_width, y2, bar_width, color="white", align="center", label="MP-FLP", alpha=0.33, edgecolor='black')

        plt.ylabel("Max-Min range")

        plt.xticks(x + bar_width / 2, tick_label)

        plt.legend()
        plt.savefig('diff_power.pdf')
        plt.show()

    xx = 6
    judge = True
    if judge == True:
        for i in range(2):
            tmp1 = 0
            i = i + 3
            loc = 1000
            for j in range(32*(5-i)):
                tmp2 = np.sum(total_power_KAIST[i][2*j:2*(j+1),:] * xx)
                if tmp2 > tmp1:
                    tmp1 = copy.deepcopy(tmp2)
                    loc = copy.deepcopy(j)
            print(loc)

    with open("total_power_ori.txt", "w") as f:
        for i in range(4):
            f.write('layer%d\n' % (i+1))
            n1, n2 = total_power_ori[i+1].shape
            num = 0
            for j in range(n1):
                for k in range(n2):
                    f.write('%.8f\t' % (total_power_ori[i+1][j, k] * xx))
                    num += 1
                    if num % 64 == 0:
                        f.write('\n')
            f.write('\n')

    with open("total_power_remap.txt", "w") as f:
        for i in range(4):
            f.write('layer%d\n' % (i+1))
            n1, n2 = total_power_remap[i+1].shape
            num = 0
            for j in range(n1):
                for k in range(n2):
                    f.write('%.8f\t' % (total_power_remap[i+1][j, k] * xx))
                    num += 1
                    if num % 64 == 0:
                        f.write('\n')
            f.write('\n')

    with open("total_power_KAIST.txt", "w") as f:
        for i in range(4):
            f.write('layer%d\n' % (i+1))
            n1, n2 = total_power_KAIST[i+1].shape
            num = 0
            for j in range(n1):
                for k in range(n2):
                    f.write('%.8f\t' % (total_power_KAIST[i+1][j, k] * xx))
                    num += 1
                    if num % 64 == 0:
                        f.write('\n')
            f.write('\n')

    print(1)