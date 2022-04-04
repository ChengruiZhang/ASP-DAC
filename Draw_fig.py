import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import copy


# KAIST = np.load('snn_vgg5_cifar10_100_final.pthweight_KAIST_test.npy', allow_pickle=True).item()
# Ori = np.load('snn_vgg5_cifar10_100_final.pthweight_test_ori.npy', allow_pickle=True).item()
# # MP_FLP = np.load('snn_vgg5_cifar10_100_final-Copy1.pthweight_test.npy', allow_pickle=True).item()
# MP_FLP = np.load('snn_vgg5_cifar10_100_final.pthweight_test.npy', allow_pickle=True).item()
# LB = np.load('snn_vgg5_cifar10_100_final.pthweight_LB_test.npy', allow_pickle=True).item()


# KAIST = np.load('snn_vgg9_cifar10_100_node-306.pthweight_KAIST_test.npy', allow_pickle=True).item()
# Ori = np.load('snn_vgg9_cifar10_100_node-306.pthweight_test_ori.npy', allow_pickle=True).item()
# # MP_FLP = np.load('snn_vgg9_cifar10_100_node-306-Copy1.pthweight_test.npy', allow_pickle=True).item()
# MP_FLP = np.load('snn_vgg9_cifar10_100_node-306.pthweight_test.npy', allow_pickle=True).item()
# LB = np.load('snn_vgg9_cifar10_100_node-306.pthweight_LB_test.npy', allow_pickle=True).item()


KAIST = np.load('snn_vgg11_cifar100_100_final-2.pthweight_KAIST_test.npy', allow_pickle=True).item()
Ori = np.load('snn_vgg11_cifar100_100_final-2.pthweight_test_ori.npy', allow_pickle=True).item()
# MP_FLP = np.load('snn_vgg11_cifar100_100_final-2-Copy1.pthweight_test.npy', allow_pickle=True).item()
MP_FLP = np.load('snn_vgg11_cifar100_100_final-2.pthweight_test.npy', allow_pickle=True).item()
LB = np.load('snn_vgg11_cifar100_100_final-2.pthweight_LB_test.npy', allow_pickle=True).item()


LB_tmp = copy.deepcopy(LB)
Key = LB_tmp.keys()


for key in Key:
    H, W = LB[key].shape
    H_num, W_num = int(H / 128), int(W / 128)
    if key == 0 or key == 1:
        continue
    LB['P%s' % key] = np.zeros((H_num, W_num))
    KAIST['P%s' % key] = np.zeros((H_num, W_num))
    MP_FLP['P%s' % key] = np.zeros((H_num, W_num))
    Ori['P%s' % key] = np.zeros((H_num, W_num))

    for i in range(H_num):
        for j in range(W_num):
            x1, x2 = int(128 * i), int(128 * (i + 1))
            y1, y2 = int(128 * j), int(128 * (j + 1))

            LB['P%s' % key][i, j] = LB[key][x1:x2, y1:y2].sum()
            KAIST['P%s' % key][i, j] = KAIST[key][x1:x2, y1:y2].sum()
            MP_FLP['P%s' % key][i, j] = MP_FLP[key][x1:x2, y1:y2].sum()
            Ori['P%s' % key][i, j] = Ori[key][x1:x2, y1:y2].sum()

power_com = np.zeros(4)
num = 1
Ori_mm, KAIST_mm, LB_mm, MP_FLP_mm = {}, {}, {}, {}

for key in Key:
    H, W = LB[key].shape
    H_num, W_num = int(H / 128), int(W / 128)
    # if key == 0 or key == 1 or key == 5:
    # if key == 0 or key == 1 or key == 9:
    if key == 0 or key == 1 or key == 10:
        continue
    Ori_tmp, KAIST_tmp, LB_tmp1, MP_FLP_tmp = Ori['P%s' % key].max() - Ori['P%s' % key].min(), KAIST['P%s' % key].max() - KAIST['P%s' % key].min(), LB['P%s' % key].max() - LB['P%s' % key].min(), MP_FLP['P%s' % key].max() - MP_FLP['P%s' % key].min()
    Ori_mm[key] = [Ori['P%s' % key].max(), Ori['P%s' % key].min()]
    KAIST_mm[key] = [KAIST['P%s' % key].max(), KAIST['P%s' % key].min()]
    LB_mm[key] = [LB['P%s' % key].max(), LB['P%s' % key].min()]
    MP_FLP_mm[key] = [MP_FLP['P%s' % key].max(), MP_FLP['P%s' % key].min()]

    print('%f\t%f\t%f\t%f' % (Ori_tmp, KAIST_tmp, LB_tmp1, MP_FLP_tmp))
    H_num, W_num = 1, 1
    if Ori_tmp != 0:
        power_com[0] += 1 * H_num * W_num
        power_com[1] += KAIST_tmp/Ori_tmp * H_num * W_num
        power_com[2] += LB_tmp1 / Ori_tmp * H_num * W_num
        power_com[3] += MP_FLP_tmp / Ori_tmp * H_num * W_num
        num += 1
power_com /= power_com[0]
print(power_com)
np.save('VGG11_KAIST.npy', KAIST)
print(1)
# O = [1,1,1]
# K = [0.60487405, 0.6347007, 0.607658]
# L = [0.60325338, 0.6321514, 0.6031891]
# M = [0.47464953, 0.3554842, 0.3889337]
# X = np.arange(3) + 1
# plt.bar(X, O, 0.2, label="Ori", alpha=0.5)
# plt.bar(X+0.2, K, 0.2, label="KAIST", alpha=0.5)
# plt.bar(X+0.4, L, 0.2, label="LB", alpha=0.5)
# plt.bar(X+0.6, M, 0.2, label="MP_FLP", alpha=0.5)
# plt.legend()
# plt.show()