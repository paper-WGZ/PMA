import torch
import numpy as np
import torch.nn as nn
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

import concurrent.futures as cf
import multiprocessing as mp

from PMA0.dPMA import pma_sp
from torch.utils.data import Dataset


def TrendDistance(base, ssp, k_b, k_s):
    return torch.abs(base.TR[k_b] - ssp.TR[k_s])

def DTWDistance(s1, s2):
    DTW = {}
    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = ((s1[i] - s2[j]) ** 2)
            DTW[(i, j)] = dist + \
                          min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])
    return np.sqrt(DTW[len(s1) - 1, len(s2) - 1])

def Distance(s1, s2, p=2):
    pdist = nn.PairwiseDistance(p)  # p=2就是计算欧氏距离，p=1就是曼哈顿距离
    dist = pdist(s1, s2)
    return dist

# 高斯核函数计算
def GaussianKernel(s1, s2, sigma=1.0):
    # sigma越小，高斯图形越窄，模型复杂度越高，容易导致过拟合
    # sigma越大，高斯图形越宽，模型复杂度越低，容易导致欠拟合
    return torch.exp(-(torch.sum((s1 - s2) ** 2)) / (2 * sigma ** 2))

# sigmoid核
def SigmoidKernel(s1, s2, alpha=1.0, c=0.0):
    return torch.tanh(alpha * torch.dot(s1, s2) + c)


def GetFeatureVector(samp, proto_set, mode="all"):
    """
    mode: all, -phase, -magnitude, -shape, -angle
    """
    feature = torch.tensor([], dtype=torch.float)

    for proto_list in proto_set:
        distance = pma_sp(samp, proto_list)[1]
        # [[tensor(6)] * K] * var * num_proto, tensor(d_angle, d_pm, d_pr, d_mm, d_mr, d_shape)

        for dist_samp in distance:  # num_proto
            for dist_var in dist_samp:  # var
                for d in dist_var:
                    if mode == "all":
                        mask = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool)
                    elif mode == "-phase":
                        mask = torch.tensor([1, 0, 0, 1, 1, 1], dtype=torch.bool)
                    elif mode == "-magnitude":
                        mask = torch.tensor([1, 1, 1, 0, 0, 1], dtype=torch.bool)
                    elif mode == "-shape":
                        mask = torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.bool)
                    elif mode == "-angle":
                        mask = torch.tensor([0, 1, 1, 1, 1, 1], dtype=torch.bool)
                    elif mode == "phase":
                        mask = torch.tensor([0, 1, 1, 0, 0, 0], dtype=torch.bool)
                    elif mode == "magnitude":
                        mask = torch.tensor([0, 0, 0, 1, 1, 0], dtype=torch.bool)
                    elif mode == "shape":
                        mask = torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.bool)
                    elif mode == "angle":
                        mask = torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.bool)
                    else:
                        raise ValueError("UnFound distance mode")

                    feature = torch.cat([feature, d[mask]], dim=-1)

    return feature

class CustomDataset(Dataset):
    def __init__(self, feature_list, label_list):
        self.features = feature_list
        self.labels = label_list

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.as_tensor(self.features[idx], dtype=torch.float)
        y = torch.as_tensor(self.labels[idx], dtype=torch.long)
        return x, y

def GetFeatureSet(prototype_set, series_list, label_list, dist_mode='all', save_path=''):

    with cf.ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = [executor.submit(GetFeatureVector, samp, prototype_set, dist_mode)
                   for samp in series_list]

        feature_list = [f.result() for f in futures]

    feature_list = torch.stack(feature_list)

    data_list = torch.cat([feature_list, label_list.unsqueeze(1)], dim=-1)
    if save_path != '':
        np.savetxt(save_path, data_list.numpy(), delimiter=',')

    return CustomDataset(feature_list, label_list)