import time

import numpy
import torch
import argparse
import numpy as np

from PMA.dPMA import pma
from PMA import args as args_pma
from dataprocess import *
import matplotlib.pyplot as plt



def simulated_align(S, theta=0.2, sigma=2, t_set=[1, 2, 3]):
    """
    Simulated aligned-pairs generation for time series stretching using PyTorch.

    Parameters:
        S (torch.Tensor): Input time series of length L_S.
        theta (float): Magnify percentage (e.g., 0.2 for 20%).

    Returns:
        S_stretch (torch.Tensor): The stretched time series.
        Match (list of tuples): Warping aligned pairs of S and S_stretch.
    """


    L_S = S.size(0)  # Length of the input time series
    num_stretch_points = int(L_S * theta)  # Number of points to stretch

    start_point = np.random.randint(0, L_S - num_stretch_points)
    end_point = start_point + num_stretch_points

    mid = (S[start_point: end_point + 1] - torch.mean(S[start_point: end_point + 1])) * sigma
    left = S[: start_point] + (mid[0] - S[start_point])
    right = S[end_point + 1: ] + (mid[-1] - S[end_point])

    S_scale = torch.cat([left, mid, right], dim=0)

    #S_scale = ((S - torch.mean(S)) / torch.std(S)) ** 2

    selected_indices = np.random.choice(np.arange(L_S), size=num_stretch_points, replace=False)

    S_stretch = []
    Match = []


    # Iterate over all indices of S
    for i in range(L_S):
        # Append original point to S_stretch
        S_stretch.append(S_scale[i].item())
        Match.append((i, len(S_stretch) - 1))

        # If the current index is in selected_indices, replicate it with random delay
        if i in selected_indices:
            t = np.random.choice(t_set)  # Random delay from {1, 2, 3}
            for _ in range(t):
                S_stretch.append(S_scale[i].item())  # Replicate the point
                Match.append((i, len(S_stretch) - 1))  # Record the match

    # Convert S_stretch back to a torch.Tensor
    S_stretch = normalize(torch.tensor(S_stretch), mode='min-max')

    return S_stretch, Match



def plot_match(s1, s2, match, title, gap=1, step=10):
    s2 = s2 - torch.max(s2) + torch.min(s1) - gap
    fig, ax = plt.subplots(figsize=(10, 8))


    s1, s2 = s1.tolist(), s2.tolist()
    index1, index2 = zip(*match)
    for i in range(0, len(match), step):
        ax.plot([index1[i], index2[i]], [s1[int(index1[i])], s2[int(index2[i])]],
                color='gray', linestyle='--', linewidth=1.5)

    ax.plot(s1, color='blue', linewidth=3)
    ax.plot(s2, color='red', linewidth=3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # 隐藏坐标轴的刻度值
    plt.xticks([])  # 隐藏x轴刻度值
    plt.yticks([])  # 隐藏y轴刻度值

    # plt.show()
    #plt.title(title)
    plt.savefig(f'./{title}.png')
    plt.close()



def plot_align_path(path_list, label_list, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    color = ['black', 'blue', 'green', 'magenta', 'red']
    for i, path in enumerate(path_list):
        x, y = zip(*path)
        ax.plot(x, y, label=label_list[i], color=color[i], linewidth=4)

    #plt.show()


    ax.legend(fontsize=20, loc='upper left', frameon=False)  # 放大图例字体和标题

    # 加粗四边框
    for spine in ['bottom', 'left', 'top', 'right']:
        ax.spines[spine].set_linewidth(2)

    ax.tick_params(axis='both', which='both', direction='in',  labelsize=16, length=6, width=2)  # 刻度线向内

    # 添加左侧和下方刻度
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')



    # 保存矢量图
    plt.tight_layout()

    plt.savefig(f'./dtw_{title}.svg', format='svg')
    plt.savefig(f'./dtw_{title}.png')
    plt.close()




def get_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--root', default='D:/PMAP/ucr1/')
    parser.add_argument('--data_name', type=str, default='ShapesAll')
    parser.add_argument('--theta', type=float, default=0.3)
    parser.add_argument('--sigma', type=float, default=1.5)

    parser.add_argument('--split_window', type=int, default=100)
    parser.add_argument('--split_stride', type=int, default=50)
    parser.add_argument('--split_vertical', type=float, default=0.008)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args_pma.root = args.root
    args_pma.data_name = args.data_name


    folder = args.root + args.data_name + '/'
    train_file = folder + args.data_name + '_TRAIN.csv'
    test_file = folder + args.data_name + '_TEST.csv'
    samples, _ = get_dataset(train_file)

    args_pma.split_window = args.split_window
    args_pma.split_stride = args.split_stride
    args_pma.split_vertical = args.split_vertical

    for series in samples[0:10]:

        series = series[0]
        series_stretch, match = simulated_align(series, args.theta, args.sigma)

        match_pma = pma(series.reshape(1, 1, -1), series_stretch.reshape(1, 1, -1))[3]
        match_pma = match_pma[0][0]

        plot_match(series, series_stretch, match, title="ground truth")
        plot_match(series, series_stretch, match_pma, title="pma")




