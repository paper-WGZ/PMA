import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
import torch.nn.functional as F

import concurrent.futures as cf
import multiprocessing as mp


from dataprocess import *
from metrics import *


def angle_similar(y, epsilon=10e-8):
    len_win = y.shape[-1]
    t = torch.zeros_like(y)
    t[:, :, :] = torch.linspace(0, 1, len_win, device=y.device)

    sim = torch.abs(
            torch.atan((y - y[..., 0].unsqueeze(-1))
                        / (t - t[..., 0].unsqueeze(-1) + epsilon))
             - torch.atan((y - y[..., -1].unsqueeze(-1))
                          / (t - t[..., -1].unsqueeze(-1) + epsilon)))
    min_sim, min_idx = torch.min(sim, dim=-1)

    return min_sim, min_idx


def vertical_similar(y):
    len_win = y.shape[-1]
    t = torch.zeros_like(y)
    t[:, :, :] = torch.linspace(0, 1, len_win, device=y.device)

    t0, t1 = t[..., 0].unsqueeze(-1), t[..., -1].unsqueeze(-1)
    y0, y1 = y[..., 0].unsqueeze(-1), y[..., -1].unsqueeze(-1)

    k = (y1 - y0) / (t1 - t0)
    d = torch.abs(k * (t - t0) +y0 - y) / torch.sqrt(k ** 2 + 1)

    max_dist, max_idx = torch.max(d, dim=-1)

    return max_dist, max_idx


def sliding_triangle_split(batch, window=10, stride=5, split_vertical=0.01):
    """

    :param batch:  (batch_size, num_var, len_seq)
    :param window: 滑动窗口的大小
    :param stride: 滑动窗口的步长
    :param device: 计算设备（'cpu' 或 'cuda'）
    :return: 分割后的子序列张量和每个窗口的左端点索引
    """
    batch_size, num_var, len_seq = batch.shape

    # 序列右侧复制填充
    pad_len = (window - len_seq % stride) % stride
    batch = F.pad(batch, (0, pad_len), mode='replicate')

    # 使用 unfold 进行滑动窗口分割
    window_batch = batch.unfold(dimension=-1, size=window, step=stride)
    # (batch_size, num_var, num_win, len_seq)

    #min_sim, min_idx = angle_similar(window_batch)
    min_sim, min_idx = vertical_similar(window_batch)
    # (batch_size, num_var, num_win)

    min_sim_ture = min_sim >= split_vertical
    # (batch_size, num_var, num_win)

    position = min_sim_ture.nonzero(as_tuple=False)
    # (num_ture, 3)

    num_win = window_batch.shape[2]
    left_index = torch.arange(0, num_win * stride, stride)

    split = [[[0, len_seq - 1] for _ in range(num_var)]
                                    for _ in range(batch_size)]

    for (i, j, k) in position:
        s = (left_index[k] + min_idx[i, j, k]).item()
        if s > split[i][j][-2] and s < len_seq - 1:
            split[i][j].insert(-1, s)

    return split


def remove_tiny(sp, tiny=1):
    k = 1
    while True:
        try:
            l, m, r = sp[k - 1], sp[k], sp[k + 1]
            if m - l <= tiny or r - m <= tiny:
                sp.pop(k)
            else:
                k += 1
        except IndexError:
            # 当索引超出范围时，退出循环
            break
    #sp.sort()
    return sp

def clean_split(sample, split, split_angle=0.523, tiny=2):

    index = torch.linspace(0, 1, sample.shape[-1], device=sample.device)

    for i, series in enumerate(sample):
        sp = split[i]


        flag = True
        while(flag):
            move = []
            flag = False
            # 使用while循环读取列表元素，直到最后一个元素
            k = 1
            while True:
                try:
                    l, m, r = sp[k - 1], sp[k], sp[k + 1]
                    #vertical_dist = vertical_distance(index[m], series[m], index[l: r], series[l: r])

                    neighbour_dist = angle_distance(index[l: m + 1], series[l: m + 1],
                                                index[m: r + 1], series[m: r + 1])
                    if neighbour_dist < split_angle:
                        sp.pop(k)
                        flag = True

                    else:
                        k += 1

                except IndexError:
                    # 当索引超出范围时，退出循环
                    break
        split[i] = remove_tiny(sp, tiny)
    return split

def get_split(batch, args):
    """
        :param batch: (batch_size, num_var, len_seq)
        :param args:
        :return:
        """
    batch_smooth = batch
    if args.is_smooth:
        batch_smooth = loess_smoothing(batch, args.local_ratio)

    batch_splits = sliding_triangle_split(batch_smooth, args.split_window,
                                          args.split_stride, args.split_vertical)


    with cf.ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = [executor.submit(clean_split,
                                   sample_smooth, batch_splits[i], args.split_angle)
                   for i, sample_smooth in enumerate(batch_smooth)]

        split_batch = [f.result() for f in futures]

    return split_batch, batch_smooth



def plot_split(series, split, series_smooth, information, save_path):

    def info2table(info):
        table = info.tolist()
        header = ['angle', 'x_median', 'x_range', 'y_median', 'y_range']
        table = [[h] + [round(float(x), 2) for x in lst]
                 for h, lst in zip(header, table)]
        return table

    # 创建绘图
    fig, (ax, ax_table) = plt.subplots(nrows=2, figsize=(10, 8),
                                       gridspec_kw={'height_ratios': [8, 1]})

    t = torch.arange(0, len(series))
    ax.plot(t, series)
    ax.plot(t, series_smooth)
    ax.vlines(split[1:-1], 0, 1, linestyles='dashed', colors='red')

    # 添加网格线
    plt.grid(True)

    # 在下方子图中添加表格
    ax_table.axis('off')
    table = info2table(information)
    _ = ax_table.table(cellText=table, loc='bottom', cellLoc='center',
                       rowLoc=[0.1] * len(table))

    plt.subplots_adjust(hspace=0.1)

    # 保存图像
    # plt.show()
    plt.savefig(save_path)
    plt.close()


def sgm_info(index: torch.Tensor, values: torch.Tensor):
    # 计算斜率 slope
    slope = sgm_slope(index, values)
    # 计算倾斜角 angle
    angle = torch.atan(slope)
    # 计算 median, range
    x_l, x_r = index[0], index[-1]
    x_median = (x_l + x_r) / 2
    y_median = torch.mean(values)

    y_l = slope * (x_l - x_median) + y_median
    y_r = slope * (x_r - x_median) + y_median

    x_range = x_r - x_l
    y_range = y_r - y_l

    info = torch.as_tensor([angle, x_median, x_range, y_median, y_range],
                           dtype=torch.float, device=values.device)
    return info

def samp2sgm(samp: torch.Tensor, split):
    """
    samp: (num_var, len_seq)
    split: [num_var, num_seg[j] + 1]
    return: sgm: [num_var, num_seg[j], tensor],
    return: info: [num_var, tensor(5, num_seg[j])]
    """

    index = torch.linspace(0, 1, samp.size(-1), device=samp.device)

    sgm, info = [], []
    for j, series in enumerate(samp):
        sgm.append([])
        info.append([])
        sp = split[j]
        for k in range(len(sp) - 1):
            l, r = sp[k], sp[k + 1] + 1
            sgm[j].append(series[l: r])
            info[j].append(sgm_info(index[l: r], series[l: r]).unsqueeze(1))
        info[j] = torch.cat(info[j], dim=1)

    return sgm, info

def batch_info(batch, split):
    with cf.ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = [executor.submit(samp2sgm, samp, split[i])
                   for i, samp in enumerate(batch)]

        results = [f.result() for f in futures]

    segments, information = map(list, zip(*results))
    return segments, information


def tensor_remove(tensor, i, dim=0):
    """
    从tensor的第 dim 维中移除第 i 个元素。
    """
    return torch.cat((tensor.narrow(dim, 0, i),
                      tensor.narrow(dim, i + 1, tensor.size(dim) - i - 1)),
                     dim=dim)



class SplitBatch:
    def __init__(self, batch: torch.Tensor, split, batch_smooth, segments, information):
        """
        sample: (batch_size, num_var, len_seq)

        """
        self.series = batch
        self.num, self.num_var, self.seq_len = batch.shape

        self.split, self.series_smooth = split, batch_smooth
        # split: [batch_size, num_var, num_seg[i][j] + 1]

        self.num_sgm = [[len(s) - 1 for s in sp] for sp in self.split]  # number of segments
        # num_sgm: [batch_size, num_var]

        self.segments, self.information = segments, information
        # segments: [batch_size, num_var, num_seg[i][j], tensor()]
        # information: [batch_size, num_var, tensor(5, num_seg[i][j])],
        # angle, phase_median, phase_range, magnit_median, magnit_range



    def update_by_split(self, split):
        self.split = split
        self.num_sgm = [[len(s) - 1 for s in sp] for sp in split]
        if self.num != len(split):
            self.series = self.series.repeat(len(split), 1, 1)
            self.num = len(split)
        self.segments, self.information = batch_info(self.series, split)


    def __getitem__(self, i):
        batch = self.series[i].unsqueeze(0)

        split = [self.split[i]]
        batch_smooth = self.series_smooth[i].unsqueeze(0)
        # split: [1, num_var, num_seg[i][j] + 1]

        segments = [self.segments[i]]
        information = [self.information[i]]

        return SplitBatch(batch, split, batch_smooth, segments, information)


    def pop(self, i):
        self.series = tensor_remove(self.series, i, dim=0)
        self.num_sample -= 1
        self.split.pop(i)
        self.series_smooth = tensor_remove(self.series_smooth, i, dim=0)
        self.num_sgm.pop(i)
        self.segments.pop(i)
        self.information.pop(i)



def plot_split_batch(samp: SplitBatch, plot_root):
    os.makedirs(plot_root, exist_ok=True)

    for i in range(samp.batch_size):
        for j in range(samp.num_var):
            plot_path = os.path.join(plot_root, f'{i}_{j}.png')

            plot_split(samp.series[i][j], samp.split[i][j],
                       samp.series_smooth[i][j], samp.information[i][j],
                       plot_path)


def split_batch(batch: torch.Tensor, args, save_root=''):
    split, batch_smooth = get_split(batch, args)
    # split: [batch_size, num_var, num_seg[i][j] + 1]

    segments, information = batch_info(batch, split)
    # segments: [batch_size, num_var, num_seg[i][j], tensor()]
    # information: [batch_size, num_var, tensor(5, num_seg[i][j])],
    # angle, phase_median, phase_range, magnit_median, magnit_range

    samp = SplitBatch(batch, split, batch_smooth, segments, information)

    if save_root != '':
        plot_split_batch(samp, save_root)
    return samp

