import torch
import numpy as np
from scipy import interpolate
import concurrent.futures as cf
import multiprocessing as mp
from scipy.stats import linregress

from dataprocess import normalize
from match import *
from split import *

def is_all_zero(tensor):
    return torch.nonzero(tensor).numel() == 0

def interp_resamp1(sgm, len_new, kind='linear'):
    """
    kind: 'linear', 'cubic', 'quadratic', 'nearest'
    """
    if len(sgm) < 2: sgm = torch.cat([sgm, sgm])

    len_old = len(sgm)
    x_old = np.linspace(0, len_old - 1, len_old)

    # 新的时间点
    x_new = np.linspace(0, len_old - 1, len_new)
    #print(len_old, len_new, x_old, x_new)
    if len_old<2:
        print(x_old, sgm)
    f = interpolate.interp1d(x_old, sgm, kind=kind)
    sgm_new = f(x_new)
    return torch.as_tensor(sgm_new, dtype=torch.float, device=sgm.device)


def interp_resamp(sgm, len_new):
    """
    使用线性插值对时间序列进行重采样。

    参数:
    sgm: 原始时间序列 (torch tensor)
    len_new: 新时间序列的长度 (int)

    返回:
    sgm_new: 重采样后的时间序列 (torch tensor)
    """
    if len(sgm) < 2:
        sgm = torch.cat([sgm, sgm])

    len_old = len(sgm)
    x_old = torch.linspace(0, len_old - 1, len_old, device=sgm.device)
    x_new = torch.linspace(0, len_old - 1, len_new, device=sgm.device)

    # 创建索引以便在xp中找到与x相邻的两个点
    idx = torch.searchsorted(x_old, x_new, right=True) - 1
    idx = idx.clamp(0, len_old - 2)

    # 计算插值
    x0 = x_old[idx]
    x1 = x_old[idx + 1]
    y0 = sgm[idx]
    y1 = sgm[idx + 1]
    slope = (y1 - y0) / (x1 - x0)
    sgm_new = y0 + slope * (x_new - x0)

    return sgm_new

def modulate_segment1(sgm_s, sgm_p, mode='min-loss', epsilon=10e-6):
    """sgm_s to sgm_p"""

    # 插值重采样
    sgm_int = interp_resamp(sgm_s, len(sgm_p))

    # 调整幅度
    if mode == 'min-loss':
        mu_p, mu_int = torch.mean(sgm_p), torch.mean(sgm_int)

        diff_int = sgm_int - mu_int
        diff_p = mu_p - sgm_p

        A_m = torch.sum(diff_int * diff_p)
        A_d = torch.sum(diff_int * diff_int)

        # 计算A和B
        A = - A_m / (A_d + epsilon)
        B = mu_p - A * mu_int

    if mode == 'min-max':
        min_p, max_p = torch.min(sgm_p), torch.max(sgm_p)
        min_int, max_int = torch.min(sgm_int), torch.max(sgm_int)
        A = (max_p - min_p) / (max_int - min_int + epsilon)
        B = min_p - A * min_int

    sgm_int = A * sgm_int + B
    # print(sgm_trans)

    return sgm_int


def linear_fit(x, y):
    """
    使用最小二乘法拟合直线，并返回斜率和截距
    参数:
    x: 时间序列的横坐标 (tensor of shape (n,))
    y: 时间序列的纵坐标 (tensor of shape (n,))

    返回:
    slope: 斜率 (tensor)
    intercept: 截距 (tensor)
    """

    x_mean = torch.mean(x)
    y_mean = torch.mean(y)

    xy_mean = torch.mean(x * y)
    xx_mean = torch.mean(x * x)

    slope = (xy_mean - x_mean * y_mean) / (xx_mean - x_mean * x_mean + 1e-8)
    intercept = y_mean - slope * x_mean

    return slope, intercept

def modulate_segment(sgm_s, sgm_p):
    """
    info: (angle, phase_median, phase_range, magnit_median, magnit_range)
    """
    index_p = torch.linspace(0, 1, len(sgm_p), device=sgm_p.device)
    slope_p, intercept_p = linear_fit(index_p, sgm_p)

    # 插值重采样
    sgm_int = interp_resamp(sgm_s, len(sgm_p))
    slope_s, intercept_s = linear_fit(index_p, sgm_int)

    sgm_mod = sgm_int - (slope_s * index_p + intercept_s) + (slope_p * index_p + intercept_p)
    return sgm_mod



def modulate_sample(samp_sgms, proto_sgms):
    """
    sgms: [num_var, num_seg[j], tensor()]
    return: mod_sgms: [num_var, num_seg[j], tensor()]
    return: shape_dist: [num_var * tensor(num_seg[j])]
    """

    device = samp_sgms[0][0].device
    mod_sgms, shape_dist = [], []
    for j in range(len(samp_sgms)):
        mod_sgms.append([])

        num_seg = len(samp_sgms[j])
        dist = torch.zeros(num_seg, dtype=torch.float, device=device)
        for k in range(num_seg):
            sgm_mod = modulate_segment(samp_sgms[j][k], proto_sgms[j][k])
            mod_sgms[j].append(sgm_mod)

            len_sgm_s, len_sgm_p = len(samp_sgms[j][k]), len(proto_sgms[j][k])
            dist[k] = torch.mean((proto_sgms[j][k] - sgm_mod) ** 2) * (len_sgm_s + len_sgm_p) / 2
        shape_dist.append(dist)
    shape_dist = torch.stack(shape_dist, dim=0)

    return mod_sgms, shape_dist


def adjust_segment(sgm, t_range, seq_len, v_median, v_range):
    """
    :sgm: raw segment

    """
    len_new = round(seq_len * t_range.item())
    t = torch.linspace(0, t_range, len_new, device=sgm.device)
    trend_new = v_range / t_range * t + v_median - v_range / 2

    sgm = interp_resamp(sgm, len_new)
    slope, intercept = linear_fit(t, sgm)
    trend_old = slope * t + intercept

    sgm = sgm - trend_old + trend_new
    return sgm


def MergeSegments(segment):
    series = segment[0]
    for k in range(1, len(segment)):
        p_l, p_r = series[-1], segment[k][0]
        segment[k] = segment[k] - p_r + p_l
        series = torch.cat([series, segment[k][1:]])
    series = normalize(series, mode='std')
    return series


def plot_modulate(samp_series, proto_series, mod_segments,
                  samp_split, proto_split, match,
                  distance, save_path):

    def dist2table(dist):
        table = dist.tolist()
        header = ['shape', 'angle', 'magnit_scale', 'magnit_shift', 'phase_scale', 'phase_shift']
        table = [[h] + [round(float(x), 2) for x in lst]
                 for h, lst in zip(header, table)]
        return table

    fig, (ax, ax_table) = plt.subplots(nrows=2, figsize=(8, 10),
                                       gridspec_kw={'height_ratios': [9, 1]})
    plt.subplots_adjust(hspace=0.5)

    # 曲线
    ax.plot(samp_series + 1.1)
    ax.vlines(samp_split[1:-1], 1.1, 2.1, linestyles='dashed', colors='red')

    ax.plot(proto_series)
    ax.vlines(proto_split[1:-1], 0, 1, linestyles='dashed', colors='red')

    ax.vlines(proto_split[1:-1], -1.1, -0.1, linestyles='dashed', colors='red')

    num_seg = len(mod_segments)
    for k in range(num_seg):
        t = torch.arange(proto_split[k], proto_split[k + 1] + 1)
        ax.plot(t, mod_segments[k] - 1.1, color='blue')

        if (k, k) in match:
            p1 = (proto_split[k] + proto_split[k + 1]) / 2
            p2 = (samp_split[k] + samp_split[k + 1]) / 2
            ax.plot([p1, p2], [1.0, 1.1], 'g--')

    ax.set_xticks([])  # 隐藏x轴刻度
    ax.set_yticks([])  # 隐藏y轴刻度

    # 表格
    distance = distance[:, :num_seg]
    table = dist2table(distance)

    ax_table.axis('off')
    _ = ax_table.table(cellText=table, loc='bottom', cellLoc='center',
                       rowLoc=[0.1] * len(table))

    # 保存图像
    # plt.show()
    plt.savefig(save_path)
    plt.close()


def plot_modulate_batch(samp: SplitBatch, proto: SplitBatch, match, mod_sgms,
                        distance, plot_root, idx):

    os.makedirs(plot_root, exist_ok=True)
    for i in range(samp.batch_size):
        for j in range(samp.num_var):
            plot_path = os.path.join(plot_root, f'{idx}_{i}_{j}.png')

            plot_modulate(samp.series[i][j], proto.series[i][j],
                          mod_sgms[i][j], samp.split[i][j],
                          proto.split[i][j], match[i][j],
                          distance[i][j], plot_path)



def Modulate_batch(samp: SplitBatch, proto: SplitBatch, match, plot_root='', batch_idx=0):

    with cf.ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = [executor.submit(modulate_sample,
                                   samp.segments[i], proto.segments[i], match[i])
                   for i in range(samp.batch_size)]

        results = [f.result() for f in futures]

    mod_sgms, mod_series, shape_dist  = zip(*results)

    mod_series = torch.stack(mod_series, dim=0)
    # tensor(batch_size, num_var, seq_len)

    shape_dist = list2tensor(shape_dist, 0).to(proto.series.device)
    # tensor(batch_size, num_var, K)

    (samp_angle, samp_phase_median, samp_phase_range, samp_magnit_median,
     samp_magnit_range) = info_padding(samp.information, 0)

    (proto_angle, proto_phase_median, proto_phase_range, proto_magnit_median,
     proto_magnit_range) = info_padding(proto.information, 0)
    # tensor(batch_size, num_var, K)

    dist_angle, phase_shift, phase_scale, magnit_shift, magnit_scale \
        = segment_distance(samp_angle, samp_phase_median, samp_phase_range,
                     samp_magnit_median, samp_magnit_range,
                     proto_angle, proto_phase_median, proto_phase_range,
                     proto_magnit_median, proto_magnit_range)
    # tensor(batch_size, num_var, K)

    distance = torch.stack([shape_dist,
                            magnit_scale, magnit_shift,
                            phase_scale, phase_shift], dim=2)
    # tensor(batch_size, num_var, 6, K)

    distance_sum = torch.sum(torch.sum(distance, dim=-1), dim=1)
    # tensor(batch_size, 6)
    #distance_sum = normalize(distance_sum, dim=1)
    if plot_root != '':
        plot_modulate_batch(samp, proto, match, mod_sgms, distance, plot_root, batch_idx)

    return distance_sum, mod_series


def Align_batch(samp, proto):
    with cf.ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = [executor.submit(modulate_sample,
                                   samp.segments[i], proto.segments[i])
                   for i in range(samp.num)]

        results = [f.result() for f in futures]

    mod_segments, d_shape = zip(*results)
    # [num_sample, num_var, num_seg, tensor()]
    d_shape = list2tensor(d_shape, 0)
    # tensor(batch_size, num_var)
    return mod_segments, d_shape

