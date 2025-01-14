import torch
import numpy as np
from scipy import interpolate
import concurrent.futures as cf
import multiprocessing as mp
from scipy.stats import linregress


from dataprocess import normalize
from .match import *
from .split import *

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

def align_segment(sgm_s, sgm_p):
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
            sgm_mod = align_segment(samp_sgms[j][k], proto_sgms[j][k])
            mod_sgms[j].append(sgm_mod)

            len_sgm_s, len_sgm_p = len(samp_sgms[j][k]), len(proto_sgms[j][k])
            dist[k] = torch.mean((proto_sgms[j][k] - sgm_mod) ** 2) * (len_sgm_s + len_sgm_p) / 2
        shape_dist.append(dist)
    shape_dist = torch.stack(shape_dist, dim=0)

    return mod_sgms, shape_dist


def compute_median(range, median, l, r, sp_l, sp_r):
    x_l = range / (r - l) * (sp_l - (l + r) / 2) + median
    x_r = range / (r - l) * (sp_r - (l + r) / 2) + median
    return (x_l + x_r) / 2


def align_series(samp_sgms, proto_sgms, samp_info, proto_info,
               samp_sp, proto_sp, proto_raw_sp):
    """
    return:
    aligned_segments: [tensor(len_seg)] * K, K=len(proto_raw_sp) - 1
    distances, [tensor(6)] * K, d_angle, d_pm, d_pr, d_mm, d_mr, d_shape
    original_information, [tensor(5, K)], angle, p_median, p_range, m_median, m_range
    alignment_pairs, [(0,0), ..., (len_samp, len_proto)]
    """
    aligned_segments, distances, original_information, alignment_pairs = [], [], [], [(0, 0)]

    for k in range(len(proto_sgms)):
        l, r = proto_sp[k], proto_sp[k + 1]
        l_s, r_s = samp_sp[k], samp_sp[k + 1]
        alignment_pairs += [((p - l) / (r - l) * (r_s - l_s) + l_s, p)
                       for p in range(l + 1, r + 1)]


        align_sgm = align_segment(samp_sgms[k], proto_sgms[k])

        # 计算距离，dist: tensor(6), d_angle, d_pm, d_pr, d_mm, d_mr, d_shape
        dist = torch.abs(samp_info[:, k] - proto_info[:, k])
        d_shape = torch.sum(torch.abs(align_sgm - proto_sgms[k]))
        dist = torch.cat([dist, d_shape.unsqueeze(0)])

        # 记录原始信息
        angle, p_median, p_range, m_median, m_range = samp_info[:, k]

        # 调整 align_sgm, dist 和 info 的段数向着 proto_raw_sp
        raw_sp = [sp - l for sp in proto_raw_sp if sp >= l and sp <= r]
        for j in range(len(raw_sp) - 1):
            sp_l, sp_r = raw_sp[j], raw_sp[j + 1]

            # 按 proto_raw_sp 细分 align_sgm
            aligned_segments.append(align_sgm[sp_l: sp_r + 1])

            # 按 proto_raw_sp 细分 dist
            distances.append(dist * (sp_r - sp_l + 1) / (r - l + 1))

            # 按 proto_raw_sp 细分 info
            pm = compute_median(p_range, p_median, l, r, sp_l, sp_r)
            pr = p_range * (sp_r - sp_l + 1) / (r - l + 1)
            mm = compute_median(m_range, m_median, l, r, sp_l, sp_r)
            mr = m_range * (sp_r - sp_l + 1) / (r - l + 1)
            info = torch.stack([angle, pm, pr, mm, mr], dim=0)
            original_information.append(info)

    original_information = torch.stack(original_information, dim=-1) # tensor(5, K)

    return aligned_segments, distances, original_information, alignment_pairs


def align_sample(samp_match, proto_match, proto_raw):
    aligned_segments, distances, original_information, alignment_pairs = [], [], [], []

    for j in range(proto_match.num_var):
        samp_align, dist, info, path = align_series(samp_match.segments[0][j], proto_match.segments[0][j],
                                        samp_match.information[0][j], proto_match.information[0][j],
                                        samp_match.split[0][j], proto_match.split[0][j],
                                        proto_raw.split[0][j])
        aligned_segments.append(samp_align)
        distances.append(dist)
        original_information.append(info)
        alignment_pairs.append(path)
    return aligned_segments, distances, original_information, alignment_pairs


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




