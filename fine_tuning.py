import time

import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import copy
import concurrent.futures as cf
import multiprocessing as mp
import os

from metrics import *
from split import *
from match import *

def more2two(series, past_l, past_r, now_l, now_r):
    index = torch.linspace(0, 1, len(series), device=series.device)

    divide, _ = max_vertical(series, past_r, now_l + 1)
    dist_pl = angle_distance(index[past_l: past_r + 1], series[past_l: past_r + 1],
                            index[past_r: divide + 1], series[past_r: divide + 1])
    dist_nl = angle_distance(index[now_l: now_r + 1], series[now_l: now_r + 1],
                            index[past_r: divide + 1], series[past_r: divide + 1])

    dist_pr = angle_distance(index[past_l: past_r + 1], series[past_l: past_r + 1],
                            index[divide: now_l + 1], series[divide: now_l + 1])
    dist_nr = angle_distance(index[now_l: now_r + 1], series[now_l: now_r + 1],
                            index[divide: now_l + 1], series[divide: now_l + 1])

    if dist_pl <= dist_nl and dist_pr >= dist_nr: divide = divide
    elif dist_pl <= dist_nl and dist_pr <= dist_nr: divide = now_l
    elif dist_pl >= dist_nl and dist_pr >= dist_nr: divide = past_r
    else:
        dist_p = angle_distance(index[past_l: past_r + 1], series[past_l: past_r + 1],
                                index[past_r: now_l + 1], series[past_r: now_l + 1])
        dist_n = angle_distance(index[now_l: now_r + 1], series[now_l: now_r + 1],
                                index[past_r: now_l + 1], series[past_r: now_l + 1])
        if dist_p <= dist_n: divide = now_l
        else: divide = past_r
    return divide



def more2two1(series, split, past, now):
    index = torch.linspace(0, 1, len(series), device=series.device)

    past_l, past_r, now_l, now_r = split[past], split[past + 1], split[now], split[now + 1]

    divide, _ = max_vertical(series, past_r, now_l + 1)
    dist_pl = angle_distance(index[past_l: past_r + 1], series[past_l: past_r + 1],
                            index[past_r: divide + 1], series[past_r: divide + 1])
    dist_nl = angle_distance(index[now_l: now_r + 1], series[now_l: now_r + 1],
                            index[past_r: divide + 1], series[past_r: divide + 1])

    dist_pr = angle_distance(index[past_l: past_r + 1], series[past_l: past_r + 1],
                            index[divide: now_l + 1], series[divide: now_l + 1])
    dist_nr = angle_distance(index[now_l: now_r + 1], series[now_l: now_r + 1],
                            index[divide: now_l + 1], series[divide: now_l + 1])

    if dist_pl <= dist_nl and dist_pr >= dist_nr: divide = divide
    elif dist_pl <= dist_nl and dist_pr <= dist_nr: divide = now_l
    elif dist_pl >= dist_nl and dist_pr >= dist_nr: divide = past_r
    else:
        dist_p = angle_distance(index[past_l: past_r + 1], series[past_l: past_r + 1],
                                index[past_r: now_l + 1], series[past_r: now_l + 1])
        dist_n = angle_distance(index[now_l: now_r + 1], series[now_l: now_r + 1],
                                index[past_r: now_l + 1], series[past_r: now_l + 1])
        if dist_p <= dist_n: divide = now_l
        else: divide = past_r

    div_i = min(range(past, now + 2), key=lambda i: abs(split[i] - divide))
    split_l, split_r = split[past + 1: div_i + 1], split[div_i: now + 2]
    return split_l, split_r


def path2match(path, series_p, split_p, series_s, split_s):
    if path == []: return [[split_s, split_p]]

    # the first segment
    now_s, now_p = path[0]
    match = [[split_s[: now_s + 2], split_p[: now_p + 2]]]

    for pair in path[1:]:
        past_s, past_p = now_s, now_p
        now_s, now_p = pair

        if now_p - past_p == 1 and now_s - past_s == 1:
            match.append([split_s[past_s + 1: now_s + 2], split_p[past_p + 1: now_p + 2]])

        elif now_p - past_p == 1 and now_s - past_s > 1:
            sp_s_l, sp_s_r = more2two(series_s, split_s, past_s, now_s)
            match[-1][0] += sp_s_l[1: ]
            match.append([sp_s_r, split_p[past_p + 1: now_p + 2]])

        elif now_p - past_p > 1 and now_s - past_s == 1:
            sp_p_l, sp_p_r = more2two(series_p, split_p, past_p, now_p)
            match[-1][1] += sp_p_l[1: ]
            match.append([split_s[past_s + 1: now_s + 2], sp_p_r])

        elif now_p - past_p > 1 and now_s - past_s > 1:
            sp_s_l, sp_s_r = more2two(series_s, split_s, past_s, now_s)
            sp_p_l, sp_p_r = more2two(series_p, split_p, past_p, now_p)
            match[-1][0] += sp_s_l[1: ]
            match[-1][1] += sp_p_l[1: ]
            match.append([sp_s_r, sp_p_r])

    if now_s + 1 != len(split_s) - 1: match[-1][0] += split_s[now_s + 2: ]
    if now_p + 1 != len(split_p) - 1: match[-1][1] += split_p[now_p + 2: ]

    return match



def adjust_split1(path, proto_series, proto_split, samp_series, samp_split):
    """
    proto_series: tensor(num_var, seq_len)
    """
    new_samp_split = []

    for j in range(len(proto_series)):
        match = path2match(path[j], proto_series[j], proto_split[j], samp_series[j], samp_split[j])
        new_samp_sp = [0]
        for mat in match:
            split_s, split_p = mat
            if len(split_p) == 2:
                new_samp_sp.append(split_s[-1])
            else:
                for sp_p in split_p[1: ]:
                    sp_s = split_s[0] + round((sp_p - split_p[0]) / (split_p[-1] - split_p[0])
                                              * (split_s[-1] - (split_s[0])))
                    new_samp_sp.append(sp_s)

        new_samp_split.append(new_samp_sp)

    print(samp_split, proto_split, new_samp_split, match)
    return new_samp_sp



def adjust_split(path, proto_series, proto_split, samp_series, samp_split):
    """
    proto_series: tensor(num_var, seq_len)
    """
    new_samp_sp = copy.deepcopy(proto_split)


    for j in range(len(proto_series)):
        if path[j] == []: continue
        # the first segment
        now_s, now_p = path[j][0]
        now_s_l, now_s_r = samp_split[j][now_s], samp_split[j][now_s + 1]
        now_p_l, now_p_r = proto_split[j][now_p], proto_split[j][now_p + 1]

        if now_p == 0:
            new_samp_sp[j][now_p] = 0
        else:
            for k in range(now_p + 1):
                new_samp_sp[j][k] = round(proto_split[j][k] / now_p_r * now_s_r)

        # the median segment
        for pair in path[j][1:]:
            past_s, past_p = now_s, now_p
            past_s_l, past_s_r = samp_split[j][past_s], samp_split[j][past_s + 1]
            past_p_l, past_p_r = proto_split[j][past_p], proto_split[j][past_p + 1]

            now_s, now_p = pair
            now_s_l, now_s_r = samp_split[j][now_s], samp_split[j][now_s + 1]
            now_p_l, now_p_r = proto_split[j][now_p], proto_split[j][now_p + 1]

            if now_p - past_p == 1:
                if now_s - past_s == 1:
                    sp_s = now_s_l
                else:
                    sp_s = more2two(samp_series[j], past_s_l, past_s_r, now_s_l, now_s_r)
                new_samp_sp[j][now_p] = sp_s

            else:
                if now_s - past_s == 1:
                    div_s = now_s_l
                else:
                    div_s = more2two(samp_series[j], past_s_l, past_s_r, now_s_l, now_s_r)
                div_p = more2two(proto_series[j], past_p_l, past_p_r, now_p_l, now_p_r)

                for k in range(past_p + 1, now_p + 1):
                    sp_p = proto_split[j][k]
                    if sp_p <= div_p:
                        sp_s = past_s_l + round((sp_p - past_p_l) / (div_p - past_p_l)
                                                * (div_s - past_s_l))
                    else:
                        sp_s = div_s + round((sp_p - div_p) / (now_p_r - div_p)
                                             * (now_s_r - div_s))
                    new_samp_sp[j][k] = sp_s

        # the end segment
        end_s, end_p = len(samp_series[j]) - 1, len(proto_series[j]) - 1
        if now_p + 1 == len(proto_split[j]) - 1:
            new_samp_sp[j][now_p + 1] = end_s
        else:
            for k in range(now_p + 1, len(proto_split[j])):
                new_samp_sp[j][k] = now_s_l + round((proto_split[j][k] - now_p_l)
                                                    / (end_p - now_p_l) * (end_s - now_s_l))
        #print(samp_split, proto_split, path, new_samp_sp)
    return new_samp_sp


def update_match(samp_info, proto_info,
                 mat_ang, mat_phs_sft, mat_phs_scl, mat_mgn_sft, mat_mgn_scl):
    """
    info: [batch_size, num_var, tensor(5, num_seg[j])],
    angle, phase_median, phase_range, magnit_median, magnit_range
    proto.num_seg[j] == samp.num_seg[j]
    """
    (samp_angle, samp_phase_median, samp_phase_range, samp_magnit_median,
     samp_magnit_range) = info_padding(samp_info, float('inf'))

    (proto_angle, proto_phase_median, proto_phase_range, proto_magnit_median,
     proto_magnit_range) = info_padding(proto_info, float('inf'))
    # tensor(batch_size, num_var, max_num_seg)

    dist_angle, phase_shift, phase_scale, magnit_shift, magnit_scale \
        = segment_distance(samp_angle, samp_phase_median, samp_phase_range,
                           samp_magnit_median, samp_magnit_range,
                           proto_angle, proto_phase_median, proto_phase_range,
                           proto_magnit_median, proto_magnit_range, epsilon=0.05)
    # shape: (batch_size, num_var, max_num_seg)

    # 找到满足条件的匹配
    matches = ((dist_angle <= mat_ang) & (phase_shift <= mat_phs_sft)
               & (phase_scale <= mat_phs_scl) & (magnit_shift <= mat_mgn_sft)
               & (magnit_scale <= mat_mgn_scl))

    matched_indices = matches.nonzero(as_tuple=False)

    # 将匹配对及其距离组合成列表
    batch_size, num_var, _ = dist_angle.shape
    match = [[[] for _ in range(num_var)] for _ in range(batch_size)]

    for (i, j, k) in matched_indices:
        match[i][j].append((k.item(), k.item()))

    return match


def phase_magnitude_dist(samp, proto):
    (samp_angle, samp_phase_median, samp_phase_range, samp_magnit_median,
     samp_magnit_range) = info_padding(samp.information, 0)

    (proto_angle, proto_phase_median, proto_phase_range, proto_magnit_median,
     proto_magnit_range) = info_padding(proto.information, 0)
    # tensor(batch_size, num_var, K)

    _, phase_shift, phase_scale, magnit_shift, magnit_scale \
        = segment_distance(samp_angle, samp_phase_median, samp_phase_range,
                           samp_magnit_median, samp_magnit_range,
                           proto_angle, proto_phase_median, proto_phase_range,
                           proto_magnit_median, proto_magnit_range)
    # tensor(batch_size, num_var, K)
    '''print(len(proto.num_sgm), len(proto.num_sgm[0], proto.num_sgm[0][0]))
    for i in range(proto.num):
        for j in range(proto.num_var):
            for k in range(proto.num_sgm[i][j]):
                l = (len(samp.segments[i][j][k]) + len(proto.segments[i][j][k]) / 2)
                phase_shift[i][j][k] *= l
                phase_scale[i][j][k] *= l
                magnit_shift[i][j][k] *= l
                magnit_scale[i][j][k] *= l'''

    d_phase = torch.stack([phase_scale, phase_shift], dim=-1)
    d_magnit = torch.stack([magnit_scale, magnit_shift], dim=-1)
    # tensor(batch_size, num_var, K, 2)
    return d_phase, d_magnit



def Fine_batch(samp: SplitBatch, proto: SplitBatch, path, plot_root='', batch_idx=0):
    """
    samp_batch, proto_batch: [batch_size * SplitSample]
    """

    samp, proto = auto_cast(samp, proto)

    with cf.ThreadPoolExecutor(max_workers=mp.cpu_count()-7) as executor:
        futures = [executor.submit(adjust_split, path[i],
                                   proto.series[i], proto.split[i],
                                   samp.series[i], samp.split[i])
                   for i in range(samp.num)]

        samp_split = [f.result() for f in futures]


    samp.update_by_split(samp_split)
    d_phase, d_magnit = phase_magnitude_dist(samp, proto)

    if plot_root != '':
        plot_match_batch(samp, proto, path, plot_root + '\match_update', batch_idx)

    return samp, proto, d_phase, d_magnit