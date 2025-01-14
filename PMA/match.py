import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import copy
import concurrent.futures as cf
import multiprocessing as mp
import os
import time

from .split import *


def pad2len(x, len_new, pad_value=float('inf')):
    """
    x: a two-dimensional tensor
    """
    n, l = x.shape
    if l >= len_new: return x
    pad = torch.full([n, len_new - l], pad_value, dtype=torch.float, device=x.device)
    return torch.cat((x, pad), dim=1)


def list2tensor(batch, pad_value):
    """
    batch: list of tensor: [batch_size, num_var, tensor()]
    """
    K = max(x.size(-1) for samp in batch for x in samp)
    batch = [[F.pad(x, (0, K - x.size(-1)), value=pad_value) for x in samp] for samp in batch]

    batch = torch.stack([torch.stack(samp) for samp in batch])
    return batch

def info_padding(information, pad_value=float('inf')):

    information = list2tensor(information, pad_value)
    # information: tensor(batch_size, num_var, 5, K)

    angle = information[:, :, 0, :]
    phase_median = information[:, :, 1, :]
    phase_range = information[:, :, 2, :]
    magnit_median = information[:, :, 3, :]
    magnit_range = information[:, :, 4, :]
    # tensor(batch_size, num_var, K)

    return angle, phase_median, phase_range, magnit_median, magnit_range

def segment_distance(samp_angle, samp_phase_median, samp_phase_range,
                     samp_magnit_median, samp_magnit_range,
                     proto_angle, proto_phase_median, proto_phase_range,
                     proto_magnit_median, proto_magnit_range, epsilon=0.1):

    dist_angle = torch.abs(proto_angle - samp_angle)

    phase_shift = torch.abs(proto_phase_median - samp_phase_median)
    magnit_shift = torch.abs(proto_magnit_median - samp_magnit_median)

    phase_scale = (torch.abs(proto_phase_range - samp_phase_range))

    magnit_scale = (torch.abs(proto_magnit_range - samp_magnit_range))

    """phase_scale = (torch.abs(proto_phase_range - samp_phase_range) /
                   torch.max(torch.max(torch.abs(proto_phase_range),
                                       torch.abs(samp_phase_range)),
                             torch.tensor(epsilon)))


    magnit_scale = (torch.abs(proto_magnit_range - samp_magnit_range) /
                    torch.max(torch.max(torch.abs(proto_magnit_range),
                                        torch.abs(samp_magnit_range)),
                              torch.tensor(epsilon)))"""

    return dist_angle, phase_shift, phase_scale, magnit_shift, magnit_scale


def init_similar(samp_info, proto_info,
                 mat_ang, mat_phs_sft, mat_phs_scl, mat_mgn_sft, mat_mgn_scl, num_candidate=1):
    """
    info: [batch_size, num_var, tensor(5, num_seg[j])],
    angle, phase_median, phase_range, magnit_median, magnit_range
    """
    # 通过广播机制并行计算
    (samp_angle, samp_phase_median, samp_phase_range, samp_magnit_median,
     samp_magnit_range) = info_padding(samp_info, float('inf'))
    # shape: (batch_size, num_var, samp_max_num_seg)

    (proto_angle, proto_phase_median, proto_phase_range, proto_magnit_median,
     proto_magnit_range) = info_padding(proto_info, float('inf'))
    # shape: (batch_size, num_var, proto_max_num_seg)

    dist_angle, phase_shift, phase_scale, magnit_shift, magnit_scale \
        = segment_distance(samp_angle[:, :, None, :], samp_phase_median[:, :, None, :], samp_phase_range[:, :, None, :],
                           samp_magnit_median[:, :, None, :], samp_magnit_range[:, :, None, :],
                           proto_angle[:, :, :, None], proto_phase_median[:, :, :, None], proto_phase_range[:, :, :, None],
                           proto_magnit_median[:, :, :, None], proto_magnit_range[:, :, :, None])
    # shape: (batch_size, num_var, proto_max_num_seg, samp_max_num_seg)

    '''dist = (normalize(dist_angle, dim=-1)
            + normalize(magnit_scale, dim=-1) + normalize(magnit_shift, dim=-1)
            + normalize(phase_scale, dim=-1) + normalize(phase_shift, dim=-1))
    '''

    dist = (mat_ang * dist_angle
            + mat_phs_sft * phase_shift + mat_phs_scl * phase_scale
            + mat_mgn_sft * magnit_shift + mat_mgn_scl * magnit_scale)
    # shape: (batch_size, num_var, proto_max_num_seg, samp_max_num_seg)

    batch_size, num_var, proto_num_seg, samp_num_seg = dist.shape
    # 生成 proto_seg 的索引并扩展到目标形状
    proto_indices = torch.arange(proto_num_seg, device=dist.device).view(1, 1, -1, 1)
    proto_indices = proto_indices.expand(batch_size, num_var, proto_num_seg, num_candidate)

    # 通过 topk 获取最小的 num_candidate 个 samp_seg 的索引
    _, samp_indices = torch.topk(dist, num_candidate, dim=-1, largest=False, sorted=False)

    # 将 proto_indices 和 samp_indices 拼接
    proto2samp_indices = torch.stack((samp_indices, proto_indices), dim=-1)
    proto2samp_indices = proto2samp_indices.view(batch_size, num_var, -1, 2)
    # shape: (batch_size, num_var, proto_max_num_seg * num_candidate, 2)

    # 生成 proto_seg 的索引并扩展到目标形状
    samp_indices = torch.arange(samp_num_seg, device=dist.device).view(1, 1, 1, -1)
    samp_indices = samp_indices.expand(batch_size, num_var, num_candidate, samp_num_seg)

    # 通过 topk 获取最小的 num_candidate 个 proto_seg 的索引
    _, proto_indices = torch.topk(dist, num_candidate, dim=-2, largest=False, sorted=False)

    # 将 proto_indices 和 samp_indices 拼接
    samp2proto_indices = torch.stack((samp_indices, proto_indices), dim=-1)
    samp2proto_indices = samp2proto_indices.view(batch_size, num_var, -1, 2)
    # shape: (batch_size, num_var, num_candidate * samp_max_num_seg, 2)

    d = torch.abs(proto2samp_indices[:, :, :, None, :] - samp2proto_indices[:, :, None, :, :])
    d = torch.sum(d, dim=-1)
    # shape: (batch_size, num_var, proto_max_num_seg * num_candidate, num_candidate * samp_max_num_seg)

    indices = torch.nonzero(d == 0)


    similar = [[[] for _ in range(num_var)] for _ in range(batch_size)]
    for (i, j, k_p, k_s) in indices:
        k_p = torch.div(k_p, num_candidate, rounding_mode='floor')
        k_s = torch.div(k_s, num_candidate, rounding_mode='floor')
        d = dist[i, j, k_p, k_s]
        if not torch.isnan(d) and not torch.isinf(d):
            similar[i][j].append((k_s.item(), k_p.item()))

    return similar, dist


def draw_graph(g):
    # 使用 spring_layout 并增加 k 值以拉开节点距离
    pos = nx.spring_layout(g, k=0.7, iterations=100)

    # 或者尝试其他布局
    #pos = nx.shell_layout(g)
    #pos = nx.circular_layout(g)
    #pos = nx.kamada_kawai_layout(g)

    # 绘制图的节点和边
    plt.figure(figsize=(12, 8))  # 调整图形大小
    nx.draw(g, pos, with_labels=True, edge_color='b', node_color='g', node_size=3000, font_size=10)

    # 绘制边的权重（标签）
    edge_labels = nx.get_edge_attributes(g, 'weight')
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, label_pos=0.5, font_size=9, font_color='red')

    # 显示图形
    plt.show()

def sim2path(sim, samp_sp, proto_sp):
    """
    sim: [(k_s, k_p)]
    """
    g = nx.DiGraph()
    start_node, end_node = (-1, -1), (len(samp_sp) - 1, len(proto_sp) - 1)
    sim.append(end_node)

    g.add_node(start_node)
    leaf_nodes = set([start_node])
    for new_node in sim:
        g.add_node(new_node)
        r_s, r_p = new_node[0], new_node[1]

        check_nodes = leaf_nodes.copy()

        # 遍历每个节点，查找它的前驱节点
        while check_nodes:
            current_node = check_nodes.pop()
            l_s, l_p = current_node[0], current_node[1]
            if r_s > l_s and r_p > l_p:
                #weight = - (samp_sp[r_s + 1] - samp_sp[r_s] + proto_sp[r_p + 1] - proto_sp[r_p])
                gap = (samp_sp[r_s] - samp_sp[l_s + 1] + proto_sp[r_p] - proto_sp[l_p + 1])
                g.add_edge(current_node, new_node, weight=gap)
                leaf_nodes.discard(current_node)

            else:
                # 获取当前节点的前驱节点添加到 check_nodes
                predecessors = list(g.predecessors(current_node))
                check_nodes.update(predecessors)

        leaf_nodes.add(new_node)

    # 绘图
    #draw_graph(g)
    #print(g.edges(data=True))
    # dijkstra 算法寻找最优路径
    path = nx.shortest_path(g, source=start_node, target=end_node,
                            weight='weight', method='dijkstra')

    return path[1: -1]



def fine_path(dist, samp_sp, proto_sp, l_s=0, l_p=0):
    """
    proto_series: all proto_series
    proto_sp: all proto_split[past_p: now_p + 1]
    samp_series: all samp_series
    samp_sp: all samp_split[past_s: now_s + 1]
    dist: all dist[past_p: now_p + 1, past_s: now_s + 1]
    """
    #print(dist)
    proto_num_seg, samp_num_seg = dist.shape
    proto_indices = torch.arange(proto_num_seg, device=dist.device).view(-1, 1)
    samp_indices = torch.argmin(dist, dim=-1).view(-1, 1)  # shape: (proto_num_seg, 1)
    proto2samp_indices = torch.cat((samp_indices, proto_indices), dim=-1)
    # shape: (proto_num_seg, 2)

    samp_indices = torch.arange(samp_num_seg, device=dist.device).view(-1, 1)
    proto_indices = torch.argmin(dist, dim=-2).view(-1, 1)  # shape: (samp_num_seg, 1)
    samp2proto_indices = torch.cat((samp_indices, proto_indices), dim=-1)
    # shape: (samp_num_seg, 2)

    d = torch.abs(proto2samp_indices[:, None, :] - samp2proto_indices[None, :, :])
    d = torch.sum(d, dim=-1)
    # shape: (proto_num_seg, samp_num_seg)

    indices = torch.nonzero(d == 0)

    sim = []
    for (k_p, k_s) in indices:
        d = dist[k_p, k_s]
        if not torch.isnan(d) and not torch.isinf(d):
            sim.append((k_s.item(), k_p.item()))

    #start_node, end_node = (0, 0), (samp_num_seg - 1, proto_num_seg -1)
    path = sim2path(sim, samp_sp, proto_sp)
    path = [(l_s + k_s, l_p + k_p) for (k_s, k_p) in path]
    return path


def fill_path(path, dist, samp_sp, proto_sp):
    num_seg_p, num_seg_s = len(samp_sp) - 1, len(proto_sp) - 1

    new_path = []
    now_s, now_p = path[0]

    if now_s > 0 and now_p > 0:
        p = fine_path(dist[: now_p, 0: now_s], samp_sp[: now_s + 1], proto_sp[: now_p + 1])
        new_path += p
    else:
        new_path.append(path[0])

    for pair in path[1:]:
        past_s, past_p = now_s, now_p
        now_s, now_p = pair
        if now_s - past_s > 1 and now_p - past_p > 1:
            p = fine_path(dist[past_p + 1: now_p, past_s + 1: now_s],
                          samp_sp[past_s + 1: now_s + 1], proto_sp[past_p + 1: now_p + 1],
                          past_s + 1, past_p + 1)
            new_path += p
        else:
            new_path.append(pair)

    '''if num_seg_s - now_s > 1 and num_seg_p - now_p > 1:
        print(num_seg_s, num_seg_p, now_s, now_p, dist[now_p + 1: , now_s + 1: ])
        p = fine_path(dist[now_p + 1: , now_s + 1: ],
                      samp_sp[now_s + 1: ], proto_sp[now_p + 1: ],
                      now_s + 1, now_p + 1)
        new_path += p'''

    len_match = 0
    for (k_s, k_p) in new_path:
        len_match += (samp_sp[k_s + 1] - samp_sp[k_s]
                      + proto_sp[k_p + 1] - proto_sp[k_p])
    unmatch = 1 - len_match / (samp_sp[-1] + proto_sp[-1])
    return new_path, unmatch


def get_path(similar, distance, samp_split, proto_split):
    path_list, unmatch_list = [], []
    for j in range(len(similar)):
        path = sim2path(similar[j], samp_split[j], proto_split[j])
        path, unmatch = fill_path(path, distance[j], samp_split[j], proto_split[j])
        path_list.append(path)
        unmatch_list.append(unmatch)
    return path_list, unmatch_list


def more2two(series, split):
    index = torch.linspace(0, 1, len(series), device=series.device)

    past_l, past_r, now_l, now_r = split[0], split[1], split[-2], split[-1]

    divide, _ = max_vertical(series, past_r, now_l + 1)
    dist_pl = angle_distance(index[past_l: past_r + 1], series[past_l: past_r + 1],
                             index[past_r: divide + 1], series[past_r: divide + 1])
    dist_nl = angle_distance(index[now_l: now_r + 1], series[now_l: now_r + 1],
                             index[past_r: divide + 1], series[past_r: divide + 1])

    dist_pr = angle_distance(index[past_l: past_r + 1], series[past_l: past_r + 1],
                             index[divide: now_l + 1], series[divide: now_l + 1])
    dist_nr = angle_distance(index[now_l: now_r + 1], series[now_l: now_r + 1],
                             index[divide: now_l + 1], series[divide: now_l + 1])

    if dist_pl <= dist_nl and dist_pr >= dist_nr:
        divide = min(split, key=lambda sp: abs(sp - divide))
    elif dist_pl <= dist_nl and dist_pr <= dist_nr:
        divide = now_l
    elif dist_pl >= dist_nl and dist_pr >= dist_nr:
        divide = past_r
    else:
        dist_p = angle_distance(index[past_l: past_r + 1], series[past_l: past_r + 1],
                                index[past_r: now_l + 1], series[past_r: now_l + 1])
        dist_n = angle_distance(index[now_l: now_r + 1], series[now_l: now_r + 1],
                                index[past_r: now_l + 1], series[past_r: now_l + 1])
        if dist_p <= dist_n: divide = now_l
        else: divide = past_r
    return divide


def fine_tuning_split(past_list, proto_series, proto_split, samp_series, samp_split):
    """
    proto_series: tensor(num_var, seq_len)
    """

    new_samp_split, new_proto_split = [], []
    end_s, end_p = len(samp_series[0]) - 1, len(proto_series[0]) - 1

    for j in range(len(proto_series)):
        path = past_list[j]
        # path = sim2path1(similar[j], samp_split[j], proto_split[j])
        # print(path)
        if path == []:
            samp_split[j], proto_split[j] = [0, end_s], [0, end_p]
            continue
        # the first segment
        now_s, now_p = path[0][0], path[0][1]
        now_s_l, now_s_r = samp_split[j][now_s], samp_split[j][now_s + 1]
        now_p_l, now_p_r = proto_split[j][now_p], proto_split[j][now_p + 1]
        if now_s > 0 and now_p > 0:
            new_samp_sp, new_proto_sp = [0, now_s_l, now_s_r], [0, now_p_l, now_p_r]
        else:
            new_samp_sp, new_proto_sp = [0, now_s_r], [0, now_p_r]

        # the median segment
        for pair in path[1:]:
            past_s, past_p = now_s, now_p
            past_s_l, past_s_r = new_samp_sp[-2], new_samp_sp[-1]
            past_p_l, past_p_r = new_proto_sp[-2], new_proto_sp[-1]

            now_s, now_p = pair[0], pair[1]
            now_s_l, now_s_r = samp_split[j][now_s], samp_split[j][now_s + 1]
            now_p_l, now_p_r = proto_split[j][now_p], proto_split[j][now_p + 1]

            if now_s - past_s > 1 and now_p - past_p > 1:
                new_samp_sp[-1] = more2two(samp_series[j], samp_split[j][past_s: now_s + 2])
                new_proto_sp[-1] = more2two(proto_series[j], proto_split[j][past_p: now_p + 2])

            elif now_s - past_s == 0 or now_p - past_p == 0:
                new_proto_sp.pop(-1)
                new_samp_sp.pop(-1)

            elif now_s - past_s > 1 and now_p - past_p == 1:
                new_samp_sp[-1] = more2two(samp_series[j], samp_split[j][past_s: now_s + 2])

            elif now_s - past_s == 1 and now_p - past_p > 1:
                new_proto_sp[-1] = more2two(proto_series[j], proto_split[j][past_p: now_p + 2])

            new_samp_sp.append(now_s_r)
            new_proto_sp.append(now_p_r)

        # the end segment
        if new_samp_sp[-1] != end_s and new_proto_sp[-1] != end_p:
            new_samp_sp.append(end_s)
            new_proto_sp.append(end_p)
        else:
            new_samp_sp[-1] = end_s
            new_proto_sp[-1] = end_p

        new_samp_split.append(new_samp_sp)
        new_proto_split.append(new_proto_sp)

    return new_samp_split, new_proto_split


def match_sample(similar, distance, samp_split, proto_split, samp_series, proto_series):
    path_list, rate_mismatch = get_path(similar, distance, samp_split, proto_split)
    new_samp_split, new_proto_split = fine_tuning_split(path_list, proto_series, proto_split, samp_series, samp_split)
    return path_list, rate_mismatch, new_samp_split, new_proto_split



def auto_cast(samp, proto):

    def one2batch(batch, batch_size):
        """
        batch.series: [1, num_var, seq_len] to [batch_size, num_var, seq_len]
        """
        batch.num = batch_size
        batch.series = batch.series.repeat(batch_size, 1, 1)
        batch.split = [batch.split[0] for _ in range(batch_size)]
        batch.segments = [batch.segments[0] for _ in range(batch_size)]
        return batch

    if samp.num != proto.num:
        if samp.num == 1:
            samp = one2batch(copy.deepcopy(samp), proto.num)
        elif proto.num == 1:
            proto = one2batch(copy.deepcopy(proto), samp.num)

    return samp, proto


