import torch
import torch.nn as nn
import copy
import os
from datetime import datetime

import concurrent.futures as cf
import multiprocessing as mp

from . import args
from .split import *
from .match import *
from .align import *
from .visualization import plot_pma

class ASample:
    def __init__(self, samples: torch.Tensor, split,  segments, information):
        """
        samples: (1, num_var, len_seq)
        split: [1, num_var, num_seg[i][j] + 1]
        segments: [1, num_var, num_seg[i][j], tensor()]
        information: [1, num_var, tensor(5, num_seg[i][j])],
        """
        self.series = samples
        self.num, self.num_var, self.seq_len = samples.shape

        self.split = split
        self.num_sgm = [[len(s) - 1 for s in sp] for sp in self.split]  # number of segments
        # num_sgm: [1, num_var]

        self.segments, self.information = segments, information
        # angle, phase_median, phase_range, magnit_median, magnit_range

    def cast(self, num):
        self.num = num
        self.series = self.series.repeat(num, 1, 1)
        self.split = [self.split[0] for _ in range(num)]
        self.num_sgm = [self.num_sgm[0] for _ in range(num)]
        self.segments = [self.segments[0] for _ in range(num)]
        self.information = [self.information[0] for _ in range(num)]

    def __getitem__(self, i):
        samples = self.series[i].unsqueeze(0)
        split = [self.split[i]]
        segments = [self.segments[i]]
        information = [self.information[i]]
        return ASample(samples, split, segments, information)


class SplitSamples(ASample):
    def __init__(self, samples: torch.Tensor, split):
        """
        samples: (batch_size, num_var, len_seq)
        split: [batch_size, num_var, num_seg[i][j] + 1]
        """
        self.series = samples
        self.num, self.num_var, self.seq_len = samples.shape
        self.split = split

        self.num_sgm = [[len(s) - 1 for s in sp] for sp in self.split]  # number of segments
        # num_sgm: [batch_size, num_var]

        self.segments, self.information = batch_info(samples, split)
        # segments: [batch_size, num_var, num_seg[i][j], tensor()]
        # information: [batch_size, num_var, tensor(5, num_seg[i][j])],
        # angle, phase_median, phase_range, magnit_median, magnit_range


class MatchSamples:
    def __init__(self, samp:SplitSamples, proto:SplitSamples,
                 fine_split_samp, fine_split_proto, unmatch):
        self.samp = samp
        self.proto = proto
        self.match_samp = SplitSamples(samp.series, fine_split_samp)
        self.fine_proto = SplitSamples(proto.series, fine_split_proto)
        self.unmatch = unmatch



def get_split(batch):
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


def get_match(samp, proto):
    """
    samp_batch, proto_batch: [batch_size * SplitSample]
    """
    similar, distance = init_similar(samp.information, proto.information,
                                     args.match_angle, args.match_phase_shift,
                                     args.match_phase_scale, args.match_magnit_shift,
                                     args.match_magnit_scale)

    samp, proto = auto_cast(samp, proto)

    with cf.ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = [executor.submit(match_sample, similar[i], distance[i],
                                   samp.split[i], proto.split[i],
                                   samp.series[i], proto.series[i])
                   for i in range(samp.num)]

        results = [f.result() for f in futures]
    path_list, rate_mismatch, match_samp_split, match_proto_split = zip(*results)

    samp_match = SplitSamples(samp.series, match_samp_split)
    proto_match = SplitSamples(proto.series, match_proto_split)
    return path_list, rate_mismatch, samp_match, proto_match


def get_align(samp_match, proto_match, proto):

    with cf.ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = [executor.submit(align_sample, samp_match[i], proto_match[i],
                                   proto[i])
                   for i in range(samp_match.num)]

        results = [f.result() for f in futures]

    aligned_segments, distances, original_information, alignment_pairs = zip(*results)

    return aligned_segments, distances, original_information, alignment_pairs


def plot_align(samp, proto, match_path, samp_match, proto_match,
               align_path, samp_align, root_plot):
    os.makedirs(root_plot, exist_ok=True)
    for i in range(samp.num):
        for j in range(samp.num_var):
            time = datetime.now().strftime('%d%H%M%S%f')
            plot_path = os.path.join(root_plot, f'{i}_{j}_{time}.png')
            plot_pma(samp.series[i][j], proto.series[i][j], samp.split[i][j], proto.split[i][j],
                     match_path[i][j], samp_match.split[i][j], proto_match.split[i][j],
                     align_path[i][j], samp_align[i][j], plot_path)
    return True


def auto_cast(samp, proto):
    if samp.num != proto.num:
        if samp.num == 1: samp.cast(proto.num)
        elif proto.num == 1: proto.cast(samp.num)
    return samp, proto


def pma_sp(samp: SplitSamples, proto: SplitSamples):
    samp, proto = auto_cast(samp, proto)
    match_path, rate_mismatch, samp_match, proto_match = get_match(samp, proto)
    aligned_segments, distances, original_information, alignment_pairs = get_align(samp_match, proto_match, proto)

    if args.is_plot:
        root_plot = f"{args.root}{args.data_name}/figure/"
        plot_align(samp, proto, match_path, samp_match, proto_match,
                   alignment_pairs, aligned_segments, root_plot)

    return aligned_segments, distances, original_information, alignment_pairs

def pma(samp: torch.Tensor, proto: torch.Tensor):
    # split
    samp_sp, samp_smooth = get_split(samp)
    proto_sp, proto_smooth = get_split(proto)
    samp = SplitSamples(samp, samp_sp)
    proto = SplitSamples(proto, proto_sp)
    return pma_sp(samp, proto)


