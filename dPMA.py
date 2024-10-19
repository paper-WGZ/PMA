import torch
import torch.nn as nn

from split import *
from match import *
from alignment import *


class SplitSamples:
    def __init__(self, samples: torch.Tensor, split, segments, information):
        """
        sample: (batch_size, num_var, len_seq)
        """
        self.series = samples
        self.num_sample, self.num_var, self.seq_len = samples.shape

        self.split = split
        # split: [batch_size, num_var, num_seg[i][j] + 1]

        self.num_sgm = [[len(s) - 1 for s in sp] for sp in self.split]  # number of segments
        # num_sgm: [batch_size, num_var]

        self.segments, self.information = segments, information
        # segments: [batch_size, num_var, num_seg[i][j], tensor()]
        # information: [batch_size, num_var, tensor(5, num_seg[i][j])],
        # angle, phase_median, phase_range, magnit_median, magnit_range

    def __getitem__(self, i):
        samples = self.series[i].unsqueeze(0)
        split = [self.split[i]]
        segments = [self.segments[i]]
        information = [self.information[i]]
        return SplitSamples(samples, split, segments, information)


class MatchSamples(SplitSamples):
    def __init__(self, samp: SplitSamples, path):
        """
        samp: A sample
        path: [1, num_var, num_match]
        """
        super().__init__(samp.series, samp.split, samp.segments, samp.information)
        self.path = path

    def append(self, samp: SplitSamples, path):
        """
        samp: A sample
        path: [1, num_var, num_match]
        """
        self.num_sample += 1
        self.series = torch.cat((self.series, samp.series), dim=0)
        self.split += samp.split
        self.num_sgm += samp.num_sgm
        self.segments += samp.segments
        self.information += samp.information
        self.path += path

    def update_by_split(self, split):
        self.split = split
        self.num_sgm = [[len(s) - 1 for s in sp] for sp in split]
        self.segments, self.information = batch_info(self.series, split)





def pma_distance_sp(samp:SplitBatch, proto:SplitBatch, args, save_path='', batch_idx=0):

    if save_path != '':
        match, samp, proto = Match_batch(samp, proto, args,
                                         '', batch_idx)
        distance, _ = Modulate_batch(samp, proto, match,
                                     save_path + '/modulate/', batch_idx)
    else:
        match, samp, proto = Match_batch(samp, proto, args)
        distance, _ = Modulate_batch(samp, proto, match)

    # tensor(batch_size, 5),
    # shape_distance, magnit_scale, magnit_shift, phase_scale, phase_shift
    return distance

def pma_distance(samp:torch.Tensor, proto:torch.Tensor, args, save_path=''):
    # split
    samp = SplitSample(samp, args)
    proto = SplitSample(proto, args)
    return pma_distance_sp(samp, proto, args, save_path)


def pma_align_sp(samp:SplitBatch, proto:SplitBatch, args, save_path=''):
    match, samp, proto = Match(samp, proto, args)
    modulate = ModulateSample(samp, proto, match, save_path)
    return modulate.segments_mod

def pma_align(samp:torch.Tensor, proto:torch.Tensor, args, save_path=''):
    # split
    samp = SplitSample(sample, args)
    proto = SplitSample(prototype, args)
    return pma_align_sp(samp, proto, args, save_path)


def pma_sp(samp:SplitBatch, proto:SplitBatch, args, save_path=''):
    match, samp, proto = Match(samp, proto, args)
    modulate = ModulateSample(samp, proto, match, save_path)
    return modulate

def pma(samp:torch.Tensor, proto:torch.Tensor, args, save_path=''):
    # split
    samp = SplitSample(samp, args)
    proto = SplitSample(proto, args)
    return pma_sp(samp, proto, args, save_path)

