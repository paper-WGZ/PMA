import os
import copy
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from split import *
from match import *
from fine_tuning import *
from alignment import *


class object_sample:
    def __init__(self, series:torch.Tensor, label, args):
        self.seq_len = len(series)
        self.Series = series    # raw series
        self.Label = label  # label
        self.Split = GetSplit(series, args)    # segment point list
        self.num_seg = len(self.Split) - 1   # number of segments
        self.Segment, self.Angle = SplitSeries(series, self.Split)

        self.idx_proto = 0
        self.Match = [-1] * self.num_seg


    def update_by_seg(self, segment):
        self.Segment = segment
        self.Series = MergeSegments(segment)
        self.seq_len = len(self.Series)

        self.Split[1] = len(segment[0])
        for k in range(self.num_seg):
            sgm = segment[k]
            index = torch.arange(0, len(sgm)) / self.seq_len
            self.Angle[k] = slant_angle(index, sgm)
            if k >= 2:
                self.Split[k] = self.Split[k -1] + len(segment[k - 1]) - 1


class SplitSamples:
    def __init__(self, samples: torch.Tensor, split, segments, information):
        """
        sample: (batch_size, num_var, len_seq)
        """
        self.series = samples
        self.num, self.num_var, self.seq_len = samples.shape

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
        self.alignment = []
        self.d_shape, self.d_phase, self.d_magnit = torch.tensor([[], [], []], dtype=torch.float)


    def append(self, samp: SplitSamples, path):
        """
        samp: A sample
        path: [1, num_var, num_match]
        """
        self.num += samp.num
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


class Prototypes:
    def __init__(self, samp: SplitSamples, label=0):
        """
        samp: A sample
        """
        self.label = label
        self.num = 0
        self.num_var, self.seq_len = samp.num_var, samp.seq_len
        self.series = []
        self.split = []
        self.num_sgm = []
        self.segments = []
        self.information = []
        self.match_samples = []

        self.append(samp)

    def append(self, samp: SplitSamples):
        """
        samp: A sample
        """
        # prototypes
        self.num += samp.num
        self.series += samp.series
        self.split += samp.split
        self.num_sgm += samp.num_sgm
        self.segments += samp.segments
        self.information += samp.information

        # matched samples
        path = [[[(k, k) for k in range(samp.num_sgm[0][j])]
                 for j in range(samp.num_var)]]
        self.match_samples.append(MatchSamples(samp, path))

    def __getitem__(self, i):
        samples = self.series[i].unsqueeze(0)
        split = [self.split[i]]
        segments = [self.segments[i]]
        information = [self.information[i]]
        return SplitSamples(samples, split, segments, information)




def init_prototypes(samples, args, label=0):
    """

    :param samples: [[split_sample,..., split_sample], ...,[...]]
    :param match_window:
    :param match_thres:
    :param max_unmatch:
    :param max_dist:
    :return:
    """

    prototypes = Prototypes(copy.deepcopy(samples[0]), label)

    for i in range(1, samples.num):
        samp = samples[i]
        path, unmatch = Match_batch(samp, prototypes, args)

        opt = min(range(prototypes.num), key=lambda i: sum(unmatch[i]))

        if sum(unmatch[opt]) > args.thres_unmatch:
            prototypes.append(copy.deepcopy(samp))
        else:
            prototypes.match_samples[opt].append(samp, [path[opt]])

    return prototypes


def update_prototypes(prototypes):
    for i in range(prototypes.num):
        proto = prototypes[i]
        samp = prototypes.match_samples[i]
        path = samp.path

        samp, proto, d_phase, d_magnit = Fine_batch(samp, proto, path)
        align, d_shape = Align_batch(samp, proto)
        # align: [num_sample, num_var, num_seg, tensor()]

        samp.alignment = align
        samp.d_phase = d_phase
        samp.d_magnit = d_magnit
        samp.d_shape = d_shape

        for j in range(proto.num_var):
            for k in range(proto.num_sgm[0][j]):
                all_sgm = [align[s][j][k] for s in range(samp.num)]
                average_sgm = torch.mean(torch.stack(all_sgm, dim=0), dim=0)

                # information: [num_sample, num_var, tensor(5, num_seg[i][j])], angle, phase_median, phase_range, magnit_median, magnit_range
                all_info = [samp.information[s][j][:, k] for s in range(samp.num)]
                average_info = torch.mean(torch.stack(all_info, dim=1), dim=1)

                t_range, v_median, v_range = average_info[2:]
                sgm = adjust_segment(average_sgm, t_range, proto.seq_len, v_median, v_range)

                prototypes.segments[i][j][k] = sgm
                prototypes.information[i][j][:, k] = average_info

    return prototypes


def get_prototypes(samples, labels, args):
    num_samples, num_var, seq_len = samples.shape
    labels = labels.tolist()

    # 按类别分组
    label_set, sample_set = [], []
    for i in range(num_samples):
        if labels[i] in label_set:
            idx = label_set.index(labels[i])
            sample_set[idx].append(samples[i])
        else:
            label_set.append(labels[i])
            sample_set.append([samples[i]])

    prototype_set = []
    for c in range(len(label_set)):
        samp = torch.stack(sample_set[c], dim=0)
        samp = split_batch(samp, args)
        proto = init_prototypes(samp, args, label_set[c])
        proto = update_prototypes(proto)
        prototype_set.append(proto)
        sample_set[c] = samp

    return prototype_set, label_set, sample_set


def evaluate_prototypes(prototype_set):
    d_shape, d_phase, d_magnit = 0, 0, 0
    num_sample = 0
    for prototypes in prototype_set:
        for i in range(prototypes.num):
            match_samples = prototypes.match_samples[i]
            num_sample += match_samples.num

            d_shape += torch.sum(match_samples.d_shape).item()
            d_phase += torch.sum(match_samples.d_phase).item()
            d_magnit += torch.sum(match_samples.d_magnit).item()

    return d_shape / num_sample, d_phase / num_sample, d_magnit / num_sample


def plot_prototypes(prototype_set, plot_path):

    for c in range(len(prototype_set)):
        num_prop = len(prototype_set[c])
        fig, axs = plt.subplots(num_prop, 1, figsize=(10, num_prop))

        for p, proto in enumerate(prototype_set[c]):
            index = np.linspace(0, 1, proto.seq_len)

            ax = axs[p]
            ax.plot(index, proto.Series)
            #ax.set_title("")
            ax.set_xticks([])  # 隐藏x轴刻度
            ax.set_yticks([])  # 隐藏y轴刻度
        fig.suptitle(f"Prototypes of Class {c}")

        # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
        # plt.tight_layout()
        # plt.show()

        # 保存图像
        os.makedirs(plot_path, exist_ok=True)
        plt.savefig(os.path.join(plot_path, f'{c}.png'))
        plt.close()


def plot_samples(prototype_set, sample_set, args, plot_path):

    for c in range(len(sample_set)):
        prototype_list = prototype_set[c]
        for i, samp in enumerate(sample_set[c]):
            proto = prototype_list[samp.idx_proto]

            match, _ = SoftMatch(samp.Angle, proto.Angle, args.match_window, args.match_thres)
            modulate_segment, _, _ = ModulateSample(samp.Segment, samp.Split,
                                                    proto.Segment, proto.Split, match)
            modulate_series = MergeSegments(modulate_segment)

            fig, ax = plt.subplots(figsize=(8, 12))

            ax.plot(normalize(samp.Series, mode='min-max') + 2.2)
            ax.vlines(samp.Split[1:-1], 2.2, 3.2, linestyles='dashed', colors='red')

            ax.plot(normalize(proto.Series, mode='min-max') + 1.1)
            ax.vlines(proto.Split[1:-1], 1.1, 2.1, linestyles='dashed', colors='red')

            ax.plot(normalize(modulate_series, mode='min-max'))
            ax.vlines(proto.Split[1:-1], 0, 1, linestyles='dashed', colors='red')

            for j in range(len(match)):
                if match[j] != -1:
                    p1 = (proto.Split[j] + proto.Split[j +1]) / 2
                    p2 = (samp.Split[match[j]] + samp.Split[match[j] + 1]) / 2
                    ax.plot([p1, p2], [2.1, 2.2], 'g--')

            ax.set_xticks([])  # 隐藏x轴刻度
            ax.set_yticks([])  # 隐藏y轴刻度
            plt.title(f'class {c} sample {i} to prototype {samp.idx_proto}')
            # plt.show()

            # 保存图像
            os.makedirs(plot_path, exist_ok=True)
            plt.savefig(os.path.join(plot_path, f'{c}_{i}.png'))
            plt.close()