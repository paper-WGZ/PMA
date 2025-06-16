import os
import copy
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from PMA0.dPMA import *
from PMA0 import args as args_pma

class ACluster(SplitSamples):
    def __init__(self, sample: ASample):
        self.series = sample.series
        self.num, self.num_var, self.seq_len = self.series.shape
        self.split = sample.split
        self.num_sgm = sample.num_sgm
        self.segments = sample.segments
        self.information = sample.information

    def append(self, sample: ASample):
        self.num += sample.num
        self.series = torch.cat((self.series, sample.series), dim=0)
        self.split += sample.split
        self.num_sgm += sample.num_sgm
        self.segments += sample.segments
        self.information += sample.information

class Clusters:
    def __init__(self, proto_init: SplitSamples):
        self.prototypes = proto_init
        self.samples = [ACluster(copy.deepcopy(proto_init[i]))
                        for i in range(proto_init.num)]

    def append(self, sample: ASample):
        self.prototypes.append(copy.deepcopy(sample))
        self.samples.append(ACluster(copy.deepcopy(sample)))



def init_prototypes(samples, labels, proto_init):
    """
    :param samples: SplitSamples
    :param proto_init: SplitSamples
    :return:
    """

    clusters = Clusters(proto_init)
    label_set = [[i] for i in range(proto_init.num)]

    for i in range(samples.num):
        samp = samples[i]

        match_path, rate_mismatch, samp_match, proto_match = get_match(copy.deepcopy(samp), proto_init)
        opt = min(range(proto_init.num), key=lambda i: sum(rate_mismatch[i]))

        match_path, rate_mismatch, samp_match, proto_match = [match_path[opt]], rate_mismatch[opt], samp_match[opt], proto_match[opt]
        proto_opt = proto_init[opt]

        aligned_segments, _, original_information, align_path = get_align(samp_match, proto_match, proto_opt)
        aligned_samp = ASample(samp.series, proto_opt.split, aligned_segments, original_information)

        clusters.samples[opt].append(aligned_samp)
        label_set[opt].append(labels[i])

        if args_pma.is_plot:
            root_plot = f"{args_pma.root}{args_pma.data_name}/figure/"
            plot_align(samp, proto_opt, match_path, samp_match, proto_match,
                       align_path, aligned_segments, root_plot)

    return clusters, label_set


def adjust_segment1(sgm, t_range, seq_len, v_median, v_range):
    """
    :sgm: raw segment

    """
    len_new = round((seq_len-1) * t_range.item())+1
    t = torch.linspace(0, t_range, len_new, device=sgm.device)
    trend_new = v_range / t_range * t + v_median - v_range / 2

    sgm = interp_resamp(sgm, len_new)
    slope, intercept = linear_fit(t, sgm)
    trend_old = slope * t + intercept

    sgm = sgm - trend_old + trend_new
    return sgm

def adjust_segment(sgm, len_seq, v_median, v_range):
    """
    :sgm: raw segment

    """
    t = torch.linspace(0, 1, len_seq, device=sgm.device)[:len(sgm)]
    trend_new = v_range / t[-1] * t + v_median - v_range / 2

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
    series = normalize(series, mode='min-max')
    return series

def update_prototypes(clusters):
    prototypes = clusters.prototypes
    samples = clusters.samples

    for p in range(prototypes.num):
        samp = samples[p]  # ACluster
        '''for i in range(samp.num):
            print(p, i, samp.num, samp.series.shape, len(samp.segments))
            print(p,i,samp.num, len(samp.segments[i]))
            len_sgm = [len(samp.segments[i][0][k]) for k in range(len(samp.segments[i][0]))]
            print(p,i,len_sgm)'''
        for j in range(prototypes.num_var):
            segment = []
            for k in range(prototypes.num_sgm[p][j]):
                sum_sgm = torch.zeros_like(prototypes.segments[p][j][k])
                sum_info = torch.zeros_like(prototypes.information[p][j][:, k])
                for i in range(samp.num):
                    '''print(p,i,k)
                    len_sgm = [len(samp.segments[i][0][k]) for k in range(len(samp.segments[i][0]))]
                    print(len_sgm)'''
                    sum_sgm += samp.segments[i][j][k]
                    sum_info += samp.information[i][j][:, k]
                # 平均子序列
                avg_sgm = sum_sgm / samp.num
                avg_info = sum_info / samp.num  # tensor(5): angle, p_median, p_range, m_median, m_range
                _, _, t_range, v_median, v_range = avg_info
                sgm = adjust_segment(avg_sgm, prototypes.seq_len, v_median, v_range)
                segment.append(sgm)
                prototypes.segments[p][j][k] = sgm

                '''# 子序列信息
                index = torch.linspace(0, (len(sgm) - 1) / (prototypes.seq_len - 1), len(sgm), device=prototypes.series.device)
                info = sgm_info(index, sgm)
                prototypes.information[p][j][:, k] = info'''

            # 合并子序列
            series = MergeSegments(segment)
            prototypes.series[p][j] = series

    prototypes = SplitSamples(prototypes.series, prototypes.split)

    return clusters



def evaluate_prototypes_pma(clusters):
    d_shape, d_phase, d_magnit, num = 0, 0, 0, 0
    prototypes = clusters.prototypes
    samples = clusters.samples
    for p in range(prototypes.num):
        proto = prototypes[p]  # ASample
        samp = samples[p]  # ACluster
        num += samp.num

        distances = pma_sp(samp, proto)[1]
        for i in range(len(distances)):
            for j in range(len(distances[i])):
                for k in range(len(distances[i][j])):
                    _, d_pm, d_pr, d_mm, d_mr, d_s = distances[i][j][k]
                    d_shape += d_s
                    d_phase += (d_pm + d_pr)
                    d_magnit += (d_mm + d_mr)

    return d_shape, d_phase, d_magnit, num


def plot_clusters(clusters, centroids, init, labels, path, title='0'):
    colors = [(0.4, 0.7, 1), (0.6, 0.9, 0.6), 'orange']
    print(labels)
    for p in range(len(centroids)):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, series in enumerate(clusters[p]):
            ax.plot(series[0], linewidth=2, color=colors[labels[p][i]],
                    alpha=1)

        ax.plot(centroids[p][0], linewidth=4, color='black')
        ax.plot(init[p][0], linewidth=4, color='red')

        plt.tick_params(axis='both', which='major', labelsize=18)

        # 保存图像
        time = datetime.now().strftime('%d%H%M%S%f')
        plt.savefig(f'{path}/{title}_{p}.svg')
        plt.close()


def get_prototypes(samples, labels, args):
    """
    input:
    samples: Tensor(num, var, seq)
    labels: Tensor()

    output:
    prototype_set: [ACluster] * num_class
    label_set: [] * num_class
    sample_set: [[ACluster] * num_proto] * num_class
    """
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


    time = datetime.now().strftime('%d%H%M%S%f')
    path = f'./figure/{args.data_name}pma_{time}'
    os.makedirs(path, exist_ok=True)

    for c in range(len(label_set)):
        samp = torch.stack(sample_set[c], dim=0)
        samp_sp = get_split(samp)[0]
        samp = SplitSamples(samp, samp_sp)

        clusters = init_prototypes(samp, args.thres_mismatch)
        clusters = update_prototypes(clusters)

        prototype_set.append(clusters.prototypes)
        sample_set[c] = samp

        protos = clusters.prototypes.series
        samps = [clusters.samples[i].series for i in range(len(protos))]
        plot_clusters(samps, protos, path, f'{c}')


    return prototype_set, label_set, sample_set


def get_prototypes_all(samples, labels, args):
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

    proto_init = [sample_set[0][1], sample_set[1][2]]
    proto_init = torch.stack(proto_init, dim=0)
    proto = copy.deepcopy(proto_init)
    proto_sp = get_split(proto)[0]
    proto = SplitSamples(proto, proto_sp)

    samp_sp = get_split(samples)[0]
    samp = SplitSamples(samples, samp_sp)

    clusters_set, label_set = init_prototypes(samp, labels, proto)
    clusters_set = update_prototypes(clusters_set)

    protos = clusters_set.prototypes.series
    samps = [clusters_set.samples[i].series for i in range(len(protos))]

    time = datetime.now().strftime('%d%H%M%S%f')
    path = f'D:/备份/pma res/prototypes/{args.data_name}_pma{time}'
    os.makedirs(path, exist_ok=True)
    plot_clusters(samps, protos, proto_init, label_set, path)

    eva = evaluate_prototypes_pma(clusters_set)

    return clusters_set, label_set, eva




def evaluate_prototypes(clusters, centroids):
    d_shape, d_phase, d_magnit, num = 0, 0, 0, 0

    for p in range(len(centroids)):
        proto = centroids[p].unsqueeze(0)
        samp = clusters[p]
        distances = pma(samp, proto)[1]
        num += len(samp)
        for i in range(len(distances)):
            for j in range(len(distances[i])):
                for k in range(len(distances[i][j])):
                    _, d_pm, d_pr, d_mm, d_mr, d_s = distances[i][j][k]
                    d_shape += d_s
                    d_phase += (d_pm + d_pr)
                    d_magnit += (d_mm + d_mr)
        #print(d_shape, d_phase, d_magnit, num)
    return d_shape, d_phase, d_magnit, num




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


