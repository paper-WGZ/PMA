import matplotlib.pyplot as plt
import torch

def plot_pma(samp_series, proto_series, samp_split, proto_split,
             match_path, match_samp_split, match_proto_split,
             align_path, align_segments, save_path):


    fig, axs = plt.subplots(nrows=2, figsize=(8, 10))
    plt.subplots_adjust(hspace=0.1)

    # split & match
    axs[0].plot(samp_series + 1.5)
    axs[0].vlines(samp_split[1:-1], 1.5, 2.5, linestyles='dashed', colors='red')

    axs[0].plot(proto_series)
    axs[0].vlines(proto_split[1:-1], 0, 1, linestyles='dashed', colors='red')

    for (s, p) in match_path:
        p1 = (proto_split[p] + proto_split[p + 1]) / 2
        p2 = (samp_split[s] + samp_split[s + 1]) / 2
        axs[0].plot([p1, p2], [1.0, 1.5], 'g--')

    #axs[0].set_xticks([])  # 隐藏x轴刻度
    axs[0].set_yticks([])  # 隐藏y轴刻度

    # align
    axs[1].plot(samp_series + 1.5)
    axs[1].vlines(match_samp_split[1:-1], 1.5, 2.5, linestyles='dashed', colors='red')

    axs[1].plot(proto_series)
    axs[1].vlines(match_proto_split[1:-1], 0, 1, linestyles='dashed', colors='red')

    point_s, point_p = zip(*align_path)
    axs[1].plot([point_s[::20], point_p[::20]], [1.5, 1.0], 'g')

    for k in range(len(align_segments)):
        t = torch.arange(proto_split[k], proto_split[k + 1] + 1)
        axs[1].plot(t, align_segments[k] - 1.1, color='blue')
    axs[1].vlines(proto_split[1:-1], -1.1, -0.1, linestyles='dashed', colors='red')

    axs[1].set_xticks([])  # 隐藏x轴刻度
    axs[1].set_yticks([])  # 隐藏y轴刻度


    # 保存图像
    # plt.show()
    print(save_path)
    plt.savefig(save_path)
    plt.close()

