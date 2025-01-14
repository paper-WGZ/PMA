import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--root', default='D:/PMAP/ucr1/')
    parser.add_argument('--data_name', type=str, default='Beef')


    # split
    parser.add_argument('--is_smooth', type=bool, default=False)
    parser.add_argument('--local_ratio', type=float, default=0.05)
    parser.add_argument('--split_window', type=int, default=10)
    parser.add_argument('--split_stride', type=int, default=5)
    parser.add_argument('--split_vertical', type=float, default=0.0 )
    parser.add_argument('--split_angle', type=float, default=0.1)

    # match
    parser.add_argument('--match_angle', type=float, default=0.8)
    parser.add_argument('--match_magnit_scale', type=float, default=1)
    parser.add_argument('--match_phase_scale', type=float, default=1)
    parser.add_argument('--match_magnit_shift', type=float, default=1)
    parser.add_argument('--match_phase_shift', type=float, default=10)

    # prototype
    parser.add_argument('--thres_unmatch', type=float, default=0.1)

    # visualization
    parser.add_argument('--is_plot', type=bool, default=False)

    # distance
    parser.add_argument('--seed', type=int, default=10)
    args = parser.parse_args()
    return args

args = get_args()


# 包的公共接口
__all__ = ["split", "match", "align", "dPMA"]

