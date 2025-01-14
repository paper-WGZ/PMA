import torch

def sgm_slope(index:torch.Tensor, values:torch.Tensor):
    # 计算斜率 slope
    n = len(index)
    sum_x = torch.sum(index)
    sum_y = torch.sum(values)
    sum_xy = torch.sum(index * values)
    sum_xx = torch.sum(index * index)
    slope = (n * sum_xy - sum_x * sum_y) \
            / (n * sum_xx - sum_x ** 2 + 10e-8)
    return slope

def angle_distance(index_a, sgm_a, index_b, sgm_b):
    return torch.abs(torch.atan(sgm_slope(index_a, sgm_a))
                   - torch.atan(sgm_slope(index_b, sgm_b)))

def vertical_distance(point_index, point_value, series_index, series_value):
    k = (series_value[-1] - series_value[0]) \
        / (series_index[-1] - series_index[0])
    d = torch.abs(k * (point_index - series_index[0])
                  - point_value + series_value[0]) / torch.sqrt(k ** 2 + 1)
    return d

def max_vertical(series, l, r, index_scale=1):
    if r - l <= 2: return l, 0
    index = torch.linspace(0, index_scale, len(series), device=series.device)
    d = vertical_distance(index[l: r], series[l: r], index[l: r], series[l: r])
    i = torch.argmax(d, dim=0)
    return int(i + l), d[i]


def argmax_angel(t, y, l):
    """
    t: time tensor, y: value tensor, l: left index
    """
    d = angle_similar(t, y)
    i = torch.argmax(d, dim=-1)
    return i + l, d[i]

