import os
import numpy as np
import pandas as pd
from pyts import datasets
from tqdm import tqdm
import torch
from sklearn.preprocessing import OneHotEncoder




def normalize(x, dim=0, mode = 'min-max'):
    """
    normalize x in dim with mode
    :param dim: normalization in dim;
    :param mode: 'min-max'--MinMaxScaler; 'std'--StandardScaler
    """
    if torch.is_tensor(x) == False:
        x = torch.tensor(x, dtype=torch.float)
    if mode == 'min-max':
        min_, _ = torch.min(x, dim, keepdim=True)
        max_, _ = torch.max(x, dim, keepdim=True)
        x = (x - min_) / (max_ - min_)
    elif mode == 'std':
        mean = torch.mean(x, dim, keepdim=True)
        std = torch.std(x, dim, keepdim=True)
        x = (x - mean) / std
    else:
        print('no such mode')
    x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)
    x = torch.where(torch.isinf(x), torch.full_like(x, 1), x)
    return x

def DownLoadUCR(root):
    if not os.path.exists(root): os.makedirs(root)
    for name in tqdm(datasets.ucr_dataset_list()):
        folder = root + '/' + name
        if not os.path.exists(folder): os.makedirs(folder)
        (data_train, data_test, target_train, target_test) \
            = datasets.fetch_ucr_dataset(name, return_X_y=True)
        # 处理缺失值
        data_train, data_test = np.nan_to_num(data_train, nan=0, copy=False), \
                                np.nan_to_num(data_test, nan=0, copy=False)
        # 归一化
        data_train, data_test = normalize(data_train, dim=1, mode='min-max'), \
                                normalize(data_test, dim=1, mode='min-max')
        target_train, target_test = np.expand_dims(target_train, axis=1), \
                                    np.expand_dims(target_test, axis=1)
        train = np.concatenate([data_train, target_train], axis=1)
        test = np.concatenate([data_test, target_test], axis=1)
        # print(data_train.shape, target_train.shape, train.shape, data_test.shape, target_test.shape, test.shape)
        pd.DataFrame(train).to_csv(folder + '/' + name + '_TRAIN.csv')
        pd.DataFrame(test).to_csv(folder + '/' + name + '_TEST.csv')

def get_dataset(csv_file, norm=True):
    data_set = pd.read_csv(csv_file, index_col=0)
    series_set = data_set.iloc[:, :-1].values
    label_set = data_set.iloc[:, -1].values

    # series 归一化
    if norm:
        series_set = torch.as_tensor(series_set, dtype=torch.float)
        series_set = normalize(series_set, dim=1, mode="std")

    # label 转换为 one-hot
    encoder = OneHotEncoder(sparse=False)
    label_set = encoder.fit_transform(label_set.reshape(-1, 1))
    label_set = torch.as_tensor(label_set, dtype=torch.float)
    return series_set, label_set


def GroupSet(csv_file, norm=True):
    data_set = pd.read_csv(csv_file, index_col=0).values.tolist()
    series_set, label_set, num_list = [], [], []
    for data in data_set:
        series = data[:-1]
        label = str(data[-1])
        if label not in label_set:
            label_set.append(label)
            series_set.append([series])
        else:
            index = label_set.index(label)
            series_set[index].append(series)
    for i in range(len(series_set)):
        num_list.append(len(series_set[i]))
        series_set[i] = torch.as_tensor(series_set[i], dtype=torch.float)
        # 归一化
        if norm:
            series_set[i] = normalize(series_set[i], dim=1, mode="min-max")
    return series_set, label_set, num_list

#DownLoadUCR('D:/dataset/ucr3')

def loess_smoothing(x, local_ratio=0.1):
    """
    :param x: (batch_size, num_variant, seq_len)
    :param local_ratio: 滑动窗口长度 / 序列长度
    :return:
    """

    device = x.device

    batch_size, num_variant, seq_len = x.shape
    window = int(np.ceil(local_ratio * seq_len))

    t = torch.arange(seq_len, dtype=torch.float, device=device)
    x_smooth = torch.zeros_like(x)

    for i in tqdm(range(seq_len), desc="trend-seasonal decomposition"):
        distances = torch.abs(t - t[i])
        weights = torch.exp(-distances / window).unsqueeze(0).unsqueeze(0)
        W = torch.diag_embed(weights.repeat(batch_size, num_variant, 1))

        A = torch.stack([torch.ones(seq_len, device=device), t], dim=1).unsqueeze(0).repeat(batch_size * num_variant, 1,
                                                                                            1)
        x_reshaped = x.view(-1, seq_len).unsqueeze(-1)

        WA = W.view(-1, seq_len, seq_len) @ A
        Wx = W.view(-1, seq_len, seq_len) @ x_reshaped

        beta = torch.linalg.lstsq(WA, Wx).solution  # Retrieve only the solution
        beta = beta.view(batch_size, num_variant, 2, 1)
        x_smooth[:, :, i] = beta[:, :, 0, 0] + beta[:, :, 1, 0] * t[i]

    return x_smooth


#DownLoadUCR('D:/dataset/ucr/')