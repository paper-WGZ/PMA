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

    # 将所有非数值（包括NaN和无效数据）替换为0
    data_set = data_set.apply(pd.to_numeric, errors='coerce').fillna(0)

    series_set = data_set.iloc[:, :-1].values
    label_set = data_set.iloc[:, -1].values

    # series 归一化
    if norm:
        label_set = torch.as_tensor(label_set, dtype=torch.float)
        series_set = torch.as_tensor(series_set, dtype=torch.float).unsqueeze(1)
        series_set = normalize(series_set, dim=-1, mode="min-max")

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


#DownLoadUCR('D:/dataset/ucr/')