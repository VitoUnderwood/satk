import torch
import os
import csv
import json
import numbers

import random
import numpy as np


"""
工具函数
1. mkdir_if_notexist(dir_)
2. get_device(gpu_ids)
3. _load_csv(file_name, skip_fisrt)
4. _load_json(file_name)
5. _save_json(data, file_name)
6. AvgVar()
7. Vn()
8. F1_Measure()
9. f1_measure(tp, fp, fn)
10. set_seed(args)
11. KFold(n_splits)
"""


def mkdir_if_notexist(dir_):
    dirname, filename = os.path.split(dir_)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_device(gpu_ids):
    if gpu_ids:
        device_name = 'cuda:' + str(gpu_ids[0])
        n_gpu = torch.cuda.device_count()
        print('device is cuda, # cuda is: %d' % n_gpu)
    else:
        device_name = 'cpu'
        print('device is cpu')
    device = torch.device(device_name)
    return device


def _load_csv(file_name, skip_first=True):
    with open(file_name, mode='r', encoding='utf-8') as f:
        if skip_first:  # 跳过表头
            f.__next__()
        for line in csv.reader(f):
            yield line


def _save_csv(data, file_name):
    with open(file_name, mode='w', encoding='utf-8', newline='\n') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(data)


def _save_txt(lst, file_name, mode='w'):
    mkdir_if_notexist(file_name)
    with open(file_name, encoding='utf-8', mode=mode) as f:
        for line in lst:
            f.write(line+'\n')


# def _save_csv(data, file_name):
#     with open(file_name, mode='w', encoding='utf-8', newline='') as f:
#         writer = csv.writer(f)
#         for row in data:
#             writer.writerow(row)


def _load_json(file_name):
    with open(file_name, encoding='utf-8', mode='r') as f:
        return json.load(f)


def _save_json(data, file_name):
    mkdir_if_notexist(file_name)
    with open(file_name, encoding='utf-8', mode='w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


class AvgVar:
    """
    维护一个累加求平均的变量
    """
    def __init__(self):
        self.var = 0
        self.steps = 0

    def inc(self, v, step=1):
        self.var += v
        self.steps += step

    def avg(self):
        return self.var / self.steps if self.steps else 0


class Vn:
    """
    维护n个累加求平均的变量
    """
    def __init__(self, n):
        self.n = n
        self.vs = [AvgVar() for i in range(n)]

    def __getitem__(self, key):
        return self.vs[key]

    def init(self):
        self.vs = [AvgVar() for i in range(self.n)]

    def inc(self, vs):
        for v, _v in zip(self.vs, vs):
            v.inc(_v)

    def avg(self):
        return [v.avg() for v in self.vs]

    def list(self):
        return [v.var for v in self.vs]


class F1_Measure:
    """
    ----------------
            真实
            P   N
    预   P  tp  fp
    测   N  fn  tn
    ----------------

    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * prec * recall / (prec + recall)
       = 2 * tp / (tp + fp) * tp / (tp + fn) / [ tp / (tp + fp) + tp / (tp + fn)]
       = 2 * tp / [tp + fp + tp + fn]
    """
    def __init__(self):
        self.tp = 0
        self.tp_fp_tp_fn = 0

    def inc(self, tp, tp_fp, tp_fn):
        # tp_fp: 预测值为正的
        # tp_fn: 真实值为正的
        self.tp += tp
        self.tp_fp_tp_fn += tp_fp + tp_fn

    def f1(self):
        f1 = 2 * self.tp / self.tp_fp_tp_fn if self.tp else 0
        return f1


def f1_measure(tp, fp, fn):
    return 2 * tp / (tp + fp + tp + fn) if tp else 0


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if len(args.gpu_ids) > 0:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


class KFold:
    """
    参考自scikit-learn
    """
    def __init__(self, n_splits, random_state):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, n_samples):
        indices = np.arange(n_samples)
        for test_index in self._iter_test_masks(n_samples):
            train_index = indices[np.logical_not(test_index)]
            test_index = indices[test_index]
            yield train_index, test_index

    def _iter_test_masks(self, n_samples):
        for test_index in self._iter_test_indices(n_samples):
            test_mask = np.zeros(n_samples, dtype=np.bool)
            test_mask[test_index] = True
            yield test_mask

    def _iter_test_indices(self, n_samples):
        indices = np.arange(n_samples)
        check_random_state(self.random_state).shuffle(indices)

        n_splits = self.n_splits
        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=np.int)
        fold_sizes[:n_samples % n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current,  current + fold_size
            yield indices[start: stop]
            current = stop
