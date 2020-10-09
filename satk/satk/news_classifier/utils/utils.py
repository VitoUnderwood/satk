# -*- coding:utf-8 -*-
import torch
import numpy as np


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed_all(seed)  # 为当前GPU设置随机种子
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    torch.backends.cudnn.benchmark = False


def time_it(task, start, end):
    """简单的计时工具"""
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    print(f'{task} cost {int(h)}:{int(m):02d}:{int(s):02d}')
    # t = str(datetime.timedelta(seconds=end - start))
    # print(f'{task} cost {t}')
