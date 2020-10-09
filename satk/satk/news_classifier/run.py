import argparse
# from importlib import import_module
from .utils.utils import set_seed, time_it
from .utils.data import build_dataloader
from .utils.trainer import train
import time


def get_args():
    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
    parser.add_argument('--task', type=str, required=True, help='choose a task: train, test')
    parser.add_argument('--dataset', type=str, default='THUCNews', help='choose dataset which include train, dev, test')
    parser.add_argument('--seed', type=int, default=2020, help='set seed for np, torch, cpu, gpu, cudnn')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    set_seed(args.seed)
    if args.model == 'bert':
        start = time.time()
        from .models import bert_news_cls as news_cls
        config = news_cls.BertConfig(args.dataset)
        end = time.time()
        time_it('加载模型配置', start, end)

        start = time.time()
        model = news_cls.BertClsModel(config)
        end = time.time()
        time_it('初始化分类模型', start, end)

        start = time.time()
        train_dataloader, dev_dataloader, test_dataloader = build_dataloader(config)
        end = time.time()
        time_it('数据初始化', start, end)

        start = time.time()
        train(config, model, train_dataloader, dev_dataloader, test_dataloader)
        end = time.time()
        time_it('训练模型', start, end)


if __name__ == '__main__':
    main()
