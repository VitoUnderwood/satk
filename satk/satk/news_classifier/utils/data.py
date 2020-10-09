# -*- coding:utf-8 -*-
from torch.utils.data import DataLoader, Dataset
from ..models.bert_news_cls import BertConfig
from tqdm import tqdm
import torch


class NewsDataset(Dataset):

    def __init__(self, data_path, tokenizer, max_length):
        # load data
        self.contents = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(data_path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                line = line.strip()
                if not line:
                    continue
                content, label = line.split('\t')
                self.contents.append(content)
                self.labels.append(int(label))

    def __getitem__(self, index):
        # tokenize input
        content = self.contents[index]
        label = self.labels[index]

        encoded_dict = self.tokenizer.encode_plus(content,
                                                  add_special_tokens=True,
                                                  max_length=self.max_length,
                                                  truncation=True,
                                                  padding='max_length',
                                                  return_tensors='pt',
                                                  return_attention_mask=True,
                                                  )
        return {'input_ids': encoded_dict['input_ids'].flatten(),
                'attention_mask': encoded_dict['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.long)
                }

    def __len__(self):
        return len(self.contents)


def build_dataloader(config: BertConfig):
    train_dataloader = DataLoader(dataset=NewsDataset(config.train_path, config.tokenizer, config.max_length),
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=6)  # 在创建dataloader的时候并无作用，使用的dataloader的时候会加速
    dev_dataloader = DataLoader(dataset=NewsDataset(config.dev_path, config.tokenizer, config.max_length),
                                batch_size=config.batch_size,
                                shuffle=True,
                                num_workers=6)
    test_dataloader = DataLoader(dataset=NewsDataset(config.test_path, config.tokenizer, config.max_length),
                                 batch_size=config.batch_size,
                                 shuffle=True,
                                 num_workers=6)
    return train_dataloader, dev_dataloader, test_dataloader
