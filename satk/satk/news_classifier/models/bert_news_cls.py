# coding: UTF-8
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class BertConfig:
    """配置参数"""

    def __init__(self, dataset):
        self.model_name = 'bert_news_cls'
        self.train_path = 'data/' + dataset + '/train.txt'
        self.dev_path = 'data/' + dataset + '/dev.txt'
        self.test_path = 'data/' + dataset + '/test.txt'
        self.class_list = [x.strip() for x in open('/Users/vito/Desktop/satk/satk/satk/news_classifier/data/' + dataset + '/class.txt').readlines()]
        self.save_path = 'pretrained/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_classes = len(self.class_list)
        self.num_epochs = 3
        self.batch_size = 128
        # 每句话处理成的长度(短填长切)
        self.max_length = 32
        self.learning_rate = 5e-5
        self.bert_path = 'pretrained/rbt3'
        self.tokenizer = BertTokenizer.from_pretrained('hfl/rbt3', cache_dir=self.bert_path)
        self.hidden_size = 768  # 和bert最后一层匹配768


class BertClsModel(nn.Module):
    def __init__(self, config):
        super(BertClsModel, self).__init__()
        self.bert = BertModel.from_pretrained('hfl/rbt3', cache_dir=config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True  # False的话相当于吧bert的参数冻结了
        # for name, parameters in self.bert.named_parameters():
        #     print(name, ':', parameters.size())
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, input_ids=None, attention_mask=None):
        _, pooled = self.bert(input_ids, attention_mask)
        output = self.drop(pooled)
        return self.fc(output)
