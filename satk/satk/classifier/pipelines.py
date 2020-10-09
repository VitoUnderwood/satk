import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import time


RANDOM_SEED = 20
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PRE_TRAINED_MODEL_NAME = 'hfl/rbt3'
# tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
# bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
class_names = ['中立', '正面', '反面']
MAX_LEN = 150




class SentimentClassifier(nn.Module):
    def __init__(self, n_class):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_class)
        
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids,
                                    attention_mask=attention_mask)
        output = self.drop(pooled_output)
        return self.out(output)
    

class Cls():
    def __init__(self):
        self.model = SentimentClassifier(len(class_names))
        self.model.load_state_dict(torch.load('/Users/vito/best_mode_state.bin', map_location=torch.device('cpu')))
        self.tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __call__(self, review_text):
        encoded_review = self.tokenizer.encode_plus(
                review_text,
                truncation=True,
                max_length=MAX_LEN,
                add_special_tokens=True,
                return_token_type_ids=False,
                padding=True,
                return_attention_mask=True,
                return_tensors='pt',
            )

        input_ids = encoded_review['input_ids'].to(self.device)
        attention_mask = encoded_review['attention_mask'].to(self.device)
        output = self.model(input_ids, attention_mask)
        _, prediction = torch.max(output, dim=1)

        print(f'Review text: {review_text}')
        print(f'Sentiment  : {class_names[prediction]}')


# model = SentimentClassifier(len(class_names))

# model.load_state_dict(torch.load('best_mode_state.bin', map_location=torch.device('cpu')))
# model = model.to(device)

# review_text = "房间的卫生间较差，不尽人意。总的感觉不是很好。但是个人用品的卫生还是很规范的，整洁而且包装的很完整。"

# encoded_review = tokenizer.encode_plus(
#     review_text,
#     truncation=True,
#     max_length=MAX_LEN,
#     add_special_tokens=True,
#     return_token_type_ids=False,
#     padding=True,
#     return_attention_mask=True,
#     return_tensors='pt',
# )

# input_ids = encoded_review['input_ids'].to(device)
# attention_mask = encoded_review['attention_mask'].to(device)

# start = time.time()
# output = model(input_ids, attention_mask)
# _, prediction = torch.max(output, dim=1)
# print(time.time() - start)
# print(f'Review text: {review_text}')
# print(f'Sentiment  : {class_names[prediction]}')