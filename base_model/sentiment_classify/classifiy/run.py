#!/usr/bin/env python
# coding: utf-8


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




RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


with open('data/train_sentiment.txt') as f:
    lines  = f.readlines()
    content = []
    sentiment = []
    for line in lines:
        line = line.split('\t')
        content.append(line[1])
        sentiment.append(int(line[2][0]))

print('load data finish')


PRE_TRAINED_MODEL_NAME = 'hfl/rbt3'




tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

print('load model finish')

class ReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        # numpy data
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return (len(self.reviews))
    
    def __getitem__(self, index):
        review = str(self.reviews[index])
        target = self.targets[index]
        
        encoding = self.tokenizer.encode_plus(review,
                                             add_special_tokens=True,
                                             max_length=self.max_len,
                                             return_token_type_ids=False,
                                             pad_to_max_length=True,
                                             return_attention_mask=True,
                                             return_tensors='pt',
                                             truncation=True)
        return{
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }



len(content)


train_content, test_content,train_sentiment, test_sentiment = train_test_split(content, sentiment,test_size=0.1, random_state=RANDOM_SEED)
test_content, val_content,test_sentiment, val_sentiment = train_test_split(test_content, test_sentiment,test_size=0.1, random_state=RANDOM_SEED)



MAX_LEN = 150




def create_data_loader(content, sentiment, tokenizer, max_len, batch_size):
    ds = ReviewDataset(reviews=content,
                      targets=sentiment,
                      tokenizer=tokenizer,
                      max_len=max_len)
    return DataLoader(ds, batch_size=batch_size, num_workers=4)

BATCH_SIZE = 16

train_data_loader = create_data_loader(train_content, train_sentiment, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(val_content, val_sentiment, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(test_content, test_sentiment, tokenizer, MAX_LEN, BATCH_SIZE)



data = next(iter(train_data_loader))
data.keys()



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



model = SentimentClassifier(3)
model = model.to(device)



EPOCHS = 10

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_setps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_setps)

loss_fn = nn.CrossEntropyLoss().to(device)




def train_epoch(model, data_loader, loss_fn, optimizer, device, schduler, n_examples):
    model = model.train()
    
    losses = []
    correct_predictions = 0
    
    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        targets = d['targets'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        
        correct_predictions += torch.sum(preds==targets)
        losses.append(loss.item())
        
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
    return correct_predictions.double() / n_examples, np.mean(losses)




def eval_model(model, data_loader, loss_fn, device, n_examples):
    mode = model.eval()
    
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            targets = d['targets'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _,  preds = torch.max(outputs, dim=1)
            
            loss = loss_fn(outputs, targets)
            
            correct_predictions += torch.sum(preds==targets)
            losses.append(loss.item())
            
    return correct_predictions.double() / n_examples, np.mean(losses)                

history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):
    print(f'Epoch {epoch+1}/{EPOCHS}')
    print('-'*10)
    
    train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler, len(train_content))
    
    print(f'Train loss {train_loss}, accuracy {train_acc}')
    
    val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, device, len(val_content))
    
    print(f'Val loss {val_loss}, accuracy {val_acc}')
    print()
    
    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['vla_acc'].append(val_acc)
    history['vla_loss'].append(val_loss)
    
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        torch.save(model.state_dict, 'bset_mode_state.bin')




