# -*- coding:utf-8 -*-
import time
from ..models.bert_news_cls import BertConfig, BertClsModel
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm


def train(config: BertConfig, model: BertClsModel, train_dataloader, dev_dataloader, test_dataloader):
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_dataloader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss().to(config.device)
    model.train()
    losses = []
    cnt = 0
    correct_predictions = 0
    n_examples = 0
    dev_best_loss = float('inf')
    for epoch in range(config.num_epochs):
        print(f'{"=" * 20}Epoch {epoch + 1}/{config.num_epochs}{"=" * 20}')
        for i, batch in tqdm(enumerate(train_dataloader)):
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['label'].to(config.device)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, labels)  # 计算loss
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数
            scheduler.step()  # 调整学习率
            optimizer.zero_grad()  # 清空梯度

            losses.append(loss.item())
            correct_predictions += torch.sum(preds == labels)
            n_examples += config.batch_size
            acc = correct_predictions.double() / n_examples
            cnt += 1
            if cnt % 100 == 0:
                dev_acc, dev_loss = evaluate(config, model, dev_dataloader)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    # torch.save(model, config.save_path)
                    torch.save(model.state_dict(), config.save_path)
                print(f'Iter: {cnt}, Train Loss: {loss.item():.2f}, Train Acc: {acc:.2f}, Val loss: {dev_loss:.2f}, Val Acc: {dev_acc:.2f}')


def evaluate(config: BertConfig, model: BertClsModel, dev_dataloader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss().to(config.device)
    losses = []
    correct_predictions = 0
    n_examples = 0
    with torch.no_grad():
        for batch in dev_dataloader:
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['label'].to(config.device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            _, preds = torch.max(outputs, dim=1)
            losses.append(loss.item())
            correct_predictions += torch.sum(preds == labels)
            n_examples += config.batch_size

    dev_acc = correct_predictions.double() / n_examples
    dev_loss = np.mean(losses)
    model.train()
    return dev_acc, dev_loss
