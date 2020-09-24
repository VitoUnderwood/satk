import argparse
import utils
import os
from torch.utils.data import Dataset, DataLoader
from transformers.tokenization_bert import BertTokenizer

def get_args():
    parser = argparse.ArgumentParser(description='ASTE三元组抽取任务')

    # 训练超参数
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='通常选取2的幂')
    parser.add_argument('--num_train_epochs', type=int, default=5)
    parser.add_argument('--warmup_proportion', type=float, default=0.1, help='Bert专用慢热参数，控制学习率的变化')
    parser.add_argument('--weight_decay', type=float, default=0., help='待定')
    parser.add_argument('--max_seq_length', type=int, default=128)

    # 路径参数
    parser.add_argument('--train_file_name', type=str, required=True, help='训练集')
    parser.add_argument('--test_file_name', type=str, required=True, help='测试集')

    parser.add_argument('--output_model_dir', type=str, default='../../models/', help='aste模型保存路径')
    parser.add_argument('--bert_model_dir', type=str, default='../../models/', help='基础bert模型保存路径')

    # 其他参数
    parser.add_argument('--print_step', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mission', type=str, default='train', help='train or predict')
    parser.add_argument('--fp16', action='store_true', help='apex参数')

    args = parser.parse_args()
    return args

class ASTEDataset(Dataset):
      pass

class ASTE(nn.Module):
      @classmethod
      def train(cls, train_data_loader, test_data_loader):
          pass
def train_epoch(model, data_loader):
    model = model.train()

def eval_model(model, data_loader):
      pass
def train():
    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(EPOCHS):
      print(f'Epoch {epoch+1}/{EPOCHS}')
      print('-'*10)
      
      train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler, len(train_content))
      
      print(f'Train loss {train_loss}, accuracy {train_acc}')
      
      val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, device, len(df_val))
      
      print(f'Val loss {val_loss}, accuracy {val_acc}')
      print()
      
      history['train_acc'].append(train_acc)
      history['train_loss'].append(train_loss)
      history['vla_acc'].append(val_acc)
      history['vla_loss'].append(val_loss)
      
      if val_acc > best_accuracy:
          best_accuracy = val_acc
          torch.save(model.state_dict, 'bset_mode_state.bin')

def create_data_loader(content, sentiment, tokenizer, max_len, batch_size):
    ds = ReviewDataset(reviews=content,
                      targets=sentiment,
                      tokenizer=tokenizer,
                      max_len=max_len)
    return DataLoader(ds, batch_size=batch_size, num_workers=4)



if __name__ == '__main__':

  args = get_args()

  base_dir = '../../datasets/'

  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

  # print('load train data', base_dir+args.train_file_name)
  # train_texts, train_opinion_ploaritys = utils.load_data_from_xml(base_dir+args.train_file_name)

  # print('load test data', base_dir+args.test_file_name)
  # test_texts, test_opinion_ploaritys = utils.load_data_from_xml(base_dir+args.test_file_name)
  print('load train data', base_dir+args.train_file_name)
  utils.load_data_from_json(tokenizer, args.max_len, base_dir+args.train_file_name)

  train_dataloader = create_data_loader(train_texts, train_opinion_ploaritys)
  test_dataloader = create_data_loader(test_texts, test_opinion_ploaritys)

  model = ASTE(args)

  model.train(train_dataloader, test_dataloader)