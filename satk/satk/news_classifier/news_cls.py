import torch
from transformers import BertTokenizer
import torch.nn.functional as F
from .utils.utils import time_it
from .models.bert_news_cls import BertConfig, BertClsModel
import time
import json
# import warnings  # pytorch 版本差异带来的waring，忽略即可
# warnings.filterwarnings("ignore")


class Pipeline:
    def __init__(self):
        self.config = BertConfig('THUCNews')
        self.model = BertClsModel(self.config)
        self.model.load_state_dict(torch.load('/Users/vito/Desktop/satk/satk/satk/news_classifier/pretrained/bert_news_cls.ckpt', map_location='cpu'))
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained('hfl/rbt3', cache_dir='/Users/vito/Desktop/satk/satk/satk/news_classifier/pretrained/rbt3')
        with open('/Users/vito/Desktop/satk/satk/satk/news_classifier/data/THUCNews/class.txt', 'r') as f:
            self.class_name = [name.strip('\n') for name in f]

    def __call__(self, text):
        encoded_dict = self.tokenizer.encode_plus(text,
                                                  add_special_tokens=True,
                                                  max_length=self.config.max_length,
                                                  truncation=True,
                                                  padding='max_length',
                                                  return_tensors='pt',
                                                  return_attention_mask=True,
                                                  )
        with torch.no_grad():
            input_ids = encoded_dict["input_ids"]
            attention_mask = encoded_dict["attention_mask"]

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            probs = F.softmax(outputs, dim=1)

            data = []
            for i in range(len(probs[0])):
                data.append({'value': probs[0][i].item(), 'name': self.class_name[i]})

            message = {'data': data, '最大概率': probs[0][preds].item(),
                       '类别': self.class_name[preds.item()]}
            message = json.dumps(message, ensure_ascii=False)  # 解决中文显示乱码
            print(message)
            return message

if __name__ == '__main__':
    start = time.time()
    p = Pipeline()
    time_it('加载模型', start, time.time())
    msg = p('广东一工地发生坍塌致7人死亡')
