from transformers.tokenization_bert import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_bert import BertPreTrainedModel
from transformers.modeling_bert import BertModel
import math

from utils.span import Span
span = Span(n_best_size=13, max_span_length=20, top_K=13)

class Feature:
    def __init__(self, idx, input_ids, input_mask, segment_ids):
        self.idx = idx
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

    @classmethod
    def make_single(cls, idx, tokens, tokenizer, max_seq_length):
        tokens = ['[CLS]'] + tokens + ['[SEP]']

        assert len(tokens) <= max_seq_length
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        return cls(idx, input_ids, input_mask, segment_ids)
    
    @classmethod
    def make_double(cls, idx, tokens1, tokens2, tokenizer, max_seq_length):
        tokens = ['[CLS]'] + tokens1 + ['[SEP]'] + tokens2 + ['[SEP]']

        assert len(tokens) <= max_seq_length
    
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * (len(tokens1) + 2) + [1] * (len(tokens2) + 1)

        padding = [0] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        return cls(idx, input_ids, input_mask, segment_ids)

def convert_feature_to_tensor(features):
    all_idx = torch.tensor([f.idx for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    return all_idx, all_input_ids, all_input_mask, all_segment_ids


class SpanModel(nn.Module):
    def __init__(self, hidden_size=768, n_span=4):
        super().__init__()

        self.n_span_layer = nn.Sequential(
            nn.Linear(hidden_size+1, hidden_size+1),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size+1, n_span) 
        )
        self.qa_outputs = nn.Sequential(
            nn.Linear(hidden_size, 2)
        )

    def span_predict(self, sequence_output, attention_mask):
        """
        sequence_ouptput: [B, L, H]
        attention_mask: [B, L]
        start_positions, end_positions: [B, L]
        """
        logits = self.qa_outputs(sequence_output) - (1-attention_mask.unsqueeze(-1)) * 10000.        
        start_logits, end_logits = logits.split(1, dim=-1)
        # [B, L]
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        return start_logits, end_logits

    def n_predict(self, sequence_output, mask, start_logits, end_logits):
        """
        sequence_output: [B, L, H]
        mask: [B, L]
        n_true: [B,]
        start_logits, end_logits: [B, L]
        """
        h_target1 = torch.max(sequence_output * mask.unsqueeze(-1), dim=1)[0]
        h_target2 = torch.sum(F.relu(start_logits * mask), dim=-1, keepdim=True)
        h_target3 = torch.sum(F.relu(end_logits * mask), dim=-1, keepdim=True)

        h_target = torch.cat((h_target1, h_target2 + h_target3), dim=-1)

        n_pred = self.n_span_layer(h_target)
        n_pred = n_pred.argmax(dim=-1) + 1
        return n_pred


class SpanExtraction(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.span_model = SpanModel(hidden_size=768, n_span=4)

        self.init_weights()
        
    def predict(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]

        seq_mask = attention_mask - token_type_ids
        start_logits, end_logits = self.span_model.span_predict(
            sequence_output, seq_mask)
        
        n_pred = self.span_model.n_predict(
            sequence_output, seq_mask, start_logits, end_logits)

        return start_logits, end_logits, n_pred


class TOWE:
    """
    1. self.init()
    2. self.train(...)
    3. cls.load(...)
    """
    def __init__(self, ModelClass):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print('init_model')

        self.model = ModelClass.from_pretrained('/Users/vito/models')

    def output_result(self, sentence, target):
        tokens = tokenizer.tokenize(sentence)
        target_tokens = tokenizer.tokenize(target)

        feature = Feature.make_double(1, tokens, target_tokens, tokenizer, len(tokens) + len(target_tokens) + 3)
        features = [feature]
        _, input_ids, attention_mask, token_type_ids = convert_feature_to_tensor(features)

        start_logits, end_logits, n_pred = self.model.predict(
            input_ids.to(self.device), attention_mask.to(self.device), token_type_ids.to(self.device))
        
        y_pred = span.parse(start_logits[0].detach().cpu().numpy(), 
                            end_logits[0].detach().cpu().numpy(),
                            pn=n_pred[0].detach().cpu().numpy())
        opinion_words = []
        print(f'句子:{tokens}中评价对象{target_tokens}对应的观点评价词：')
        for s, e in y_pred:
            print(s-1,e-1, tokens[s-1:e-1])

class Absa():
    def __init__(self):
        
        self.srt = TOWE(SpanExtraction)

    def __call__(self, sentence, target):
        self.srt.output_result(sentence, target)

