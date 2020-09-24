import re
import xml.etree.ElementTree as ET
import os
import json
from tqdm import tqdm

class Tagset:
    """
    self.make_bio_idx(entities, tokens, max_seq_length)
    """
    def __init__(self, tags):
        self.tag_to_idx = self._to_idx(tags)
        self.idx_to_tag = self._to_tag(tags)

    def _to_tag(self, tags):
        return dict((k, v) for k, v in enumerate(tags))

    def _to_idx(self, tags):
        return dict((k, v) for v, k in enumerate(tags))

    def __getitem__(self, tag):
        return self.tag_to_idx[tag]

    def size(self):
        return len(self.tag_to_idx)

    def is_entity(self, idx):
        if idx not in (
            self.tag_to_idx['PAD'],
            self.tag_to_idx['SOS'],
            self.tag_to_idx['O'],
        ):
            return True
        else:
            return False

    def make_bio_idx(self, entities, tokens, text, max_seq_length):
        bio_seq = from_entities_to_bio(entities, tokens, text)
        bio_seq = ['SOS'] + bio_seq
        bio_seq += ['PAD'] * (max_seq_length - len(bio_seq))
        return self.from_tag_seq_to_idx_seq(bio_seq)

    def from_tag_seq_to_idx_seq(self, tag_seq):
        return [self.tag_to_idx[tag] for tag in tag_seq]

    def from_idx_seq_to_tag_seq(self, idx_seq):
        return [self.idx_to_tag[idx] for idx in idx_seq if self.idx_to_tag[idx] != 'PAD']

    def parse_idx(self, idx_seq):
        bio_seq = self.from_idx_seq_to_tag_seq(idx_seq)
        return from_bio_to_entites(bio_seq), bio_seq

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text

def load_data_from_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    texts = []
    opinion_ploaritys = []
    for node in tqdm(root):
        text = node.find('text')
        

        aspectTerms = node.find('aspectTerms')
        opinion_ploarity = []
        if aspectTerms is not None:
            for a in aspectTerms:
                dic = {}
                for k, v in a.attrib.items():
                    dic[preprocess_text(k).strip()] = preprocess_text(v).strip() 
                opinion_ploarity.append(dic)

        for o in opinion_ploarity:
            o.pop('from')
            o.pop('to') 
        
        if opinion_ploarity:
            texts.append(preprocess_text(text.text).strip())
            opinion_ploaritys.append(opinion_ploarity)

    return texts, opinion_ploaritys

entity_tagset = Tagset(['PAD', 'B-target', 'I-target', 'B-opinion', 'I-opinion', 'O', 'SOS', 'opinion'])
target_tagset = Tagset(['PAD', 'B-target', 'I-target', 'O', 'SOS'])
target_polarity_tagset = Tagset(['PAD', 'B-POS', 'I-POS', 'B-NEG', 'I-NEG', 'B-NEU', 'I-NEU', 'O', 'SOS'])
opinion_tagset = Tagset(['PAD', 'B-opinion', 'I-opinion', 'O', 'SOS'])
opinion_polarity_tagset = Tagset(['PAD', 'B-POS', 'I-POS', 'B-NEG', 'I-NEG', 'B-NEU', 'I-NEU', 'O', 'SOS'])

def load_data_from_json(tokenizer, max_seq_length, file_path):
    with open(file_path, encoding='utf-8', mode='r') as f:
        json_objs = json.load(f)
        for json_obj in json_objs:
            sentence = json_obj['Sentence'],
            aste=json_obj['entities']
            # tokenize
            tokens = tokenizer.tokenize(sentence)
            # TS, TE, TP, TN, OS, OE, OP, ON = self.make_target_opinion_span(tokens, max_seq_length)
            target = []
            opinion = []
            for tri in aste:
                polarity = tri['polarity']
                ts, te, tw = tri['target']
                target.append((polarity, ts+1, te+1, tw))
                opinion.extend([(polarity, os+1, oe+1, ow)
                                for (os, oe, ow) in tri['opinions']]) # 同一target可能有多种opinion，但只有一种极性

            TS, TE, TP, TN = self._make_span(target, tokens, max_seq_length, target_polarity_tagset)
            OS, OE, OP, ON = self._make_span(opinion, tokens, max_seq_length, opinion_polarity_tagset)

            return TS, TE, TP, TN, OS, OE, OP, ON
            TOS, TOE, TON = self.make_opinion_span_matrix(tokens, max_seq_length)

            feature = Feature.make_single(self.idx, tokens, tokenizer, max_seq_length)
    return feature, TS, TE, TP, TN, OS, OE, OP, ON, TOS, TOE, TON



    return 