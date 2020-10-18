import pandas as pd
import jieba_fast as jieba
import re
from tqdm import tqdm
from gensim import corpora, models
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt


class Lda:
    def __init__(self):
        print("加载模型")
        self.loaded_dct = corpora.Dictionary.load_from_text("/Users/vito/models/lda/dict.txt")
        self.lda = models.LdaModel.load('/Users/vito/models/lda/model6')
        self.topics = self.lda.show_topics()
    
    def __call__(self, news):
        # 处理成正确的输入格式       
        news = re.sub(r'[^\u4e00-\u9fa5]+','',news).strip()
        news = jieba.cut(news)
        tokens = [i for i in news]
        # 新闻ID化    
        corpus_test = self.loaded_dct.doc2bow(tokens)
        # 得到每条新闻的主题分布
        topics_test = self.lda.get_document_topics(corpus_test)  
        print("主题分布：")
        print(sorted(topics_test,key=lambda x:x[1], reverse=True))
        return sorted(topics_test,key=lambda x:x[1], reverse=True)
        
    def show_topics(self):
        for t in self.topics:
            print(t)
    
    def get_topic_dis(self, topic_id):
        return self.lda.show_topic(topic_id, topn=20)

df = pd.read_csv('../../../datasets/simplifyweibo_4_moods.csv')
reviews = df.review.to_list()

stopwords = [line.strip() for line in open('../../../datasets/stopwords.txt',encoding='UTF-8').readlines()]


def data_process(texts, stopwords):
    clean_texts = []
    for line in tqdm(texts):
        line = re.sub(r'[^\u4e00-\u9fa5]+','',str(line)).strip()
        line = jieba.cut(line)
        line = [word for word in line if word not in stopwords]
        if line is not None:
            clean_texts.append(line)
    return clean_texts
clean_texts = data_process(reviews, stopwords)

id2word = corpora.Dictionary(clean_texts)
corpus = [id2word.doc2bow(line) for line in clean_texts]

coherence_values = []
model_list = []
for num_topics in tqdm(range(2, 40, 2)):
    model = models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics)
    model_list.append(model)
    coherencemodel = CoherenceModel(model=model, texts=clean_texts, dictionary=id2word, coherence='c_v')
    coherence_values.append(coherencemodel.get_coherence())

limit=40; start=2; step=2;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()