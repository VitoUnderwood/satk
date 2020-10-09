from gensim import corpora, models
import re
import jieba

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