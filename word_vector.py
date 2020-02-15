import os
import numpy as np
from util import downloadByURL
from gensim.models import FastText, fasttext # 둘이 다름 주의!

"""
FastText base word embedding
"""
class WordVector():
    
    def __init__(self):
        # corpus
        self.WIKI_KO_DATA = './data/cc.ko.300.bin.gz'
        self.WIKI_KO_DATA_URL = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ko.300.bin.gz'

        # pretrained model
        self.WIKI_KO_MODEL_PATH = f'./wv_model/ko.wiki'

    def getCustomModel(self, text, size=4, window=3, min_count=1, epochs=10):
        """
        FastText 기반 모델 학습
        """
        model = FastText(size=size, window=window, min_count=min_count)
        model.build_vocab(sentences=text)
        model.train(sentences=text, total_examples=len(text), epochs=epochs)
        return model
    
    def getWikiModel(self):
        """
        위키 한국어 데이터 기반 모델 학습
        """
        model = None
        if not os.path.isfile(self.WIKI_KO_MODEL_PATH):
            print('학습된 모델이 없습니다.')
            
            if not os.path.isfile(self.WIKI_KO_DATA):
                print('모델 학습에 필요한 데이터를 다운로드를 시작합니다.')
                downloadByURL(self.WIKI_KO_DATA_URL, self.WIKI_KO_DATA)
            
            print('모델 학습을 시작합니다.')
            model = fasttext.load_facebook_model(self.WIKI_KO_DATA)
            model.save(self.WIKI_KO_MODEL_PATH)
            
        else:
            model = FastText.load(self.WIKI_KO_MODEL_PATH)
        
        # print(f'vocab size : {len(model.wv.vocab)}') # 2,000,000
        return model
    
    def getSimilarWords(self, wv_model, word, topn=5):
        """
        유사단어 조회
        """
        return wv_model.wv.similar_by_word(word, topn)
    
    def vectorization(self, wv_model, text, embedding_dim=300):
        """
        주어진 문장을 단어별로 벡터화한 뒤 평균값을 문장의 벡터로 반환
        embedding_dim : wv_model vector의 차원 수 (wiki 기반 fasttext는 300차원)
        """
        words = text.split(' ')
        words_num = len(words)
        
        # model dimension (wiki festtext의 경우 300)
        vector = np.zeros(embedding_dim)
        for word in words:
            vector += wv_model[word]
        return vector/words_num
