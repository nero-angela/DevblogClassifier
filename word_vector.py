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

    def getCustomModel(self, sentences, embedding_dim=4, window=3, min_count=1, epochs=10):
        """
        주어진 문장들을 기반으로 FastText 단어 임베딩 모델 학습
        
        - input
        : sentences / list / 학습에 사용될 문장 배열
        : embedding_dim / int / 단어 벡터화시 차원 수
        : window / int / 학습에 사용될 n-gram
        : min_count / int / 학습에 사용될 단어의 최소 등장횟수
        : epochs / int / 학습 횟수
        
        - return
        : wv_model
        """
        model = FastText(size=embedding_dim, window=window, min_count=min_count)
        model.build_vocab(sentences=sentences)
        model.train(sentences=sentences, total_examples=len(sentences), epochs=epochs)
        return model
    
    def getWikiModel(self):
        """
        위키 한국어 데이터를 기반으로 FastText 단어 임베딩 모델 학습
        : 기존 학습된 모델이 있는 경우 해당 모델 반환
        : 위키 한국어 데이터(./data/cc.ko.300.bin.gz)가 없는 경우 다운로드
        : 기존 학습된 모델이 없는 경우 학습
        : 학습된 결과를 ./wv_model에 저장
        
        - export
        : self.WIKI_KO_MODEL_PATH
        """
        model = None
        if not os.path.isfile(self.WIKI_KO_MODEL_PATH):
            print('학습된 단어 임베딩 모델이 없습니다.')
            
            if not os.path.isfile(self.WIKI_KO_DATA):
                print('단어 임베딩 모델 학습에 필요한 데이터를 다운로드를 시작합니다.')
                downloadByURL(self.WIKI_KO_DATA_URL, self.WIKI_KO_DATA)
            
            print('단어 임베딩 모델 학습을 시작합니다.')
            model = fasttext.load_facebook_model(self.WIKI_KO_DATA)
            
            print('단어 임베딩 모델을 저장합니다.')
            model.save(self.WIKI_KO_MODEL_PATH)
        else:
            model = FastText.load(self.WIKI_KO_MODEL_PATH)
        
        # print(f'vocab size : {len(model.wv.vocab)}') # 2,000,000
        return model
    
    def getSimilarWords(self, wv_model, word, topn=5):
        """
        유사단어 조회
        
        - input
        : wv_model / FastText 단어 임베딩 모델
        : word / str / 유사도를 측정하려는 단어
        : topn / int / 조회 개수
        """
        return wv_model.wv.similar_by_word(word, topn)
    
    def vectorization(self, wv_model, text, embedding_dim=300):
        """
        주어진 문장을 단어별로 벡터화한 뒤 평균값을 문장의 벡터로 반환
        
        - input
        : wv_model / FastText 단어 임베딩 모델
        : text / str / 벡터화하려는 문장
        : embedding_dim / int / wv_model vector의 차원 수 (wiki 기반 fasttext는 300차원)
        
        - return
        : nparray / shape = (embedding_dim)
        """
        words = text.split(' ')
        words_num = len(words)
        
        # model dimension (wiki festtext의 경우 300)
        vector = np.zeros(embedding_dim)
        for word in words:
            vector += wv_model[word]
        return vector/words_num
