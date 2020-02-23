import os
import numpy as np
from absl import app
from flags import create_flags, FLAGS, CONST
from document import Document
from util import downloadByURL, han2Jamo
from gensim.models import FastText, fasttext # 둘이 다름 주의!

"""
FastText base word embedding
"""
class WordEmbedding():
    
    def __init__(self):
        return

    def loadDevblogModel(self,
                         embedding_dim,
                         epochs,
                         window,
                         min_count):
        """
        Devblog 데이터를 기반으로 FastText 단어 임베딩 모델 학습
        
        - input
        : embedding_dim / int / 단어 벡터화시 차원 수
        : epochs / int / 학습 횟수
        : window / int / 학습에 사용될 n-gram
        : min_count / int / 학습에 사용될 단어의 최소 등장횟수
        
        - return
        : we_model
        """
        model = None
        if not os.path.isfile(CONST.devblog_model_path):
            print('🐈  학습된 단어 임베딩 모델이 없습니다.')
            dc = Document()
            docs = dc.getDocs(labeled_only=False) # 전체 데이터 가져오기
            print('🐈  단어 임베딩 모델 학습을 시작합니다.')
            sentences = docs.text.apply(lambda x: [han2Jamo(s) for s in x.split(' ')])
            model = FastText(size=embedding_dim, window=window, min_count=min_count)
            model.build_vocab(sentences=sentences)
            model.train(sentences=sentences, total_examples=len(sentences), epochs=epochs)
            
            print('🐈  단어 임베딩 모델을 저장합니다.')
            model.save(CONST.devblog_model_path)
        else:
            model = FastText.load(CONST.devblog_model_path)
        return model
    
    def loadWikiModel(self):
        """
        위키 한국어 데이터를 기반으로 FastText 단어 임베딩 모델 학습
        : 기존 학습된 모델이 있는 경우 해당 모델 반환
        : 위키 한국어 데이터(./data/cc.ko.300.bin.gz)가 없는 경우 다운로드
        : 기존 학습된 모델이 없는 경우 학습
        : 학습된 결과를 ./we_model에 저장
        
        - export
        : CONST.wiki_model_path
        """
        model = None
        if not os.path.isfile(CONST.wiki_model_path):
            print('🐈  학습된 단어 임베딩 모델이 없습니다.')
            
            if not os.path.isfile(CONST.wiki_data_path):
                print('🐈  단어 임베딩 모델 학습에 필요한 데이터를 다운로드를 시작합니다.')
                downloadByURL(CONST.wiki_data_url, CONST.wiki_data_path)
            
            print('🐈  단어 임베딩 모델 학습을 시작합니다.')
            model = fasttext.load_facebook_model(CONST.wiki_data_path)
            
            print('🐈  단어 임베딩 모델을 저장합니다.')
            model.save(CONST.wiki_model_path)
        else:
            model = FastText.load(CONST.wiki_model_path)
        
        # print(f'vocab size : {len(model.wv.vocab)}') # 2,000,000
        return model
    
    def getSimilarWords(self, we_model, word, topn=5):
        """
        유사단어 조회
        
        - input
        : we_model / FastText 단어 임베딩 모델
        : word / str / 유사도를 측정하려는 단어
        : topn / int / 조회 개수
        """
        return we_model.wv.similar_by_word(word, topn)
    
    def embedding(self, we_model, text, embedding_dim=300):
        """
        주어진 문장을 단어별로 벡터화한 뒤 평균값을 문장의 벡터로 반환
        
        - input
        : we_model / FastText 단어 임베딩 모델
        : text / str / 벡터화하려는 문장
        : embedding_dim / int / we_model vector의 차원 수 (wiki 기반 fasttext는 300차원)
        
        - return
        : nparray / shape = (embedding_dim)
        """
        words = text.split(' ')
        words_num = len(words)
        
        # model dimension (wiki festtext의 경우 300)
        vector = np.zeros(embedding_dim)
        for word in words:
            vector += we_model[word]
        return vector/words_num

def main(_):
    we = WordEmbedding()
    we_model = we.loadWikiModel()
    similar_words = we.getSimilarWords(we_model, FLAGS.predict)
    print(similar_words)

if __name__ == '__main__':
    create_flags(True)
    app.run(main)