from document import Document
from analysis import Analysis
from word_vector import WordVector
from classifier import Classifier
import pandas as pd
import numpy as np

doc = Document()
wv = WordVector()
cf = Classifier()
wv_model = wv.getWikiModel()
model = cf.loadModel()

def train(epochs=10,
          batch_size=100,
          validation_split=0.1,
          verbose=0,
          checkpoint_path='./model/checkpoints'):
    """
    classifier 학습
    
    - input
    : epochs / int / 학습 횟수
    : batch_size / int / 배치 사이즈
    : validation_split / float / validation data ratio
    : checkpoint_path / str / 학습 중간 결과물 저장 경로
    
    - export
    : ./model/classifier.json (graph)
    : ./model/classifier.h5 (weights)
    """
    
    # load data
    data = getTrainData()
    
    # train
    model = cf.train(data,
                 checkpoint_path=checkpoint_path,
                 epochs=epochs,
                 batch_size=batch_size,
                 validation_split=validation_split,
                 verbose=verbose)
    cf.showHistory()

def getTrainData():
    """
    라벨링 된 데이터를 임베딩하여 반환
    
    - return
    : DataFrame
    """
    
    # 라벨링 된 데이터만 가져오기
    data = doc.getDocs(True) 
    
    # vectorization
    data['vector'] = data.text.apply(lambda x: wv.vectorization(wv_model, x))
    return data

def predict(text, criterion=0.5):
    """
    개발관련 문서여부 반환
    
    - input
    : text / str / 확인하려는 문서 내용 (영어 또는 한글이 포함되어있어야함)
    : criterion / float / 개발관련 문서 판단 기준
    
    - return
    : boolean / 개발문서 여부
    : float / 1에 가까울수록 개발관련 문서
    """
    data = doc.preprocessing(pd.DataFrame([{
        'title': text,
        'description': '',
        'tags': []
    }]))
    data = data.text.apply(lambda x: wv.vectorization(wv_model, x)).tolist()

    if len(data) == 0:
        print('text is not valid')
        return
    data = np.array(data)
    confidence = round(model.predict(data)[0][1], 3)
    is_dev_doc = confidence > criterion
    return is_dev_doc, confidence

def getDataAnalysis():
    """
    가지고있는 전체 데이터 분석
    : label 별 수량
    : 문장 길이 histogram
    : WordCloud 
    """
    # 학습 데이터 분석
    Analysis(doc.getDocs(labeled_only=False))
    
def getSimilarWords(text, topn=5):
    """
    단어 임베딩 모델을 이용하여 주어진 단어와 유사도가 높은 단어를 반환
    
    - input
    : text / str / 유사도를 구하려는 단어
    : topn / int / 조회하려는 단어의 개수(유사도가 높은 순서로 자름)
    
    - return
    : list / [(string : word, float : similarity)]
    """
    # 유사 단어 조회
    return wv.getSimilarWords(wv_model, text, topn)
