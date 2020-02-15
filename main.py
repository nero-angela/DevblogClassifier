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

def train():
    # load data
    data = getTrainData()
    
    # train
    model = cf.train(data,
                 checkpoint_path='./model/200215',
                 epochs=10,
                 batch_size=100,
                 validation_split=0.1,
                 verbose=0)
    cf.showHistory()

def getTrainData():
    # 라벨링 된 데이터만 가져오기
    data = doc.getDocs(True) 
    
    # vectorization
    data['vector'] = data.text.apply(lambda x: wv.vectorization(wv_model, x))
    return data

def isDevPosting(text, criterion=0.5):
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
    # 학습 데이터 분석
    Analysis(doc.getDocs(labeled_only=False))
    
def getSimilarWords(text, topn=5):
    # 유사 단어 조회
    return wv.getSimilarWords(wv_model, text, topn)
