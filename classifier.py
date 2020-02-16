import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import backend as K
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

class Classifier():
    
    def __init__(self,
                 MODEL_PATH='./model/classifier.json',
                 WEIGHT_PATH = './model/classifier.h5'):
        self.MODEL_PATH = MODEL_PATH
        self.WEIGHT_PATH = WEIGHT_PATH
        self.history = None
        
    def _reshape(self, x):
        """
        LSTM 계열의 레이어 사용시 필요한 (total, embedding_dim, 1) 형태의 shape로 변환
        
        - input
        : x / nparray / 변환하려는 배열
        
        - return
        : nparray
        """
        return x.reshape(x.shape[0], x.shape[1], 1)
    
    def _dataSeperator(self, data, test_size=0.33):
        """
        데이터 분할
        
        - input
        : data / DataFrame / documents.csv 데이터
        : test_size / float / 데이터 분할 비율
        
        - return
        : [nparray, nparray, nparray, nparray]
        """
        X_train, X_test, y_train, y_test = train_test_split(data.vector,
                                                            data.label,
                                                            test_size=test_size,
                                                            random_state=321)
        X_train = np.array(X_train.tolist(), dtype=np.float32)
        X_test = np.array(X_test.tolist(), dtype=np.float32)
        y_train = np.array(y_train.tolist(), dtype=np.int32)
        y_test = np.array(y_test.tolist(), dtype=np.int32)
        return X_train, X_test, np.asarray(y_train), np.asarray(y_test)
        
    def train(self,
              data,
              checkpoint_path,
              epochs=75,
              batch_size=100,
              validation_split=0.1,
              verbose=0):
        """
        모델 학습
    
        - input
        : data / DataFrame / documents.csv 데이터
        : checkpoint_path / str / 학습 중간 결과물 저장 경로
        : epochs / int / 학습 횟수
        : batch_size / int / 배치 사이즈
        : validation_split / float / validation data ratio
        : verbose / int / 0 = silent, 1 = progress bar, 2 = one line per epoch.

        - return
        : classifier
        
        - export
        : ./model/classifier.json (graph)
        : ./model/classifier.h5 (weights)
        """
        
        # seperate data
        X_train, X_test, y_train, y_test = self._dataSeperator(data)
        
        # model
        K.clear_session()
        model = Sequential()
        model.add(Dense(100, activation='relu', kernel_initializer='he_normal', input_shape=(X_train.shape[1],)))
        model.add(Dense(80, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['acc',
                               self.f1_m,
                               self.precision_m,
                               self.recall_m])
        model.summary()
        
        # checkpoint
        checkpoint = ModelCheckpoint(filepath=checkpoint_path, mode='max', monitor='val_acc', verbose=2, save_best_only=True)
        callbacks_list = [checkpoint]
        
        self.history = model.fit(X_train,
                            y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=validation_split,
                            callbacks=callbacks_list)
        loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=verbose)
        print(f'loss : {loss}')
        print(f'accuracy : {accuracy}')
        print(f'f1_score : {f1_score}')
        print(f'precision : {precision}')
        print(f'recall : {recall}')
        self.saveModel(model)
        return model
        
    def saveModel(self, model):
        """
        모델의 parameter와 weights를 저장한다.
        
        - input
        : model / classifier
        
        - export
        : ./model/classifier.json / parameter
        : ./model/classifier.h5 / weights
        """
        # save model
        model_json = model.to_json()
        with open(self.MODEL_PATH, "w") as json_file : 
            json_file.write(model_json)
        
        # save weights
        model.save_weights(self.WEIGHT_PATH)
        
    def loadModel(self):
        """
        모델을 불러옴
        
        - return
        : classifier
        """
        # load model
        with open(self.MODEL_PATH, "r") as json_file:
            json_model = json_file.read()
        model = model_from_json(json_model)
        
        # load weight
        model.load_weights(self.WEIGHT_PATH)
        return model
        
    def showHistory(self):
        """
        train history를 그래프로 나타냄
        """
        if self.history == None:
            print('학습내역이 없습니다.')
            return
        
        fig, loss_ax = plt.subplots()
        acc_ax = loss_ax.twinx()
        acc_ax.plot(self.history.history['acc'], 'b', label='train acc')
        acc_ax.plot(self.history.history['val_acc'], 'g', label='val acc')
        acc_ax.set_ylabel('accuracy')
        acc_ax.legend(loc='upper left')
        plt.show()
    
    def recall_m(self, y_true, y_pred):
        """
        재현율(실제 True인 것 중에서 모델이 True라고 예측한 것의 비율) 계산
        
        - input
        : y_true / int / 정답
        : y_pred / int / 모델 예측결과
        
        - return
        : float
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(self, y_true, y_pred):
        """
        정밀도(모델이 True라고 분류한 것 중에서 실제 True인 것의 비율) 계산
        
        - input
        : y_true / int / 정답
        : y_pred / int / 모델 예측결과
        
        - return
        : float
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1_m(self, y_true, y_pred):
        """
        F1 score(Precision과 Recall의 조화평균) 계산
        
        - input
        : y_true / int / 정답
        : y_pred / int / 모델 예측결과
        
        - return
        : float
        """
        precision = self.precision_m(y_true, y_pred)
        recall = self.recall_m(y_true, y_pred)
        return 2 * ((precision * recall)/(precision + recall + K.epsilon()))
