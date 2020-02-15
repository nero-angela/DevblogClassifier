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
        return x.reshape(x.shape[0], x.shape[1], 1)
    
    def _dataSeperator(self, data):
        X_train, X_test, y_train, y_test = train_test_split(data.vector,
                                                            data.label,
                                                            test_size=0.33,
                                                            random_state=321)
        X_train = np.array(X_train.tolist(), dtype=np.float32)
        X_test = np.array(X_test.tolist(), dtype=np.float32)
        y_train = np.array(y_train.tolist(), dtype=np.int32)
        y_test = np.array(y_test.tolist(), dtype=np.int32)
        return X_train, X_test, np.asarray(y_train), np.asarray(y_test)
        
    def train(self,
              data,
              checkpoint_path,
              embedding_dim=300,
              epochs=75,
              batch_size=100,
              validation_split=0.3,
              verbose=0):
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
        # save model
        model_json = model.to_json()
        with open(self.MODEL_PATH, "w") as json_file : 
            json_file.write(model_json)
        
        # save weight
        model.save_weights(self.WEIGHT_PATH)
        
    def loadModel(self):
        # load model
        with open(self.MODEL_PATH, "r") as json_file:
            json_model = json_file.read()
        model = model_from_json(json_model)
        
        # load weight
        model.load_weights(self.WEIGHT_PATH)
        return model
        
    def showHistory(self):
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
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(self, y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision

    def f1_m(self, y_true, y_pred):
        precision = self.precision_m(y_true, y_pred)
        recall = self.recall_m(y_true, y_pred)
        return 2 * ((precision * recall)/(precision + recall + K.epsilon()))
