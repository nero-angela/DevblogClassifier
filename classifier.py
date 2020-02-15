import numpy as np
import matplotlib.pyplot as plt
from util import reshape
from keras import backend as K
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense

class Classifier():
    
    def __init__(self):
        return
    
    def _dataSeperator(self, data, embedding_dim):
        X_train, X_test, y_train, y_test = train_test_split(data.vector,
                                                            data.label,
                                                            test_size=0.33,
                                                            random_state=321)
        X_train = reshape(X_train, embedding_dim)
        X_test = reshape(X_test, embedding_dim)
        return X_train, X_test, y_train, y_test
        
    def train(self,
              data,
              embedding_dim=300,
              epochs=75,
              batch_size=100,
              validation_split=0.3,
              verbose=0):
        # seperate data
        X_train, X_test, y_train, y_test = self._dataSeperator(data, embedding_dim)
        
        # layer
        K.clear_session()
        model = Sequential()
        model.add(Dense(embedding_dim, input_shape=(X_train.shape[1], 1), activation='relu'))
        model.add(SimpleRNN(32))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['acc',
                               self.f1_m,
                               self.precision_m,
                               self.recall_m])
        history = model.fit(X_train,
                            np.asarray(y_train),
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=validation_split)
        
        loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, np.asarray(y_test), verbose=verbose)
        print(f'loss : {loss}')
        print(f'accuracy : {accuracy}')
        print(f'f1_score : {f1_score}')
        print(f'precision : {precision}')
        print(f' recall : {recall}')
        
    def showHistory(self, history):
        fig, loss_ax = plt.subplots()
        acc_ax = loss_ax.twinx()

        acc_ax.plot(history.history['acc'], 'b', label='train acc')
        acc_ax.plot(history.history['val_acc'], 'g', label='val acc')
        acc_ax.set_ylabel('accuracy')
        acc_ax.legend(loc='upper left')

        plt.show()
    
    def predict(self):
        return
    
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
