from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, Input, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.models import Model, Sequential
from keras.utils.vis_utils import plot_model
from tensorflow.keras.utils import to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import unicodedata
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

def saveByPickle(object, path):
    pickle.dump(object, open(path, "wb"))
    print(f"{object} has been saved at {path}.")

def convertToNFX(series, type: str):
    return series.apply(lambda x: unicodedata.normalize(type, x))

def tokenize(psequences: pd.Series, pnum_words:int=5000):
    tokenizer = Tokenizer(num_words=pnum_words) 
    tokenizer.fit_on_texts(psequences)
    return tokenizer

def padding(pencoded_seqs:pd.Series, pmax_len:int=100):
    return pad_sequences(pencoded_seqs, maxlen=pmax_len)

def defineLSTM(pvocab_size:int, pembedding_dim:int, pnum_units:int, pdropout:float):
    model = Sequential()
    model.add(Input(shape=(100,), name='input'))
    model.add(Embedding(pvocab_size, pembedding_dim, input_length=100, name='embedding'))
    model.add(LSTM(units=pnum_units, dropout=pdropout, recurrent_dropout=pdropout, name='lstm'))
    model.add(Dense(2, activation='softmax', name='output'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def prepareData(pX, py, pval_size=0.1):
    X_train, X_test, y_train, y_test = train_test_split(pX, py, test_size=pval_size, random_state=42)
    return X_train, X_test, y_train, y_test
    
    

def createCallbacks(name):
    tensorboard_callback = TensorBoard(log_dir=os.path.join(os.getcwd(), "tb_log_sentiment", name), write_graph=True,
                                       write_grads=False)
    checkpoint_callback = ModelCheckpoint(filepath=name + "/weights/" + "{epoch:02d}-{val_loss:.6f}.h5",
                                          monitor='val_loss', verbose=0, save_best_only=True)
    return [tensorboard_callback, checkpoint_callback]

def fit(pmodel, pX_train, py_train, pbatch_size, pepochs, psave_path):
    y = to_categorical(py_train)
    his = pmodel.fit(pX_train, y, batch_size=pbatch_size, epochs=pepochs, 
                     validation_split=0.1, callbacks=createCallbacks(name=psave_path[:-3]))
    
    pmodel.save(psave_path)
    print(his)


class SentimentLSTM:
    def __init__(self, pmodel, ptokenizer):
        self.model = pmodel
        self.tokenizer = ptokenizer
        
    def predict(self, pnew_data):
        new_data = self.tokenizer.texts_to_sequences(pnew_data)
        new_data = pad_sequences(new_data, maxlen=100)
        yhat_proba = self.model.predict(new_data)
        
        return pd.DataFrame({
            'input': pnew_data,
            'output_proba': [tuple(x) for x in yhat_proba],
            'output_class': np.argmax(yhat_proba, axis=1)
        })
        
        
class SentimentCNN1D:
    def __init__(self) -> (None):
        pass
    
    def _initTokenizer(self, pdata: pd.Series, pnum_words:int=None):
        self.tokenizer = Tokenizer(num_words=pnum_words)
        self.tokenizer.fit_on_texts(pdata)
        
    def _loadWordVectors(self, pfasttext):
        self.word_vectors = pfasttext
        
    def _embedIndex2Matrix(self, pembedding_dim=100):
        num_words = len(self.getTokenizerWordIndex())
        self.embeded_matrix = np.zeros((num_words, pembedding_dim))
        for word, i in self.getTokenizerWordIndex().items():
            if i >= num_words:
                continue
            self.embeded_matrix[i] = self.word_vectors.get_word_vector(word)
            
    def _callback(self, pname):
        tensorboard_callback = TensorBoard(log_dir=os.path.join(os.getcwd(), "tb_log_sentiment", pname), write_graph=True,
                                       write_grads=False)
        checkpoint_callback = ModelCheckpoint(filepath=pname + "/weights/" + "{epoch:02d}-{val_loss:.6f}.h5",
                                          monitor='val_loss', verbose=0, save_best_only=True)
        return [tensorboard_callback, checkpoint_callback]
        
    def _defineModel(self, pno_neurons=10, pno_filters=5, pembedding_dim=100, pseq_length=100):
        seq_input = Input(shape=(pseq_length,), dtype='int32', name='input')
        embedding_layer = Embedding(input_dim=len(self.getTokenizerWordIndex()),
                                    output_dim=pembedding_dim,
                                    weights=[self.embeded_matrix],
                                    input_length=pseq_length,
                                    trainable=False,
                                    name='embedding')(seq_input)
        layer = Conv1D(pno_neurons, pno_filters, activation='relu')(embedding_layer)
        layer = MaxPooling1D(pno_filters)(layer)
        layer = Conv1D(pno_neurons, pno_filters, activation='relu')(layer)
        layer = GlobalMaxPooling1D()(layer)
        layer = Dense(pno_neurons, activation='relu')(layer)
        output = Dense(2, activation='softmax', name='output')(layer)
        self.model = Model(seq_input, output)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        
    def define(self, pX, py, pfasttext, pno_neurons=10, pno_filters=5, pnum_words=None, pseq_length=100, pembedding_dim=100, **kwargs):
        self._initTokenizer(pdata=convertToNFX(pX, 'NFC'), pnum_words=pnum_words)
        self._loadWordVectors(pfasttext)
        self.X = pad_sequences(self.tokenizer.texts_to_sequences(pX), maxlen=pseq_length)
        self.y = to_categorical(py)
        self._embedIndex2Matrix(pembedding_dim)
        self._defineModel(pno_neurons, pno_filters, pembedding_dim, pseq_length)
        
    def fit(self, pbatch_size=32, pepochs=10, psave_path=None, **kwargs):
        self.model.fit(self.X, self.y, batch_size=pbatch_size, epochs=pepochs,
                       validation_split=0.1, callbacks=self._callback(psave_path[:-3]))
        
    def getTokenizerWordIndex(self):
        return self.tokenizer.word_index
    
    def save(self, pmodel_path:str, ptoken_path:str):
        self.model.save(pmodel_path)
        saveByPickle(self.tokenizer, ptoken_path)
        print(f"📢 Model has been saved at {pmodel_path} - Tokenizer has been saved at {ptoken_path}.")
        