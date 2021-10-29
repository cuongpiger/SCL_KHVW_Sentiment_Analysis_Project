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
import numpy as np
from random import random
from math import log, ceil
from time import time, ctime

def saveByPickle(object, path):
    pickle.dump(object, open(path, "wb"))
    print(f"{object} has been saved at {path}.")

def convertToNFX(series, type: str):
    return series.apply(lambda x: unicodedata.normalize(type, x))

class SentimentLSTM:
    def __init__(self) -> (None):
        pass
    
    def _initTokenizer(self, pdata: pd.Series, pnum_words:int=None):
        self.tokenizer = Tokenizer(num_words=pnum_words)
        self.tokenizer.fit_on_texts(pdata)
        
    def _callback(self, pname):
        tensorboard_callback = TensorBoard(log_dir=os.path.join(os.getcwd(), "tb_log_sentiment", pname), write_graph=True,
                                       write_grads=False)
        checkpoint_callback = ModelCheckpoint(filepath=pname + "/weights/" + "{epoch:02d}-{val_loss:.6f}.h5",
                                          monitor='val_loss', verbose=0, save_best_only=True)
        return [tensorboard_callback, checkpoint_callback]
        
    def _defineModel(self, pno_units:int=10, pdropout:float=0.2, poptimizer:str='adam'):
        vocab_size = len(self.tokenizer.word_index) + 1
        embedding_dim = self.X.shape[1]
        
        self.model = Sequential()
        self.model.add(Input(shape=(100,), name='input'))
        self.model.add(Embedding(vocab_size, embedding_dim, input_length=100, name='embedding'))
        self.model.add(LSTM(units=pno_units, dropout=pdropout, recurrent_dropout=pdropout, name='lstm'))
        self.model.add(Dense(2, activation='softmax', name='output'))
        self.model.compile(loss='binary_crossentropy', optimizer=poptimizer, metrics=['accuracy'])
        
            
    def define(self, pX, py, pnum_words:int=None, pseq_length:int=None, poptimizer:str='adam', pno_units:int=10, pdropout:float=0.2, **kwargs):
        pX = convertToNFX(pX, 'NFC')
        self._initTokenizer(pdata=pX, pnum_words=pnum_words)  
        self.X = pad_sequences(self.tokenizer.texts_to_sequences(pX), maxlen=pseq_length)
        self.y = to_categorical(py)        
        self._defineModel(pno_units, pdropout, poptimizer)
        
    def fit(self, pbatch_size:int=32, pepochs:int=10, psave_weights:str=None, **kwargs):
        self.model.fit(self.X, self.y, batch_size=pbatch_size, epochs=pepochs,
                       validation_split=0.1, callbacks=self._callback(psave_weights[:-3]))
              
    def predict(self, pnew_data):
        new_data = self.tokenizer.texts_to_sequences(convertToNFX(pnew_data, 'NFC'))
        new_data = pad_sequences(new_data, maxlen=self.X.shape[1])
        yhat_proba = self.model.predict(new_data)

        return pd.DataFrame({
            'input': pnew_data,
            'output_proba': [tuple(x) for x in yhat_proba],
            'output_class': np.argmax(yhat_proba, axis=1)
        })
        
    def evaluation(self):
        ixs = np.random.choice(self.X.shape[0], int(self.X.shape[0]*0.1), replace=False)
        loss = self.model.evaluate(self.X[ixs], self.y[ixs])
        
        return {'loss': loss}
        
    def getTokenizerWordIndex(self):
        return self.tokenizer.word_index
    
    def save(self, pmodel_path:str, ptoken_path:str):
        self.model.save(pmodel_path)
        saveByPickle(self.tokenizer, ptoken_path)
        print(f"📢 Model has been saved at {pmodel_path} - Tokenizer has been saved at {ptoken_path}.") 
        
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
        
class Hyperband:
    def __init__(self, get_params, try_params, max_iter=81):
        self.get_params = get_params
        self.try_params = try_params

        self.max_iter = max_iter  # maximum iterations per configuration
        self.eta = 3  # defines configuration downsampling rate (default = 3)

        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.max_iter))
        self.B = (self.s_max + 1) * self.max_iter

        self.results = []  # list of dicts
        self.counter = 0
        self.best_loss = np.inf
        self.best_counter = -1

    # can be called multiple times
    def run(self, skip_last=0, dry_run=False):

        for s in reversed(range(self.s_max + 1)):

            # initial number of configurations
            n = int(ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))

            # initial number of iterations per config
            r = self.max_iter * self.eta ** (-s)

            # n random configurations
            T = [self.get_params() for i in range(n)]

            for i in range((s + 1) - int(skip_last)):  # changed from s + 1

                # Run each of the n configs for <iterations>
                # and keep best (n_configs / eta) configurations

                n_configs = n * self.eta ** (-i)
                n_iterations = r * self.eta ** (i)

                print("\n*** {} configurations x {:.1f} iterations each".format(
                    n_configs, n_iterations))

                val_losses = []
                early_stops = []

                for t in T:

                    self.counter += 1
                    print("\n{} | {} | lowest loss so far: {:.4f} (run {})\n".format(
                        self.counter, ctime(), self.best_loss, self.best_counter))

                    start_time = time()

                    if dry_run:
                        result = {'loss': random(), 'log_loss': random(), 'auc': random()}
                    else:
                        result = self.try_params(**t)  # <---

                    assert (type(result) == dict)
                    assert ('loss' in result)

                    seconds = int(round(time() - start_time))
                    print("\n{} seconds.".format(seconds))

                    loss = result['loss']
                    val_losses.append(loss)

                    early_stop = result.get('early_stop', False)
                    early_stops.append(early_stop)

                    # keeping track of the best result so far (for display only)
                    # could do it be checking results each time, but hey
                    if loss < self.best_loss:
                        self.best_loss = loss
                        self.best_counter = self.counter

                    result['counter'] = self.counter
                    result['seconds'] = seconds
                    result['params'] = t
                    result['iterations'] = n_iterations

                    self.results.append(result)

                # select a number of best configurations for the next loop
                # filter out early stops, if any
                indices = np.argsort(val_losses)
                T = [T[i] for i in indices if not early_stops[i]]
                T = T[0:int(n_configs / self.eta)]

        return self.results