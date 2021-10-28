from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, Input
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

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
    checkpoint_callback = ModelCheckpoint(filepath=name + "/weights/" + "{epoch:02d}-{val_loss:.6f}.hdf5",
                                          monitor='val_loss', verbose=0, save_best_only=True)
    return [tensorboard_callback, checkpoint_callback]

def fit(pmodel, pX_train, py_train, pbatch_size, pepochs, psave_path):
    y = to_categorical(py_train)
    his = pmodel.fit(pX_train, y, batch_size=pbatch_size, epochs=pepochs, 
                     validation_split=0.1, callbacks=createCallbacks(name=psave_path[:-3]))
    
    pmodel.save(psave_path)
    print(his)
    # score, acc = pmodel.evaluate(x=pX_val, y=py_val, batch_size=pbatch_size)
    # print('Test loss:', score)
    # print('Test accuracy:', acc)