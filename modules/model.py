import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import unicodedata
import numpy as np
import pickle
import emojis
import time
import re
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, plot_roc_curve
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score

from typing import List


def textNFxformat(pdata: pd.DataFrame, pcols: List[str], type):
    for col in pcols:
        pdata[col] = pdata[col].apply(lambda s: unicodedata.normalize(type, s))

    return pdata


def expandEmojisDecode(pcomment: str):
    expand_emojis = ''
    for e in emojis.get(pcomment):
        amount = pcomment.count(e)
        expand_emojis += (f"{emojis.decode(e)[1:-1]} ")*amount
        
    return expand_emojis.strip()  

def loadData(ppath: str):
    X = pd.read_csv(f"{ppath}/X.csv")
    y = pd.read_csv(f"{ppath}/y.csv")
    
    return X, y


def reviewStatistic(pdata: pd.Series, pis_emoji=False):
    words_dict = {}
    
    for i, sen in enumerate(pdata):
        for word in sen.split(' '):
            if words_dict.get(word, None) == None:
                words_dict[word] = [1, 1] # freq whole, freq document
            else:
                words_dict[word][0] += 1
                words_dict[word][1] += int(words_dict[word][1] <= i)
    
    if not pis_emoji:     
        return pd.DataFrame({
            'word': words_dict.keys(),
            'freq': [v for v, _ in words_dict.values()],
            'freq_doc': [v for _, v in words_dict.values()]
        }).sort_values(by=['freq', 'freq_doc']).reset_index(drop=True)

    return pd.DataFrame({
        'word': words_dict.keys(),
        'encode': [emojis.encode(f":{k}:") for k in words_dict.keys()],
        'freq': [v for v, _ in words_dict.values()],
        'freq_doc': [v for _, v in words_dict.values()]
    }).sort_values(by=['freq', 'freq_doc']).reset_index(drop=True)
    
def convertToNFX(series, type: str):
    return series.apply(lambda x: unicodedata.normalize(type, x))
    
    
def wordFrequencyBarplot(pdata: pd.DataFrame):
    ind = np.arange(len(pdata))
    width = 0.4

    fig, ax = plt.subplots(figsize=(10, 30))
    b1 = ax.barh(ind, pdata['freq'], width, color='red', label='Entire-based')
    b2 = ax.barh(ind + width, pdata['freq_doc'], width, color='green', label='Document-based')
    
    ax.set(yticks=ind + width, yticklabels=pdata['word'], ylim=[2*width -1, len(pdata)])
    ax.legend()
    ax.bar_label(b1, label_type='edge')
    ax.bar_label(b2, label_type='edge')

    plt.show()
    

def vectorizer(pdata: pd.Series, pmethod: str ,pmin_df=1, pmax_df=1.0):
    pdata = convertToNFX(pdata, 'NFC')
    if pmethod == 'bow': vec = CountVectorizer(min_df=pmin_df, max_df=pmax_df)
    else: vec = TfidfVectorizer(min_df=pmin_df, max_df=pmax_df)
    
    transform = vec.fit_transform(pdata)
    return [vec, transform]


def dataSplitSaved(pdata: pd.DataFrame, ptest_size: float, ppath: str):
    X_train, X_test, y_train, y_test = train_test_split(pdata.iloc[:, :-1], pdata.iloc[:, -1], test_size=ptest_size, random_state=42)
    
    X_train.to_csv(f"{ppath}/train/X.csv", index=False)
    y_train.to_csv(f"{ppath}/train/y.csv", index=False)

    X_test.to_csv(f"{ppath}/test/X.csv", index=False)
    y_test.to_csv(f"{ppath}/test/y.csv", index=False)
    
    print(f"📢 Your dataset has saved at {ppath}.")
    
def train(lst_models, X_vectorizer, y, cv):
    res_table = []
    for vec_name, vec in X_vectorizer:
        print(f"{vec_name}:")
        X = vec[1]
        for mdl_name, model in lst_models:
            tic = time.time()
            cv_res = cross_validate(model, X, y, cv=cv, return_train_score=True, scoring=['accuracy', 'roc_auc'])
            res_table.append([vec_name, mdl_name,
                              cv_res['train_accuracy'].mean(),
                              cv_res['test_accuracy'].mean(),
                              np.abs(cv_res['train_accuracy'].mean() - cv_res['test_accuracy'].mean()),
                              cv_res['train_accuracy'].std(),
                              cv_res['test_accuracy'].std(),
                              cv_res['train_roc_auc'].mean(),
                              cv_res['test_roc_auc'].mean(),
                              np.abs(cv_res['train_roc_auc'].mean() - cv_res['test_roc_auc'].mean()),
                              cv_res['train_roc_auc'].std(),
                              cv_res['test_roc_auc'].std(),
                              cv_res['fit_time'].mean()
            ])
            toc = time.time()
            print('\tModel {} has been trained in {:,.2f} seconds'.format(mdl_name, (toc - tic)))
            
    
    res_table = pd.DataFrame(res_table, columns=['vectorizer', 'model', 'train_acc', 'test_acc', 'diff_acc',
                                                 'train_acc_std', 'test_acc_std', 'train_roc_auc', 'test_roc_auc',
                                                 'diff_roc_auc', 'train_roc_auc_std', 'test_roc_auc_std', 'fit_time'])
    res_table.sort_values(by=['test_acc', 'test_roc_auc'], ascending=False, inplace=True)
    return res_table.reset_index(drop=True)    
    

def emojiEvaluationGroupedBarChart(pdata, phead=5):
    fig = go.Figure()
    pdata = pdata.head(phead)
    x_labels = [f'{x}<br>{y}' for x, y in zip(pdata['model'], pdata['vectorizer'])]

    fig.add_trace(go.Bar(x=x_labels, y=pdata['train_acc']*100, name='Train Acc'))
    fig.add_trace(go.Bar(x=x_labels, y=pdata['test_acc']*100, name='Test Acc'))
    fig.add_trace(go.Bar(x=x_labels, y=pdata['train_roc_auc']*100, name='Train ROC-AUC'))
    fig.add_trace(go.Bar(x=x_labels, y=pdata['test_roc_auc']*100, name='Test ROC-AUC'))

    fig.show()
    
def trainTunningModel(lst_models, X_vectorizer, y, cv):
    models_final = []
    for model_name, model, params in lst_models:
        tic = time.time()
        search = GridSearchCV(estimator=model, param_grid=params, cv=cv, scoring='accuracy')
        search.fit(X_vectorizer, y)
        model_tunned = model.set_params(**search.best_params_)
        models_final.append((model_name, model_tunned))
        toc = time.time()
        print('Model {} has been tunned in {:,.2f} seconds'.format(model_name, (toc - tic)))
        
    return models_final    
    
    
def evaluation(tunning_models, X_train_vec, y_train, X_test_vec, y_test):
    res = []
    for name, model in tunning_models:
        model.fit(X_train_vec, y_train)
        y_train_pred = model.predict(X_train_vec)
        y_test_pred = model.predict(X_test_vec)
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        train_roc_auc = roc_auc_score(y_train, y_train_pred)
        test_roc_auc = roc_auc_score(y_test, y_test_pred)
        res.append([name, train_acc, test_acc, train_roc_auc, test_roc_auc])
        
    res = pd.DataFrame(res, columns=['model', 'train_acc', 'test_acc', 'train_roc_auc', 'test_roc_auc'])
    res.sort_values(by=['test_acc', 'test_roc_auc'], ascending=False, inplace=True)
    
    return res.reset_index(drop=True)
    
    
    
def confusionMatrix(y_true, y_pred):
    target_names = ['Negative', 'Positive']
    print(classification_report(y_true, y_pred, target_names=target_names))
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index=target_names, columns=target_names)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='g')
    plt.title('Confusion matrix')
    plt.xlabel('Predicted values')
    plt.ylabel('Actual values')
    plt.show()

    
def saveByPickle(object, path):
    pickle.dump(object, open(path, "wb"))
    print(f"{object} has been saved at {path}.")


def generateNGrams(ptext: str, pn: int):
    words = ptext.split(" ")
    compounds = zip(*[words[i:] for i in range(pn)])
    return np.array([(' '.join(compound), '_'.join(compound)) for compound in compounds])

def getDashWords(pcomments, pngrams):
    ngram_words = {}
    for i, cmt in enumerate(pcomments):
        for n in pngrams:
            cpws = generateNGrams(cmt, n)
            for cp, cp_ in cpws:
                if ngram_words.get(cp, None) == None:
                    ngram_words[cp] = [1, 1] # dash, entrire, doc
                else:
                    ngram_words[cp][0] += 1
                    ngram_words[cp][1] += int(ngram_words[cp][1] <= i)
                    
    df = pd.DataFrame({
        'freq_doc': [v for _, v in ngram_words.values()],
        'freq': [v for v, _ in ngram_words.values()]
    }, index=ngram_words.keys())
    
    df = df.sort_values(by=['freq_doc', 'freq'], ascending=False)
    return df
        
        
def replaceInNGrams(pcomments, pngrams: List[int], ngrams_dict, on_col, min_df=1, max_df=99999999999):                    
    dash_comments = []
    for cmt in pcomments:
        ngrams_words = []
        for n in pngrams:
            for cp, cp_ in generateNGrams(cmt, n):
                if ngrams_dict.get(cp, None) is None: continue
                if min_df <= ngrams_dict.loc[cp, on_col] <= max_df:
                    ngrams_words.append((ngrams_dict.loc[cp, on_col], cp, cp_))
        
        ngrams_words = sorted(ngrams_words, reverse=True)
        for _, cp, cp_ in ngrams_words:
            cmt = re.sub(cp, f" {cp_} ", cmt)
            
        dash_comments.append(re.sub("\s+", " ", cmt.strip()))
                    
    return dash_comments

def combinePrediction(a, b, c):
    neg = (a[0] + b[0] + c[0])/3
    pos = (a[1] + b[1] + c[1])/3
    
    return 0 if neg > pos else 1    

class SentimentModel:
    def __init__(self, pmodel, pvector, py):
        self.model = pmodel
        self.vectorizer = pvector[1][0]
        self.model.fit(pvector[1][1], py)
        
    def predict(self, pnew_data):
        new_data = self.vectorizer.transform(pnew_data)
        yhat_class = self.model.predict(new_data)
        yhat_proba = self.model.predict_proba(new_data)
        
        return pd.DataFrame({
            'input': pnew_data,
            'output_proba': [tuple(x) for x in yhat_proba],
            'output_class': yhat_class, 
        })
        
    def info(self):
        print(self.model)
        
    def rocAuc(self, X, y_true):
        X_vec = self.vectorizer.transform(X)
        plot_roc_curve(self.model, X_vec, y_true)
        plt.show()