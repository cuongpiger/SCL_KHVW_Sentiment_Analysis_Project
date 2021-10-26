import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import unicodedata
import numpy as np
import pickle
import emojis
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, plot_roc_curve

from typing import List


def textNFCformat(pdata: pd.DataFrame, pcols: List[str]):
    for col in pcols:
        pdata[col] = pdata[col].apply(lambda s: unicodedata.normalize('NFC', s))

    return pdata


def expandEmojisDecode(pcomment: str):
    expand_emojis = ''
    for e in emojis.get(pcomment):
        amount = pcomment.count(e)
        expand_emojis += (f"{emojis.decode(e)[1:-1]} ")*amount
        
    return expand_emojis.strip()  


def dataSplitSaved(pdata: pd.DataFrame, ptest_size: float, ppath: str):
    X_train, X_test, y_train, y_test = train_test_split(pdata.iloc[:, :-1], pdata.iloc[:, -1], test_size=ptest_size, random_state=42)
    
    X_train.to_csv(f"{ppath}/train/X.csv", index=False)
    y_train.to_csv(f"{ppath}/train/y.csv", index=False)

    X_test.to_csv(f"{ppath}/test/X.csv", index=False)
    y_test.to_csv(f"{ppath}/test/y.csv", index=False)
    
    print(f"📢 Your dataset has saved at {ppath}.")

def emojiEvaluationGroupedBarChart(df, head=5):
    fig = go.Figure()
    df = df.head(head)
    x_labels = [f'{x}<br>{y}' for x, y in zip(df['model'], df['vectorizer'])]

    fig.add_trace(go.Bar(x=x_labels, y=df['train_acc']*100, name='Train Acc'))
    fig.add_trace(go.Bar(x=x_labels, y=df['test_acc']*100, name='Test Acc'))
    fig.add_trace(go.Bar(x=x_labels, y=df['train_roc_auc']*100, name='Train ROC-AUC'))
    fig.add_trace(go.Bar(x=x_labels, y=df['test_roc_auc']*100, name='Test ROC-AUC'))

    fig.show()
    
def confusionMatrix(y_true, y_pred, target_names):
    print(classification_report(y_true, y_pred, target_names=target_names))
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index=target_names, columns=target_names)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='g')
    plt.title('Confusion matrix')
    plt.xlabel('Predicted values')
    plt.ylabel('Actual values')
    plt.show()
    
def rocAuc(model, X_vectorizer, y_true):
    plot_roc_curve(model, X_vectorizer, y_true)
    plt.show()
    
def saveByPickle(object, path):
    pickle.dump(object, open(path, "wb"))
    print(f"{object} has been saved at {path}.")
    
def convertToNFC(series):
    return series.apply(lambda x: unicodedata.normalize('NFC', x))


def generateNGrams(ptext: str, pn: int):
    words = ptext.split(" ")
    compounds = zip(*[words[i:] for i in range(pn)])
    return np.array([(' '.join(compound), '_'.join(compound)) for compound in compounds])

def replaceInNGrams(pcomments, pngrams: List[int], min_df: float, max_df: float):
    compound_words = {}
    pngrams = sorted(pngrams, reverse=True)
    dash_comments = []
    
    for i, cmt in enumerate(pcomments):
        for n in pngrams:
            cpws = generateNGrams(cmt, n)
            for cp, cp_ in cpws:
                if compound_words.get(cp, 0) <= i:
                    compound_words[cp] = compound_words.get(cp, 0) + 1
                    
    for cmt in pcomments:
        ngrams_words = []
        for n in pngrams:
            for cp, cp_ in generateNGrams(cmt, n):
                if min_df <= compound_words.get(cp) <= max_df:
                    ngrams_words.append((compound_words.get(cp), cp, cp_))
        
        ngrams_words = sorted(ngrams_words, reverse=True)
        for _, cp, cp_ in ngrams_words:
            cmt = re.sub(cp, f" {cp_} ", cmt)
            
        dash_comments.append(re.sub("\s+", " ", cmt.strip()))
                    
    return dash_comments