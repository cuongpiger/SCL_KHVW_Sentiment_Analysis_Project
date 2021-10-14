import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import modules.user_object_defined as udt
import statsmodels.api as sm
import unicodedata
from statsmodels.formula.api import ols
from wordcloud import WordCloud
from typing import Dict, Tuple


def pieChart(previews: udt.Dataframe):
    plt.figure(figsize=(7, 7))
    negative_class = previews[previews['label'] == 0].shape[0]
    positive_class = previews.shape[0] - negative_class
    colors = sns.color_palette('tab10', 4)[2:]
    
    plt.pie([positive_class, negative_class], labels=[f'Positive - {positive_class} samples', f'Negative - {negative_class} samples'], colors=colors, autopct="%.0f%%")
    plt.show()
    
def emojiClassBarplot(previews: udt.Dataframe):
    plt.figure(figsize=(10, 10))
    
    negative_class = previews[previews['label'] == 0]
    negative_emoji = negative_class[negative_class['emoji'] != ""].shape[0]
    negative_noemoji = negative_class.shape[0] - negative_emoji
    
    positive_class = previews[previews['label'] == 1]
    positive_emoji = positive_class[positive_class['emoji'] != ""].shape[0]
    positive_noemoji = positive_class.shape[0] - positive_emoji
    
    have_emoji = [negative_emoji, positive_emoji]
    havent_emoji = [negative_noemoji, positive_noemoji]
    ind = np.arange(2)
    
    bar1 = plt.bar(ind, have_emoji, 0.5, label="Yes")
    bar2 = plt.bar(ind, havent_emoji, 0.5, label="No", bottom=have_emoji)
    
    plt.ylabel("Amount")
    plt.xticks(ind, ("Negative", "Positive"))
    plt.yticks(np.arange(0, negative_class.shape[0] + 10, 500))
    plt.legend(title="Have emoji\n")
    plt.show()
    
def lengthDistplot(previews: udt.Dataframe):
    plt.figure(figsize=(10, 7))
    sns.distplot(previews.loc[previews.label == 0, 'length'], color='red', label='Negative')
    sns.distplot(previews.loc[previews.label == 1, 'length'], color='green', label='Positive')
    plt.legend()
    plt.show()
    
def noWordsFrequency(previews: udt.Dataframe):
    fig = plt.figure(figsize=(20, 7))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.set_title("Negitive")
    ax2.set_title("Positive")
    previews.loc[previews.label == 0, 'no_words'].hist(bins=30, ax=ax1)
    previews.loc[previews.label == 1, 'no_words'].hist(bins=30, ax=ax2)
    fig.tight_layout()
    plt.show()
    
def boxplotDescribeStatistics(previews: udt.Dataframe):
    df = previews.copy()
    df['label'] = df['label'].apply(lambda x: "Positive" if x > 0 else "Negative")
    fig, axs = plt.subplots(nrows=2, figsize=(10, 15))
    sns.boxplot(data=df, x='length', y='label', ax=axs[0])
    sns.boxplot(data=df, x='no_words', y='label', ax=axs[1])
    plt.show()
    
def onewayANOVA(previews: udt.Dataframe, continous_var: str, post_hoc: bool = False):
    df = previews.copy()
    df['label'] = df['label'].apply(lambda x: "Positive" if x > 0 else "Negative")
    mod = ols(f'{continous_var} ~ label', data=df).fit()
    aov_tbl = sm.stats.anova_lm(mod, type=2)
    
    if not post_hoc: print(aov_tbl)
    
    if post_hoc:
        return mod.t_test_pairwise('label').result_frame
    
def regplotLengthNoWords(previews: udt.Dataframe):
    df = previews.copy()
    df['label'] = df['label'].apply(lambda x: "Positive" if x > 0 else "Negative")
    plt.figure(figsize=(12, 10))
    sns.lmplot(data=df, x='length', y='no_words', hue='label')
    plt.show()

def commentWordCloud(pcolumn_df: udt.DfStrColumn):
    wc = WordCloud(background_color='white', width=1600, height=800).generate(
        unicodedata.normalize('NFC', " ".join(pcolumn_df.to_list()))
    )
    plt.figure(figsize=(20, 10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    
def createWordDictionary(pcolumn: udt.DfStrColumn):
    word_set = set((" ".join(pcolumn)).split(" "))
    
    word_id: Dict[str, int] = {}
    id_word: Dict[int, str] = {}
    for id, word in enumerate(word_set):
        word_id[word] = id
        id_word[id] = word
        
    return word_id, id_word

def createBagOfWordsFrequency(pcolumn: udt.DfStrColumn):
    wordfreq = {}
    for sen in pcolumn:
        for word in sen.split(" "):
            wordfreq[word] = wordfreq.get(word, 0) + 1
        
    return sorted(wordfreq.items(), key=lambda x: x[1], reverse=True)

def bagOfWordsGetRangeBased(pword_freq: Dict[str, int], prange: Tuple[int, int]):
    words = []
    prange = sorted(prange, reverse=True)
    
    for key, value in pword_freq:
        if value > prange[0]: continue
        elif value >= prange[1]:
            words.append((key, value))
        else: break
        
    return words