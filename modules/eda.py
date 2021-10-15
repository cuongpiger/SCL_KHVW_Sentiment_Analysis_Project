import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow.python.keras.backend import count_params
import modules.user_object_defined as udt
import statsmodels.api as sm
import unicodedata
import plotly
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE
from statsmodels.formula.api import ols
from wordcloud import WordCloud
from typing import Dict, Tuple, List


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
        
    return pd.DataFrame(sorted(wordfreq.items(), key=lambda x: x[1], reverse=True), columns=('word', 'freq'))

def bagOfWordsGetRangeBased(pword_freq: udt.Dataframe, prange: Tuple[int, int]):
    prange = sorted(prange)
    return pword_freq.loc[(pword_freq['freq'] >= prange[0]) & (pword_freq['freq'] <= prange[1])]

def wordFrequencyGroupBarplot(pwordfreq: List[int], yscale=False):
    fig, ax = plt.subplots(figsize=(15, 7))
    bins = np.linspace(0, max(pwordfreq) + 10, 51, dtype=np.int)
    lbls = [f"({bins[i]} ~ {bins[i + 1]})" for i in range(50)]
    freq = np.zeros(50, dtype=np.int)
    for v in pwordfreq:
        idx = np.argwhere(bins > v)[0][0] - 1
        freq[idx] += 1
        
    ax = fig.add_axes([0,0,1,1])
    bars = ax.bar(lbls, freq, color='g')
    ax.bar_label(bars, label_type="edge")
    
    if yscale: ax.set_yscale('log')
    plt.xticks(rotation=77)
    plt.xlabel("")
    plt.show()
    
def wordFrequencyBarplot(pwordfreq: udt.Dataframe):
    fig, ax = plt.subplots(figsize=(10, 20))
    ax = fig.add_axes([0,0,1,1])
    bars = ax.barh(pwordfreq['word'], pwordfreq['freq'], color='olive')
    ax.bar_label(bars, label_type="edge")
    plt.xticks(rotation=0)
    plt.xlabel("")
    plt.show()
    
def intersectComplementWords(previews: udt.Dataframe):
    word_set = set((" ".join(previews['normalize_comment'])).split(" "))
    word_dct = {word: [0, 0] for word in word_set}
    inter_words = {}
    negative_words = {}
    positive_words = {}
    
    for sen, lbl in zip(previews['normalize_comment'], previews['label']):
        for word in sen.split(" "):
            word_dct[word][lbl] += 1
            
    for key, elem in word_dct.items():
        if elem[0] == 0:
            positive_words[key] = elem[1]
        elif elem[1] == 0:
            negative_words[key] = elem[0]
        else:
            inter_words[key] = tuple(elem)
    
    return (
        pd.DataFrame(sorted(inter_words.items(), key=lambda x: sum(x[1]), reverse=True), columns=('word', 'freq')),
        pd.DataFrame(sorted(negative_words.items(), key=lambda x: x[1], reverse=True), columns=('word', 'freq')),
        pd.DataFrame(sorted(positive_words.items(), key=lambda x: x[1], reverse=True), columns=('word', 'freq')),
    )
    
def barplotTwoDirections(pwords: udt.Dataframe):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=pwords['word'], y=[-elm[0] for elm in pwords['freq']], name="Negative", marker_color='crimson'))
    fig.add_trace(go.Bar(x=pwords['word'], y=[elm[1] for elm in pwords['freq']], name="Positive", marker_color='cornflowerblue'))

    fig.update_layout(barmode='relative', 
                      xaxis=dict(tickfont=dict(size=11, color='black')))
    fig.show()
    
def ftVectorization(previews: udt.Dataframe, ft):
    inter_words, neg_words, pos_words = intersectComplementWords(previews)
    embeded_vectors = []
    
    for i, words_df in enumerate([neg_words, pos_words, inter_words]):
        for index, row in words_df.iterrows():
            w2v = ft.get_word_vector(row['word'])
            freq = sum(row['freq']) if i == 2 else row['freq']
            embeded_vectors.append((row['word'], freq, w2v, i))
            
    return pd.DataFrame(embeded_vectors, columns=['word', 'freq', 'ft_vec', 'label'])
        
def tsneGetNCompenent(pembeded_comment, n=2):
    cmps_reshape = pd.DataFrame(index=list(pembeded_comment.index), columns=np.arange(100))
    for index in range(len(pembeded_comment)):
        cmps_reshape.loc[index] = pembeded_comment.loc[index, 'ft_vec']
    tsne_cmts = TSNE(n_components=n).fit_transform(cmps_reshape)
    tsne_res = pembeded_comment.copy()
    
    for i in range(n):
        tsne_res[f'component_{i + 1}'] = tsne_cmts[:, i]
        
    return tsne_res

def wordScatterPlot(tsne_df):
    df = tsne_df.copy()
    df['label'] = tsne_df['label'].apply(lambda x: "negative" if x < 1 else "positive" if x < 2 else "overlap")
    
    fig = px.scatter(
        df,
        x="component_1",
        y="component_2",
        hover_name="word",
        text="word",
        size="freq",
        color="label",
        size_max=45,
        template="plotly_white",
        title="Bigram similarity and frequency",
        labels={"words": "Avg. Length<BR>(word)"},
        color_continuous_scale=px.colors.sequential.Sunsetdark,
    )
    fig.update_traces(marker=dict(line=dict(width=1, color="Gray")))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(width=1200, height=500)
    fig.show()
    
def word3dPlot(tsne_df):
    df = tsne_df.copy()
    df['label_name'] = tsne_df['label'].apply(lambda x: "negative" if x < 1 else "positive" if x < 2 else "overlap")
    df['freq'] = np.log(df['freq'])*10

    fig1 = go.Scatter3d(x=df.loc[df['label'] == 0, 'component_1'],
                        y=df.loc[df['label'] == 0, 'component_2'],
                        z=df.loc[df['label'] == 0, 'component_3'],
                        marker=dict(size=df.loc[df['label'] == 0, 'freq'],
                                    color=0,
                                    opacity=0.9,
                                    reversescale=True),
                        line=dict(width=0.02), name="Negative",
                        mode='markers', text=df.loc[df['label'] == 0, 'word'])
    fig2 = go.Scatter3d(x=df.loc[df['label'] == 1, 'component_1'],
                        y=df.loc[df['label'] == 1, 'component_2'],
                        z=df.loc[df['label'] == 1, 'component_3'],
                        marker=dict(size=df.loc[df['label'] == 1, 'freq'],
                                    color=1,
                                    opacity=0.9,
                                    reversescale=True),
                        line=dict(width=0.02), name="Positive",
                        mode='markers', text=df.loc[df['label'] == 1, 'word'])
    
    fig3 = go.Scatter3d(x=df.loc[df['label'] == 2, 'component_1'],
                        y=df.loc[df['label'] == 2, 'component_2'],
                        z=df.loc[df['label'] == 2, 'component_3'],
                        marker=dict(size=df.loc[df['label'] == 2, 'freq'],
                                    color=1,
                                    opacity=0.9,
                                    reversescale=True),
                        line=dict(width=0.02), name="Overlap",
                        mode='markers', text=df.loc[df['label'] == 2, 'word'])

    mylayout = go.Layout(scene=dict(xaxis=dict(title="component_1"),
                                    yaxis=dict(title="component_2"),
                                    zaxis=dict(title="component_3")), showlegend=True,)

    plotly.offline.plot({"data": [fig1, fig2, fig3],
                        "layout": mylayout},
                        auto_open=True, 
                        filename=("3D_Plot.html"))
    