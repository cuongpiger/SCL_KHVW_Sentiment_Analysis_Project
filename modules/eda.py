import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import modules.user_object_defined as udt


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