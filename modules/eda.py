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
    
    plt.pie([positive_class, negative_class], labels=['Positive', 'Negative'], colors=colors, autopct="%.0f%%")
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
    
    