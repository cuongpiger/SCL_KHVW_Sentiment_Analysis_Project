import pickle
import pandas as pd
import numpy as np
import emojis
import unicodedata
import modules.processor as Processor
import modules.utils as Utils
import enchant
import plotly.graph_objects as go

def loadByPickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def convertToNFX(series, type: str):
    return series.apply(lambda x: unicodedata.normalize(type, x))

def expandEmojisDecode(pcomment: str):
    expand_emojis = ''
    for e in emojis.get(pcomment):
        amount = pcomment.count(e)
        expand_emojis += (f"{emojis.decode(e)[1:-1]} ")*amount
        
    return expand_emojis.strip()  
    
def accBarChart(pmodels_acc):
    y_values = [round(float(x)*100, 5) for x in pmodels_acc[:, 1]]
    fig = go.Figure(data=[go.Bar(
                x=pmodels_acc[:, 0], y=y_values,
                text=y_values, textposition='auto')])

    fig.show()


class ShopeeSentiment:
    def __init__(self, pemoji_model, pcomment_model) -> (None):
        self.emoji_model = pemoji_model
        self.comment_model = pcomment_model
        self.config = self._loadConfigFiles("./modules/dependencies/abbreviate.txt",
                                            "./modules/dependencies/vocabulary.txt",
                                            "./modules/dependencies/stopwords.txt")
        
    def _loadConfigFiles(self, pabbreviate_path, pvocabulary_path, pstop_words_path):
        return {
            'pabbreviate_dict': Utils.buildDictionaryFromFile(pabbreviate_path),
            'pvocabulary_dict': Utils.buildDictionaryFromFile(pvocabulary_path, True),
            'peng_dict': enchant.Dict("en_US"),
            'pstop_words': Utils.buildListFromFile(pstop_words_path)
        }
        
    def _sterilize(self, pdata: pd.Series, pabbreviate_dict, pvocabulary_dict, peng_dict, pstop_words, **kwargs):
        df = pd.DataFrame()
        df['raw_review'] = pdata
        df['review'] = convertToNFX(pdata, 'NFD').str.strip().str.lower()
        df['emoji'] = df['review'].apply(lambda cmt: expandEmojisDecode(cmt))
        df['review'] = df['review'].apply(lambda cmt: Processor.removeSpecialLetters(cmt))
        df['review'] = df['review'].apply(lambda cmt: Processor.replaceWithDictionary(cmt, pabbreviate_dict))
        df['review'] = df['review'].apply(lambda cmt: Processor.removeNoiseWord(cmt, pvocabulary_dict, peng_dict))
        df['review'] = df['review'].apply(lambda cmt: Processor.removeStopwords(cmt, pstop_words))
        df['review'] = convertToNFX(df['review'], 'NFC')
        df['emoji'] = convertToNFX(df['emoji'], 'NFC')

        return df
    
    def _predict(self, pemoji, pcomment):
        if pemoji != -1 and pcomment != -1:
            return ((pemoji[0] + pcomment[0]) / 2.0, (pemoji[1] + pcomment[1]) / 2.0)
        elif pcomment != -1:
            return pcomment
        else:
            return pemoji
        
    def predict(self, pnew_data: pd.Series):
        res = self._sterilize(pnew_data, **self.config)
        res.loc[res['emoji'] != "", 'predicted_emoji'] = self.emoji_model.predict(res.loc[res['emoji'] != "", 'emoji'])['output_proba']
        res.loc[res['review'] != "", 'predicted_comment'] = self.comment_model.predict(res.loc[res['review'] != "", 'review'])['output_proba']
        res = res.fillna(-1)
        res['probability'] = res.apply(lambda x: self._predict(x['predicted_emoji'], x['predicted_comment']), axis=1)
        res['class'] = res['probability'].apply(lambda x: 0 if x[0] > x[1] else 1)
        
        return res[['raw_review', 'probability', 'class']]        
    