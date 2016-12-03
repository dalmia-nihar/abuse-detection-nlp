import tweepy
import pickle
import sys
import pandas as pd
import nltk, re, string
import numpy as np
from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import preprocessing
from sklearn import svm
from nltk.corpus import stopwords

from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
import itertools
# get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_validation import StratifiedKFold
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re, collections
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import words as w

from nltk.corpus import *
from nltk.collocations import *
from string import punctuation
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer


CONSUMER_KEY = "STFpndP02iqX61Q02gSwWSeju"
CONSUMER_SECRET = "bIzoT7jRXxP0jHYK7U2l6BWCZzqz3vOfWrO7WA8EdkOkbY9K7P"
ACCESS_TOKEN = "797174826007834625-hr4WaBsASLneOpHv7vjb2TidFNljkKr"
ACCESS_TOKEN_SECRET = "lTQEGUk5j7gSn7egSoWdSi8i0Fy2PwcjYiWH1MmHIaO0R"


# In[5]:

def build_tokens(text):
    tweetTokenizer = nltk.tokenize.TweetTokenizer(strip_handles=True, reduce_len=True)
    tokens = tweetTokenizer.tokenize(text)
    #tokens = preprocess(text, lowercase=True)
    #tokens = [nltk.WordNetLemmatizer().lemmatize(token) for token in tokens]
    #tokens= [nltk.PorterStemmer().stem(token) for token in tokens]
    return tokens


# In[6]:

class AdditionalFeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def get_feature_names(self):
        return (['percent_bad','vader_compound','num_words','vader_neg','vader_pos'])
    
    def num_bad(self, df):
        #get number of words in each sentence
        num_words = [len(word) for word in df]
        
        #get percent of abusive words in each sentence
        with open("list_of_abuses.txt", "r") as abuse_list:
            abuses = abuse_list.read().split()
            num_abuses = 0
            for abuse in abuses:
                num_abuses += 1
            # number of badwords in list of abuses
            num_bad = [np.sum([word.lower().count(abuse) for abuse in abuses])
                                                for word in df]
            norm_bad = np.array(num_bad) / np.array(num_words, dtype=np.float)
        return norm_bad
    
    def num_words(self,df):
        #get number of words in each sentence
        num_words = [len(word) for word in df]
        return num_words
    
    def vader_helper(self, df):
        #vader analysis
        sid = SentimentIntensityAnalyzer()
        vader_feature = []
        for sentence in df:
            ss = sid.polarity_scores(sentence)
            vader_feature.append(ss['compound'])
        return vader_feature
    
    def vader_helper_neg(self, df):
        #vader analysis
        sid = SentimentIntensityAnalyzer()
        vader_feature = []
        for sentence in df:
            ss = sid.polarity_scores(sentence)
            vader_feature.append(ss['neg'])
        return vader_feature
    
    def vader_helper_pos(self, df):
        #vader analysis
        sid = SentimentIntensityAnalyzer()
        vader_feature = []
        for sentence in df:
            ss = sid.polarity_scores(sentence)
            vader_feature.append(ss['pos'])
        return vader_feature
    def transform(self, df, y=None):     
        #add both the features to an array
        X = np.array([self.num_bad(df), self.vader_helper(df),self.num_words(df),self.vader_helper_neg(df),self.vader_helper_pos(df)]).T
        #X = np.array([self.num_bad(df),self.vader_helper(df)]).T
        #X.reshape(-1, 1) #use if only 1 feature
        if not hasattr(self, 'scalar'):
            self.scalar = preprocessing.StandardScaler().fit(X)
        return self.scalar.transform(X)      

    def fit(self, df, y=None):
        return self


# In[7]:

stopwords = ['a','an','the','or','is','are','was','were','have']
def all_features():
    features = []
    custom_features = AdditionalFeatureExtractor() # this class includes my custom features 
    vect = TfidfVectorizer(sublinear_tf=True, ngram_range=(1,6), analyzer= "char", stop_words = stopwords, tokenizer= build_tokens)
    vect1 = TfidfVectorizer(sublinear_tf=True, ngram_range=(1,6), analyzer= "word", stop_words = stopwords, tokenizer= build_tokens)
    
    features.append(('ngram', vect))
    features.append(('ngram1', vect1))
   
    features.append(('custom_features', custom_features))
    return features


    

# returns confidence score ()
def is_abuse(sentence):
    model = pickle.load(open('model.pkl','rb'))
    confidence = model.predict_proba([sentence])
    print(confidence)
    return confidence[0][0]




def init_twitter_api():
    '''
    Initializes the tweepy twitter API
    '''
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    return tweepy.API(auth)

def retrieve_model():
    '''
    Retrieves machine learning model from pickle
    '''
    return True
