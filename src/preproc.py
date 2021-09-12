import pandas as pd
import matplotlib.pyplot as plt
import cufflinks as cf
import plotly
import plotly.express as px
import seaborn as sns
import gensim, os, re, itertools, nltk, snowballstemmer
import plotly.graph_objs as go
import plotly.offline as py

from string import ascii_lowercase
from nltk.corpus import stopwords
from collections import Counter
from IPython.core.display import HTML
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from pandas import DataFrame
from collections import OrderedDict 
from colorama import Fore, Back, Style
from sklearn.feature_extraction.text import TfidfVectorizer
from yellowbrick.text import TSNEVisualizer
from yellowbrick.datasets import load_hobbies
import numpy as np
import warnings
import transformers
import sklearn.metrics as metrics
import itertools, snowballstemmer
import collections as cll
import keras.backend as K
import tensorflow as tf
import tensorflow_addons as tfa
warnings.filterwarnings('ignore')
from string import ascii_lowercase
from numpy import mean, isnan, asarray, polyfit
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GRU, Activation, LSTM, Bidirectional, RNN, Conv1D, MaxPool1D, SpatialDropout1D
from tensorflow.keras import models, layers, preprocessing as kprocessing
from tensorflow.keras import backend as K
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing
from matplotlib import pyplot
from scipy.stats import pearsonr
from keras.layers.normalization import BatchNormalization


#Autor: María Camila Agudelo Castaño

###################################################################################################################################
###################################################################################################################################
#LECTURA Y TRATAMIENTO BD: Descomentar la bd que se quiera usar y comentar la complementaria
###################################################################################################################################
###################################################################################################################################

#BD1

#Cambiar ruta a donde tengamos el archivo csv
db = pd.read_csv(".../news_articles.csv")

#Eliminamos filas Nan
db.dropna(axis=0, how="any", thresh=None, subset=['text'], inplace=True)
db = db.fillna(' ')

#Juntamos título y texto
db["text"] = db["title"] + " " + db["text"]

#Cambio de etiquetas a int
for i, row_value in db["label"].iteritems():
    if db["label"][i] == 'Real':
        db["label"][i] = 1
    else:
        db["label"][i] = 0

db["label"] = db["label"].values
y = db["label"].values

#---------------------------------------#
'''
#BD2

#Cambiar ruta a donde tengamos el archivo csv
fake = pd.read_csv('.../Fake.csv', delimiter = ',')
true = pd.read_csv('.../True.csv', delimiter = ',')

fake['label']= 0
true['label']= 1

db =pd.DataFrame()
db = true.append(fake)

#Eliminamos filas Nan
db.dropna(axis=0, how="any", thresh=None, subset=['text'], inplace=True)
db = db.fillna(' ')

#Juntamos título y texto
db["text"] = db["title"] + " " + db["text"]
y = db["label"].values
'''
#---------------------------------------#

###################################################################################################################################
###################################################################################################################################
#GRÁFICAS
###################################################################################################################################
###################################################################################################################################

db = db.dropna()
#Conteo por campos de la bd
print(db.count())

print(" ")
print(" ")

###################################################################################################################################################################################################

#Valores
print(" ")
print(" ")
print(db['label'].value_counts())

#Gráfica
db['label'].value_counts().plot.pie(figsize = (8,8), startangle = 75, autopct="%0.1f %%")
plt.title('Label of articles', fontsize = 20)
plt.axis("equal")
plt.show()

#Histograma
db['label'].value_counts().plot.bar(figsize = (8,8), rot = 0)
plt.title('Label of articles', fontsize = 20)
plt.show()

###################################################################################################################################################################################################

#Eliminamos stopwords, signos de puntuación y caracteres simples
minimum_count = 5
stemmer = snowballstemmer.EnglishStemmer()

def utils_preprocess_text(text, flg_stemm = False, flg_lemm = True, lst_stopwords = None):
    #Lowercarse y puntuación
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
    #Tokenización
    lst_text = text.split()
    #Borrado de stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in 
                    lst_stopwords]
                
    #Stemming
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
    
    str_frequencies = pd.DataFrame(list(cll.Counter(filter(None,list(itertools.chain(*text.split(' '))))).items()),columns=['word','count'])
    low_frequency_words = set(str_frequencies[str_frequencies['count'] < minimum_count]['word'])
    text = [' '.join(filter(None,filter(lambda word: word not in low_frequency_words, line))) for line in text.split(' ')]
    text = [" ".join(stemmer.stemWords(re.sub('[!"#%\'()*+,-./:;<=>?@\[\]^_`{|}~1234567890’”“′‘\\\]',' ', next_text).split(' '))) for next_text in text]
    

    #Lemmatisation 
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]
            
    text = " ".join(lst_text)
    return text

stop_words = nltk.corpus.stopwords.words("english")
stop_words.extend(['may','also','zero','one','two','three','four','five','six','seven','eight','nine','ten','across','among','beside','however','yet','within']+list(ascii_lowercase))
stoplist = stemmer.stemWords(stop_words)
stoplist = set(stoplist)
stop_words = set(sorted(stop_words + list(stoplist))) 

db["text"].replace('[!"#%\'()*+,-./:;<=>?@\[\]^_`{|}~1234567890’”“′‘\\\]',' ',inplace=True,regex=True)

db["text_clean"] = db["text"].apply(lambda x: utils_preprocess_text(x, flg_stemm = False, flg_lemm = True, lst_stopwords = stop_words))

###################################################################################################################################################################################################

#AGRUPACIONES

print("Media de palabras por etiqueta:")
#Palabras en el texto por categoría:
db['totalwords'] = db['text_clean'].str.split().str.len()
print(db.groupby(['label'])['totalwords'].mean())
print(" ")
print(" ")
print(" ")
print(" ")

#Frases en el texto por categoría:
db['totalsentences'] = db['text_clean'].str.count("\n") + db['text'].str.count("  ") 
print("Media de frases por etiqueta:")
print(db.groupby(['label'])['totalsentences'].mean())

###################################################################################################################################################################################################

db['totalwords'].hist(bins = int(180/5))
plt.tight_layout()
plt.title('Distribución de las palabras en los textos', fontsize = 20)
plt.show()