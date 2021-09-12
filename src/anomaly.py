import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import warnings
import gensim
import itertools
warnings.filterwarnings('ignore')

from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing
from tensorflow.keras import models, layers, preprocessing as kprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix


#Autor: María Camila Agudelo Castaño


#VARIABLES GLOBALES PARA MÉTRICAS
acc_per_fold = []
prec_per_fold = []
rec_per_fold = []
f1_per_fold = []
kappa_per_fold = []
roc_per_fold = []


###################################################################################################################################
###################################################################################################################################
#MÉTRICAS
###################################################################################################################################
###################################################################################################################################

def show_confusion_matrix(cm, labels):
  plt.figure(figsize=(10, 8))
  sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True, fmt='g')
  plt.xlabel('Prediction')
  plt.ylabel('Label')
  plt.show()

def plot_confusion_matrix(cm, classes,title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def metrics(c1,c2):
    global acc_per_fold, prec_per_fold, rec_per_fold, f1_per_fold, kappa_per_fold, roc_per_fold
    scores = accuracy_score(c1,c2)
    print(f'Score for fold {fold_no}: {scores} - {scores*100}%')
    acc_per_fold.append(scores*100)

    precision = precision_score(c1,c2)
    print(f'Precision for fold {fold_no}: {precision} - {precision*100}%')
    prec_per_fold.append(precision*100)
 
    recall = recall_score(c1,c2)
    print(f'Recall for fold {fold_no}: {recall} - {recall*100}%')
    rec_per_fold.append(recall*100)
  
    f1 = f1_score(c1,c2)
    print(f'F1-score for fold {fold_no}: {f1} - {f1*100}%')
    f1_per_fold.append(f1*100)

    kappa = cohen_kappa_score(c1,c2)
    print(f'Kappa for fold {fold_no}: {kappa} - {kappa*100}%')
    kappa_per_fold.append(kappa*100)

    auc = roc_auc_score(c1,c2)
    print(f'ROC score for fold {fold_no}: {auc} - {auc*100}%')
    roc_per_fold.append(auc*100)

def metrics_avg(c1,c2,c3,c4,c5,c6):
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(c1)):
      print('------------------------------------------------------------------------')
      print(f'> Fold {i+1} - Accuracy: {c1[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(c1)} (+- {np.std(c1)})')
    print('------------------------------------------------------------------------')

    print('------------------------------------------------------------------------')
    print('Precision per fold')
    for i in range(0, len(c2)):
      print('------------------------------------------------------------------------')
      print(f'> Fold {i+1} - Precision: {c2[i]}%')
    print('------------------------------------------------------------------------')
    print('Average precision for all folds:')
    print(f'> Precision: {np.mean(c2)} (+- {np.std(c2)})')
    print('------------------------------------------------------------------------')

    print('------------------------------------------------------------------------')
    print('Recall per fold')
    for i in range(0, len(c3)):
      print('------------------------------------------------------------------------')
      print(f'> Fold {i+1} - Recall: {c3[i]}%')
    print('------------------------------------------------------------------------')
    print('Average recalls for all folds:')
    print(f'> Recall: {np.mean(c3)} (+- {np.std(c3)})')
    print('------------------------------------------------------------------------')

    print('------------------------------------------------------------------------')
    print('F1-score per fold')
    for i in range(0, len(c4)):
      print('------------------------------------------------------------------------')
      print(f'> Fold {i+1} - F1-score: {c4[i]}%')
    print('------------------------------------------------------------------------')
    print('Average f1-scores for all folds:')
    print(f'> F1-score: {np.mean(c4)} (+- {np.std(c4)})')
    print('------------------------------------------------------------------------')

    print('------------------------------------------------------------------------')
    print('Kappa per fold')
    for i in range(0, len(c5)):
      print('------------------------------------------------------------------------')
      print(f'> Fold {i+1} - Kappa: {c5[i]}%')
    print('------------------------------------------------------------------------')
    print('Average kappa for all folds:')
    print(f'> Kappa: {np.mean(c5)} (+- {np.std(c5)})')
    print('------------------------------------------------------------------------')

    print('------------------------------------------------------------------------')
    print('ROC per fold')
    for i in range(0, len(c6)):
      print('------------------------------------------------------------------------')
      print(f'> Fold {i+1} - ROC: {c6[i]}%')
    print('------------------------------------------------------------------------')
    print('Average ROC for all folds:')
    print(f'> ROC: {np.mean(c6)} (+- {np.std(c6)})')
    print('------------------------------------------------------------------------')

###################################################################################################################################
###################################################################################################################################
#TRATAMIENTO BD: Descomentar BD que se desee probar y comentar la complementaria
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
#PREPROCESAMIENTO
###################################################################################################################################
###################################################################################################################################

#Eliminamos stopwords, signos de puntuación y caracteres simples

def utils_preprocess_text(text, flg_stemm = False, flg_lemm = True, lst_stopwords = None):
    #Lowercase y puntuación
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
    #Tokenizar
    lst_text = text.split()
    #Eliminamos stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in 
                    lst_stopwords]
                
    #Stemming
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
                
    #Lemmatisation
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]
            
    text = " ".join(lst_text)
    return text

stop_words = set(nltk.corpus.stopwords.words("english"))

db["text_clean"] = db["text"].apply(lambda x: utils_preprocess_text(x, flg_stemm = False, flg_lemm = True, lst_stopwords = stop_words))

###################################################################################################################################
###################################################################################################################################
#SPLIT DATA: proporción 80% textos reales - 20% textos falsos
###################################################################################################################################
###################################################################################################################################

truebd = db[db["label"] == 1]
falsebd = db[db["label"] == 0]

length_truebd = len(truebd)
l = int((length_truebd*80)/100)
length_falsebd = int((20 * l) / 80)

t = truebd.sample(l)
f = falsebd.sample(length_falsebd)

t2 = truebd.sample(l)
f2 = falsebd.sample(length_falsebd)

falseandtrue2 = t2.append(f2)

###################################################################################################################################
###################################################################################################################################
#VECTORIZACIÓN
###################################################################################################################################
###################################################################################################################################

#Vectorización -- Word2Vec -- TRAIN

corpus = t["text_clean"]

lst_corpus = []
for string in corpus:
   lst_words = string.split()
   lst_grams = [" ".join(lst_words[i:i+1]) 
               for i in range(0, len(lst_words), 1)]
   lst_corpus.append(lst_grams)

bigrams_detector = gensim.models.phrases.Phrases(lst_corpus, delimiter=" ", min_count=5, threshold=10)
bigrams_detector = gensim.models.phrases.Phraser(bigrams_detector)
trigrams_detector = gensim.models.phrases.Phrases(bigrams_detector[lst_corpus], delimiter=" ", min_count=5, threshold=10)
trigrams_detector = gensim.models.phrases.Phraser(trigrams_detector)

#Dimensión de los vectores a generar
EMBEDDING_DIM = 300
#Creación de vectores
w2v = gensim.models.word2vec.Word2Vec(lst_corpus, vector_size = EMBEDDING_DIM, window = 7, min_count = 1, sg = 1)

#Tokenización
tokenizer = kprocessing.text.Tokenizer(lower=True, split=' ', oov_token="NaN", filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(lst_corpus)
dic_vocabulary = tokenizer.word_index
lst_text2seq= tokenizer.texts_to_sequences(lst_corpus)
Xt = kprocessing.sequence.pad_sequences(lst_text2seq, maxlen=15, padding="post", truncating="post")
yt = t["label"].values

Xt = np.asarray(Xt).astype(np.float32)
yt = np.asarray(yt).astype(np.float32)

corpus = f["text_clean"]

lst_corpus = []
for string in corpus:
   lst_words = string.split()
   lst_grams = [" ".join(lst_words[i:i+1]) 
               for i in range(0, len(lst_words), 1)]
   lst_corpus.append(lst_grams)

bigrams_detector = gensim.models.phrases.Phrases(lst_corpus, delimiter=" ", min_count=5, threshold=10)
bigrams_detector = gensim.models.phrases.Phraser(bigrams_detector)
trigrams_detector = gensim.models.phrases.Phrases(bigrams_detector[lst_corpus], delimiter=" ", min_count=5, threshold=10)
trigrams_detector = gensim.models.phrases.Phraser(trigrams_detector)

#Dimensión de los vectores a generar
EMBEDDING_DIM = 300
#Creación de vectores
w2v = gensim.models.word2vec.Word2Vec(lst_corpus, vector_size = EMBEDDING_DIM, window = 7, min_count = 1, sg = 1)

#Tokenización
tokenizer = kprocessing.text.Tokenizer(lower=True, split=' ', oov_token="NaN", filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(lst_corpus)
dic_vocabulary = tokenizer.word_index
lst_text2seq= tokenizer.texts_to_sequences(lst_corpus)
Xf = kprocessing.sequence.pad_sequences(lst_text2seq, maxlen=15, padding="post", truncating="post")
yf = f["label"].values

Xf = np.asarray(Xf).astype(np.float32)
yf = np.asarray(yf).astype(np.float32)


#Vectorización -- Word2Vec -- TEST

corpus = falseandtrue2["text_clean"]

lst_corpus = []
for string in corpus:
   lst_words = string.split()
   lst_grams = [" ".join(lst_words[i:i+1]) 
               for i in range(0, len(lst_words), 1)]
   lst_corpus.append(lst_grams)

bigrams_detector = gensim.models.phrases.Phrases(lst_corpus, delimiter=" ", min_count=5, threshold=10)
bigrams_detector = gensim.models.phrases.Phraser(bigrams_detector)
trigrams_detector = gensim.models.phrases.Phrases(bigrams_detector[lst_corpus], delimiter=" ", min_count=5, threshold=10)
trigrams_detector = gensim.models.phrases.Phraser(trigrams_detector)

#Dimensión de los vectores a generar
EMBEDDING_DIM = 300
#Creación de vectores
w2v = gensim.models.word2vec.Word2Vec(lst_corpus, vector_size = EMBEDDING_DIM, window = 7, min_count = 1, sg = 1)

#Tokenización
tokenizer = kprocessing.text.Tokenizer(lower=True, split=' ', oov_token="NaN", filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(lst_corpus)
dic_vocabulary = tokenizer.word_index
lst_text2seq= tokenizer.texts_to_sequences(lst_corpus)

X2 = kprocessing.sequence.pad_sequences(lst_text2seq, maxlen=15, padding="post", truncating="post")
y2 = falseandtrue2["label"].values

X2 = np.asarray(X2).astype(np.float32)
y2 = np.asarray(y2).astype(np.float32)

###################################################################################################################################
###################################################################################################################################
#K fold proporcional con Isolation Forest, OneClassSVM: descomentar el modelo que se desee probar
###################################################################################################################################
###################################################################################################################################

num_folds = 10
t1 = []
t2 = []
t3 = []
t4 = []

#K-Foldd cross validator
kfold = KFold(n_splits=num_folds, shuffle=True)

for xa in kfold.split(Xf):
    t1.append(xa)
for xb in kfold.split(Xt):
    t2.append(xb)
for xc, xd in kfold.split(X2, y2):
    t3.append(X2[xc])
    t4.append(y2[xc])

t1 = list(t1)
t2 = list(t2)
t3 = list(t3)
t4 = list(t4)

#Definición y entrenamiento del modelo, descomentar el que se desee probar y comentar el otro
#model = IsolationForest(n_estimators  = 300,max_samples ='auto',contamination = 0.2 ,bootstrap=False, n_jobs=-1 ,random_state=777 ,verbose=0)
model = OneClassSVM(nu=0.2)

for i in range(num_folds):
    var = np.append(t1[i][0], t2[i][0])
    fold_no = i+1
    print(f'Fold número: ', i+1)
    var2 = t3[i]

    var = var.reshape(1, -1)

    #Entrenamiento
    model.fit(var)

    #MATRIZ 
    ypred = model.fit_predict(var2)
    ypred[ypred == 1] = 1
    ypred[ypred == -1] = 0

    cm = confusion_matrix(t4[i], ypred)
    print(cm)
    show_confusion_matrix(cm, ['FAKE', 'REAL'])
    metrics(t4[i],ypred)

metrics_avg(acc_per_fold, prec_per_fold, rec_per_fold, f1_per_fold, kappa_per_fold, roc_per_fold)

###################################################################################################################################
###################################################################################################################################
#K fold proporcional con LOF
###################################################################################################################################
###################################################################################################################################

num_folds = 10

t1 = []
t2 = []
t3 = []
t4 = []

#K-Foldd cross validator
kfold = KFold(n_splits=num_folds, shuffle=True)

for xa in kfold.split(Xf):
    t1.append(xa)
for xb in kfold.split(Xt):
    t2.append(xb)
for xc, xd in kfold.split(X2, y2):
    t3.append(X2[xc])
    t4.append(y2[xc])

t1 = list(t1)
t2 = list(t2)
t3 = list(t3)
t4 = list(t4)

#Definición y entrenamiento del modelo
modelo_isof = LocalOutlierFactor(n_neighbors = 1, contamination = 0.2)

for i in range(num_folds):
    var = np.append(t1[i][0], t2[i][0])
    fold_no = i+1
    print(f'Fold número: ', i+1)
    var2 = t3[i]

    var = var.reshape(-1, 1)

    #Entrenamiento
    modelo_isof.fit_predict(var)

    #MATRIZ 
    ypred = modelo_isof.fit_predict(var2)
    ypred[ypred == 1] = 1
    ypred[ypred == -1] = 0

    cm = confusion_matrix(t4[i], ypred)
    print(cm)
    show_confusion_matrix(cm, ['FAKE', 'REAL'])
    metrics(t4[i],ypred)

metrics_avg(acc_per_fold, prec_per_fold, rec_per_fold, f1_per_fold, kappa_per_fold, roc_per_fold)

###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################