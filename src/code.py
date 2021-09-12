import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import warnings
import gensim
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


#GLOBALES PARA MÉTRICAS
acc_per_fold = []
prec_per_fold = []
rec_per_fold = []
f1_per_fold = []
kappa_per_fold = []
roc_per_fold = []



#####################################
#                                   #
#                                   #
#       FUNCIONES AUXILIARES        #
#                                   #
#                                   #
#####################################


###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################

def get_lstm():
    lstm_model = Sequential(name = 'lstm_nn_model')
    lstm_model.add(layer = Embedding(input_dim = 1000000, output_dim = 120, name = '1st_layer'))
    lstm_model.add(layer = LSTM(units = 120, dropout = 0.2, recurrent_dropout = 0.2, name = '2nd_layer'))
    lstm_model.add(layer = Dropout(rate = 0.5, name = '3rd_layer'))
    lstm_model.add(layer = Dense(units = 120,  activation = 'relu', name = '4th_layer'))
    lstm_model.add(layer = Dropout(rate = 0.5, name = '5th_layer'))
    lstm_model.add(layer = Dense(units = len(set(y)),  activation = 'sigmoid', name = 'output_layer'))
    lstm_model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    return lstm_model

def get_bi_lstm():
    modelg = Sequential()
    modelg.add(layer = Embedding(input_dim = 1000000, output_dim = 120, name = '1st_layer'))
    modelg.add(Bidirectional(LSTM(64)))
    modelg.add(Dense(256,name='FC1'))
    modelg.add(Activation('relu'))
    modelg.add(Dropout(0.5))
    modelg.add(Dense(units = len(set(y)),name='out_layer', activation='sigmoid'))
    modelg.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return modelg

def get_gru():
    modelg = Sequential()
    modelg.add(layer = Embedding(input_dim = 1000000, output_dim = 120, name = '1st_layer'))
    modelg.add(GRU(100, dropout=0.2, recurrent_dropout=0.2))
    modelg.add(BatchNormalization())
    modelg.add(Dense(2, activation='softmax'))
    modelg.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return modelg

###################################################################################################################################

#MÉTRICAS

def show_confusion_matrix(cm, labels):
  plt.figure(figsize=(10, 8))
  sns.heatmap(cm, xticklabels=labels, yticklabels=labels, 
              annot=True, fmt='g')
  plt.xlabel('Prediction')
  plt.ylabel('Label')
  plt.show()


#Pinta la matriz de confusión para las métricas
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
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
#LECTURA Y TRATAMIENTO BD: Descomentar la bd que se desee probar y comentar la complementaria
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
minimum_count = 5
stemmer = snowballstemmer.EnglishStemmer()

def utils_preprocess_text(text, flg_stemm = False, flg_lemm = True, lst_stopwords = None):
    #Lowercase y eliminación de puntuación y caracteres
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
    #Tokenización
    lst_text = text.split()
    #Eliminar stopwords
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

###################################################################################################################################
###################################################################################################################################
###################################################################################################################################






#####################################
#                                   #
#                                   #
#          VECTORIZACIÓN            #  Descomentar el modelo de vectorización que se desee probar
#                                   #   y comentar los otros
#                                   #
#####################################




###################################################################################################################################
###################################################################################################################################
#Vectorización -- TFIDF
###################################################################################################################################
###################################################################################################################################

#Tf-Idf (advanced variant of BoW)
vectorizer = feature_extraction.text.TfidfVectorizer(max_features = 300, stop_words = 'english',ngram_range = (1,2), max_df = 0.7)

corpus = db["text_clean"]
vectorizer.fit(corpus)
X = vectorizer.transform(corpus)


X = X.toarray()

###################################################################################################################################
###################################################################################################################################
#Vectorización -- Word2Vec
###################################################################################################################################
###################################################################################################################################
'''
EMBEDDING_DIM = 300

corpus = db["text_clean"]

#Unigramas
lst_corpus = []
for string in corpus:
   lst_words = string.split()
   lst_grams = [" ".join(lst_words[i:i+1]) 
               for i in range(0, len(lst_words), 1)]
   lst_corpus.append(lst_grams)

#Bigramas y trigramas
bigrams_detector = gensim.models.phrases.Phrases(lst_corpus, delimiter=" ", min_count=5, threshold=10)
bigrams_detector = gensim.models.phrases.Phraser(bigrams_detector)
trigrams_detector = gensim.models.phrases.Phrases(bigrams_detector[lst_corpus], delimiter=" ", min_count=5, threshold=10)
trigrams_detector = gensim.models.phrases.Phraser(trigrams_detector)


#DESCOMENTAR UNO PARA PROBAR Y LUEGO INTERCAMBIAR
#---------------------------------------#

#Skip-gram
w2v = gensim.models.word2vec.Word2Vec(lst_corpus, vector_size = EMBEDDING_DIM, window = 5, min_count = 1, sg = 1)

#---------------------------------------#

#CBOW
#w2v_model = gensim.models.word2vec.Word2Vec(lst_corpus, vector_size=EMBEDDING_DIM, window=5, min_count=1, sg = 0)

#---------------------------------------#


tokenizer = kprocessing.text.Tokenizer(lower=True, split=' ', oov_token="NaN", filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(lst_corpus)
dic_vocabulary = tokenizer.word_index
lst_text2seq= tokenizer.texts_to_sequences(lst_corpus)
X = kprocessing.sequence.pad_sequences(lst_text2seq, maxlen=15, padding="post", truncating="post")
'''
###################################################################################################################################
###################################################################################################################################
#Vectorización -- BERT
###################################################################################################################################
###################################################################################################################################
'''
tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased', TOKENIZERS_PARALLELISM=False,do_lower_case=True)

corpus = db["text_clean"]
maxlen = 1500
#Añadiendo tokens
maxqnans = np.int((maxlen-20)/2)
corpus_tokenized = ["[CLS] "+" ".join(tokenizer.tokenize(re.sub(r'[^\w\s]+|', '', str(txt).lower().strip()))[:maxqnans])+" [SEP] " for txt in corpus]

#Añadiendo tokens
sentences = ["[CLS] " + query + " [SEP]" for query in corpus_tokenized]

tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased',TOKENIZERS_PARALLELISM=False, do_lower_case=True)
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

#Máximo
MAX_LEN = 700
#Ids
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts], dtype="long", truncating="post", padding="post")
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
X = pad_sequences(input_ids, dtype="long", truncating="post", padding="post")
'''
###################################################################################################################################

#CONVERSIÓN NECESARIA EN EL FLUJO
X = np.asarray(X).astype(np.float32)
y = np.asarray(y).astype(np.float32)

###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################








#####################################
#                                   #
#                                   #
#          CLASIFICACIÓN            #  Descomentar el enfoque que se desee probar 
#                                   #   y comentar los restantes
#                                   #
#####################################




###################################################################################################################################
###################################################################################################################################
#CLASIFICADOR INDIVIDUAL PARA ANÁLISIS PROFUNDO CON K-FOLD DISTINTOS A 10 ... ESTUDIO EXTRA
###################################################################################################################################
###################################################################################################################################
'''
model = naive_bayes.MultinomialNB()                 #!!!!!!!!!!!!!!!CAMBIAR MODELO AQUI!!!!!!!!!!!!!!!!!!

#Evaluación del modelo
def evaluate_model(cv):
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return mean(scores), scores.min(), scores.max()

#Condición ideal
ideal, _, _ = evaluate_model(LeaveOneOut())
print('Ideal: %.3f' % ideal)

#Definición de folds
folds = range(2,31)

means, mins, maxs = list(),list(),list()

#Evaluar valor de k
for k in folds:
    #Condiciones de test
    cv = KFold(n_splits=k, shuffle=True, random_state=1)
    k_mean, k_min, k_max = evaluate_model(cv)
    print('> folds=%d, accuracy=%.3f (%.3f,%.3f)' % (k, k_mean, k_min, k_max))
    #Accuracy
    means.append(k_mean)
    #Min y max relativa a la media
    mins.append(k_mean - k_min)
    maxs.append(k_max - k_mean)

#Plot de barras max/min
pyplot.errorbar(folds, means, yerr=[mins, maxs], fmt='o')
#Pintamos ideal
pyplot.plot(folds, [ideal for _ in range(len(folds))], color='r')
pyplot.show()
'''
###################################################################################################################################
###################################################################################################################################
#CLASIFICADORES COMPARADOS ENTRE SI CON 10-FOLD ... PARA COMPARACIÓN RÁPIDA: Estudio extra.
###################################################################################################################################
###################################################################################################################################
'''
#Lista de modelos a evaluar
def get_models():
    models = list()
    models.append(MultinomialNB())
    models.append(LogisticRegression())
    models.append(RidgeClassifier())
    models.append(SGDClassifier())
    models.append(PassiveAggressiveClassifier())
    models.append(KNeighborsClassifier(n_neighbors = 10,weights = 'distance',algorithm = 'brute'))
    models.append(DecisionTreeClassifier(criterion= 'entropy', max_depth = 20, splitter='best', random_state=42))
    models.append(ExtraTreeClassifier())
    models.append(LinearSVC(max_iter=100)) #Buscar como quitar warning
    models.append(SVC())
    models.append(GaussianNB())
    models.append(AdaBoostClassifier())
    models.append(BaggingClassifier())
    models.append(RandomForestClassifier(n_estimators=150, max_depth=None, n_jobs=-1))
    models.append(GradientBoostingClassifier())
    models.append(LinearDiscriminantAnalysis())
    models.append(QuadraticDiscriminantAnalysis())
    models.append(BernoulliNB())
    return models
 
#Evaluación del modelo
def evaluate_model(cv, model):
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return mean(scores)
 
#Condicion ideal
ideal_cv = LeaveOneOut()

#Ten-fold
cv = KFold(n_splits=10, shuffle=True, random_state=1)

models = get_models()

ideal_results, cv_results = list(), list()

for model in models:
    cv_mean = evaluate_model(cv, model)
    ideal_mean = evaluate_model(ideal_cv, model)
    #Miramos si algún resultado no es válido
    if isnan(cv_mean) or isnan(ideal_mean):
        continue
    cv_results.append(cv_mean)
    ideal_results.append(ideal_mean)
    print('>%s: ideal=%.3f, cv=%.3f' % (type(model).__name__, ideal_mean, cv_mean))

#Correlación entre cada prueba
corr, _ = pearsonr(cv_results, ideal_results)
print('Correlation: %.3f' % corr)
pyplot.scatter(cv_results, ideal_results)
coeff, bias = polyfit(cv_results, ideal_results, 1)
line = coeff * asarray(cv_results) + bias
pyplot.plot(cv_results, line, color='r')
pyplot.title('10-fold CV vs LOOCV Mean Accuracy')
pyplot.xlabel('Mean Accuracy (10-fold CV)')
pyplot.ylabel('Mean Accuracy (LOOCV)')
pyplot.show()
'''
###################################################################################################################################
###################################################################################################################################
#CLASIFICADORES INDIVIDUALES DETALLADOS POR FOLD
###################################################################################################################################
###################################################################################################################################

num_folds = 1
fold_no = 1

#K-Foldd cross validator
kfold = KFold(n_splits=num_folds, shuffle=True)

for train, test in kfold.split(X, y):
    model = MultinomialNB() #!!!!!!!!!CAMBIAR MODELO AQUI!!!!!!!!!!!!!!!!!! Ir cambiandolo para evaluar cada modelo (la lista de modelos está en el apartado anterior get_models())

    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    #Entrenamiento
    history = model.fit(X[train], y[train])

    #MÉTRICAS:
    pred = model.predict(X[test])

    cm = confusion_matrix(y[test],pred)
    print(cm)
    plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

    metrics(y[test],pred)

    fold_no = fold_no + 1

metrics_avg(acc_per_fold, prec_per_fold, rec_per_fold, f1_per_fold, kappa_per_fold, roc_per_fold)

###################################################################################################################################
###################################################################################################################################
#CLASIFICADORES KERAS USANDO 10-FOLD
###################################################################################################################################
###################################################################################################################################
'''
#Configuración del modelo
batch_size = 50
num_folds = 10
no_epochs = 10
fold_no = 1

#K-Foldd cross validator
kfold = KFold(n_splits=num_folds, shuffle=True)

for train, test in kfold.split(X, y):
    model = get_lstm() #!!!!!!!!!CAMBIAR MODELO AQUI!!!!!!!!!!!!!!!!!! Elegir modelo entre get_lstm(), get_bi_lstm() y get_gru()

    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    #Entrenamiento
    history = model.fit(X[train], y[train], batch_size=batch_size, epochs=no_epochs, verbose=1)

    pred = model.predict(X[test])
    classes = np.argmax(pred, axis = 1)

    cm = confusion_matrix(y[test],classes)
    print(cm)
    #show_confusion_matrix(conf_matrix, ['FAKE', 'REAL'])
    metrics(y[test],classes)

    fold_no = fold_no + 1

metrics_avg(acc_per_fold, prec_per_fold, rec_per_fold, f1_per_fold, kappa_per_fold, roc_per_fold)
'''