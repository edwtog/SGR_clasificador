# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 12:25:58 2017

@author: etorres
"""

import gensim
import numpy as np
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, precision_score

import matplotlib.pyplot as plt
from itertools import cycle
from time import time

def remove_shortW(words):
    return [word for word in words if len(word) > 2]

def remove_num(words):
    return [word for word in words if not word.isnumeric()]

def remove_noChar(words):
    return [re.sub(ur"[^a-zA-ZñÑáéíóúÁÉÍÓÚ ]","", word) for word in words]

def remove_sw(words,sw_list):
    return [word for word in words if word not in sw_list]

def read_list(fname):
    for i, line in enumerate(fname):
        yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

def join_text(words):
    return ' '.join(words)

def text2cat(values_list):
    numNomVar = {}
    i=0
    nomVarList = list(set(values_list))
    nomVarList = sorted(nomVarList)
    for item in nomVarList:
        numNomVar[item] = i
        i += 1
    return numNomVar

def plot_dr(X, labels, plot_title = None, i = None):
    n_clusters_ = len(np.unique(labels))
    if n_clusters_ <= 3:
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    else:
        colors = plt.cm.jet(np.linspace(0, 1, n_clusters_))
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        plt.scatter(X[my_members, 0], X[my_members, 1], color=col, s=0.1)
    plt.tick_params(axis='both', bottom='off', top='off', labelbottom='off',
                    labelsize = 3, pad = 2)
    plt.title('%d)' % i + plot_title + '#c:%d' % n_clusters_, fontsize=3)
    #plt.savefig('results.png', dpi=1200)
    return

###############################################################################
####  Extraccion datos de texto, categoricas y numericas
###############################################################################

## Leer archivo de datos (pre-Procesado)
#C:/Users/etorres/Documents/REG Manuales/IGPR/etiquetas
dirFile = 'C:/Users/etorres/Documents/REG Manuales/IGPR/etiquetas/'
df = pd.read_csv(dirFile+'df_OCAD_TEXT_INDICA.txt', sep='|', encoding='utf-8')
df.dropna(subset=['texto_unificado'], inplace=True)

### Data exploration
## Remove outlier from VALORTOTAL
filt_df = df.loc[:,['VALORTOTAL']]
lowPerc = .00001
highPerc = .99999
quant_df = filt_df.quantile([lowPerc, highPerc])

filt_df = filt_df.apply(lambda x: x[(x < quant_df.loc[highPerc,x.name])], axis=0)

df_proc = pd.concat([df.loc[:,df.columns != 'VALORTOTAL'], filt_df], axis=1)
df_proc.dropna(subset=['VALORTOTAL'], inplace=True)

## varibale indicadora de critico
df_proc['criticoSGRP58'].fillna(2, inplace=True)
df_proc['criticoSGRPProm72'].fillna(2, inplace=True)
df_proc['critico_visitas'].fillna(2, inplace=True)

## Columnas de variables a utilizar
cols_to_keep = ['NOMBREOCAD', 'TIPOOCAD', 'REGION','DEPARTAMENTO',
                'RECURSOSPERTENECIENTESA', 'ENTIDADEJECUTORA', 'TIPODEEJECUTOR',
                'SECTOR','SUBSECTOR', 'NOMBREDELPROYECTO', 'ESTADO', 'VALORSGR',
                u'VALORNACIÓN', 'VALOROTROS','VALORTOTAL', 'texto_unificado',
                'criticoSGRP58', 'criticoSGRPProm72','critico_visitas']

## Columnas de variables categoricas
cols_categorical = ['NOMBREOCAD', 'TIPOOCAD','REGION','DEPARTAMENTO',
                    'RECURSOSPERTENECIENTESA','ENTIDADEJECUTORA','TIPODEEJECUTOR',
                    'SECTOR', 'SUBSECTOR', 'ESTADO']#,'criticoSGRP58',
                    #'criticoSGRPProm72','critico_visitas']

## Columnas de variables numericas
cols_numeric = ['VALORSGR',u'VALORNACIÓN','VALOROTROS','VALORTOTAL']
cols_text = [x for x in cols_to_keep if (x not in cols_categorical and x not in cols_numeric)]

## Variables categoricas: text -> int
labels_categorical = []
dicts_categorical = []
for text_field in cols_categorical:
    dict_categorical = text2cat(df[text_field])
    dicts_categorical.append(dict_categorical)
    df_proc[text_field] = df_proc[text_field].apply(dict_categorical.get).astype(float)
    labels = df[text_field].as_matrix()
    labels_categorical.append(labels)

## Texto unido
d_text = df_proc['texto_unificado']
d_text = d_text.fillna('')

## Convertir a lista, separar palabras remover noChar y num
proy_RAWtext = d_text.tolist()
proy_texts = [[word for word in document.lower().split()]
         for document in proy_RAWtext]
proy_texts = [remove_noChar(text) for text in proy_texts]
proy_texts = [remove_shortW(text) for text in proy_texts]

with open('entrada\\stop_words_spanish.txt', 'rb') as f:
    sw_spanish = f.read().decode('latin-1').replace(u'\r', u'').split(u'\n')

### Lista adicional de stop words en español
with open('entrada\\filtro_palabras.txt', 'rb') as f:
    filtro_palabras = f.read().decode('utf-8').replace(u'\r', u'').split(u'\n')

###############################################################################
custom_stopwords = set(sw_spanish)
all_stopwords = custom_stopwords | set(filtro_palabras)
proy_texts = [remove_sw(proy_txt, all_stopwords) for proy_txt in proy_texts]

## Unir palabras resultantes por proyecto
proy_texts_join = [join_text(text) for text in proy_texts]

doc2vec = True
tf_idf = True
if doc2vec:
    print("Vectorizacion D2V")
    ## Construir corpus
    train_corpus = list(read_list(proy_texts_join))
    
    ## Vectorizacion
    modelSize = 400
    model = gensim.models.doc2vec.Doc2Vec(size=modelSize, min_count=10, window=5,
                                          workers=4, iter=70)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)
    
    ## Extraer vectores para cada proyecto y determinar similaridad entre todos
    ranks = []
    second_ranks = []
    inferred_vectors = np.zeros((len(train_corpus),modelSize))
    for doc_id in range(len(train_corpus)):
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)
        
        second_ranks.append(sims[1])
        inferred_vectors[doc_id] = inferred_vector.transpose()
    
    text_vectors_d2v = inferred_vectors

elif tf_idf:
    print("Vectorizacion BOW")
    use_hashing = False          
    use_idf = True               
    n_features = 6000            
    n_components = 750           
    
    t0 = time()
    if use_hashing:
        if use_idf:
            # Normalizacion IDF normalization a la salida del HashingVectorizer
            hasher = HashingVectorizer(n_features=n_features, analyzer='word',
                                       non_negative=True, norm=None, binary=False)
            vectorizer = make_pipeline(hasher, TfidfTransformer())
        else:
            vectorizer = HashingVectorizer(n_features=n_features, analyzer='word',
                                           non_negative=False, norm='l2',
                                           binary=False)
    else:
        vectorizer = TfidfVectorizer(max_df=0.5, max_features=n_features,
                                     min_df=2, analyzer='word', use_idf=use_idf)
    
    text_vectors_tfidf = vectorizer.fit_transform(proy_texts_join)
    
    if n_components:
        print("Reduccion de dimensionalidad - LSA")
        t0 = time()
        svd = TruncatedSVD(n_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)
    
        text_vectors_tfidf = lsa.fit_transform(text_vectors_tfidf)

        print("Finalizado en %fs" % (time() - t0))
    
        explained_variance = svd.explained_variance_ratio_.sum()
        print("Varianza explicada despues de SVD: {}%".format(
            int(explained_variance * 100)))

###############################################################################
## Reduccion de dimensionalidad sobre el resultado de la vectorizacion de texto unicamente

## LSA - PCA
dLSA = False
dPCA = False
if dLSA:
    n_components = 20
    svd = TruncatedSVD(n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    vectorsRed = lsa.fit_transform(inferred_vectors)
elif dPCA:
    pca = PCA(n_components=20)
    vectorsRed = pca.fit_transform(inferred_vectors)
else:
    vectorsRed = inferred_vectors

## tSNE sobre reduced inferred_vector (vectorsRed) **para todo train corpus**
model_tsne = TSNE(n_components=2, init='pca', random_state=None)
vectors_tsne = model_tsne.fit_transform(vectorsRed)

## Graficar tSNE con diferentes etiquetas
# Para todo i (variables categoricas)
i = 0
plt.figure(1)
for item in cols_categorical:
    plt.subplot(2,6,i+1)
    plot_dr(vectors_tsne, labels_categorical[i], cols_categorical[i], i)
    i = i + 1
plt.savefig('results.png', dpi=1200)
plt.show()

# Para solo una variable categorica (i)
i=10
plot_dr(vectors_tsne, labels_categorical[i], cols_categorical[i], i)
plt.savefig('resultsOne.png', dpi=1200)
plt.show()

###############################################################################
##################   Construccion del modelo   ################################

X = df_proc.drop(['BPIN','NOMBREDELPROYECTO', u'CÓDIGODANEENTEEJECUTOR',
                u'AVANCEFÍSICO', 'AVANCEFINANCIERO',
                'ESTADO', u'FECHAAPROBACIÓN',
                'VALORSGR',u'VALORNACIÓN', 'VALOROTROS', 'texto_unificado',
                'criticoSGRP58', 'criticoSGRPProm72','critico_visitas'],axis=1)

Y = df_proc["criticoSGRPProm72"]
Y_array = np.array(Y).astype(int)

X_array = np.array(X)
enc = preprocessing.OneHotEncoder()
X_enc = enc.fit_transform(X_array[:,0:9]).toarray()

X_compl = np.hstack((X_enc, text_vectors_d2v))

X_scaled = preprocessing.scale(np.array(X_compl[Y_array < 2, :]))
Y_b = Y_array[Y_array < 2]

x_train, x_test, y_train, y_test = cross_validation.train_test_split(X_scaled, Y_b, test_size = 0.2)


###############################################################################
#####     SVM - Model
###############################################################################

class_weight = {0: 0.1, 1: 100}
model = SVC(kernel='rbf', C=1, probability=True, class_weight=class_weight)

model.fit(x_train, y_train)
y_proba = model.predict_proba(x_test)
y_predict = (y_proba[:,1]>0.5).astype(int)
model.score(x_train, y_train)

false_positive_rate, true_positive_rate, thresholds1 = roc_curve(y_test, y_proba[:,1])
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.subplot(1, 2, 1)
plt.plot(false_positive_rate, true_positive_rate, 'b', label = 'AUC = %0.2f'% roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid()
plt.title('ROC Curve')
plt.legend(loc="lower right")
TP, FP, FN, TN = 0, 0, 0, 0

for i in xrange(len(y_predict)):
    if y_test[i]==1 and y_predict[i]==1:
        TP += 1
    if y_test[i]==0 and y_predict[i]==1:
        FP += 1
    if y_test[i]==1 and y_predict[i]==0:
        FN += 1
    if y_test[i]==0 and y_predict[i]==0:
        TN += 1
print 'TP: '+ str(TP)
print 'FP: '+ str(FP)
print 'FN: '+ str(FN)
print 'TN: '+ str(TN)

f1 = 2*TP / float(2*TP+FP+FN)
print ('F1 score = %0.2f' %f1)

predict_proba_train = model.predict_proba(x_train)
y_predict_train = (predict_proba_train[:,1]>0.5).astype(int)
mse_training  = np.mean(( y_predict_train - y_train )** 2)
print('MSE-training = %0.2f' %mse_training)

predict_proba_test = model.predict_proba(x_test)
y_predict_test = (predict_proba_test[:,1]>0.5).astype(int)
mse_test  = np.mean(( y_predict_test - y_test )** 2)
print('MSE-test = %0.2f' %mse_test)

