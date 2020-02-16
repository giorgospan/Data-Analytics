import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import preprocessing

import time
import numpy as np
from nltk.corpus import stopwords


stop_words = set(stopwords.words('english'))
with open('./extra_stopwords.txt') as f:
    for line in f:
        stop_words.add(line[:-1])
stop_words = list(stop_words)
#------------Read the csv file and change the encoding----------------
df  = pd.read_csv('../datasets/q1/train.csv', encoding='utf-8')
df['Title'] = df['Title'].str.encode('ascii', 'ignore').str.decode('ascii')
df['Content'] = df['Content'].str.encode('ascii', 'ignore').str.decode('ascii')
df['Label'] = df['Label'].str.encode('ascii', 'ignore').str.decode('ascii')

df['Combined']  = df['Title'] + ' ' + df['Title'] + ' ' + df['Title'] + df['Content']
#---------------------------------------------------------------------

#--------- INITIALIZATIONS -------------
le = preprocessing.LabelEncoder()
y = le.fit_transform(df['Label'])
clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
#----------------------------------------


total_time = time.time()
#------------------  BAG OF WORDS ------------------
vectorizer = TfidfVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(df['Combined'])

print('Starting 5-fold for BOW')
kfold_time = time.time()
kf = KFold(n_splits=5)
accuracy = 0
precision = 0
recall = 0
fmeasure = 0

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    
    accuracy += accuracy_score(y_test, predictions)
    precision += precision_score(y_test, predictions, average='macro')
    recall += recall_score(y_test, predictions, average='macro')
    fmeasure += f1_score(y_test, predictions, average='macro')

accuracy /= 5
precision /= 5
recall /= 5
fmeasure /= 5

print('accuracy = {}, precision = {}, recall = {}, f1-measure = {}'.format(round(accuracy, 4), round(precision,4), round(recall,4), round(fmeasure,4)))
#----------------------------------------------------
print('5-fold time: {} s'.format(time.time() - kfold_time))
print('Total for BOW: {} s'.format(time.time() - total_time))

total_time = time.time()
#--------------------- SVD --------------------------
svd = TruncatedSVD(n_components=30, random_state=42)
X = svd.fit_transform(X)

print('Starting 5-fold for SVD')
kfold_time = time.time()
kf = KFold(n_splits=5)
accuracy = 0
precision = 0
recall = 0
fmeasure = 0

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    
    accuracy += accuracy_score(y_test, predictions)
    precision += precision_score(y_test, predictions, average='macro')
    recall += recall_score(y_test, predictions, average='macro')
    fmeasure += f1_score(y_test, predictions, average='macro')

accuracy /= 5
precision /= 5
recall /= 5
fmeasure /= 5

print('accuracy = {}, precision = {}, recall = {}, f1-measure = {}'.format(round(accuracy, 4), round(precision,4), round(recall,4), round(fmeasure,4)))
#----------------------------------------------------
print('5-fold time: {} s'.format(time.time() - kfold_time))
print('Total for SVD: {} s'.format(time.time() - total_time))