import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
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
df_train = pd.read_csv('../datasets/q1/train.csv', encoding='utf-8')
df_train['Title'] = df_train['Title'].str.encode('ascii', 'ignore').str.decode('ascii')
df_train['Content'] = df_train['Content'].str.encode('ascii', 'ignore').str.decode('ascii')
df_train['Label'] = df_train['Label'].str.encode('ascii', 'ignore').str.decode('ascii')

df_train['Combined']  = df_train['Title'] + ' ' + df_train['Content']
#---------------------------------------------------------------------

#--------- INITIALIZATIONS -------------
le = preprocessing.LabelEncoder()
y = le.fit_transform(df_train['Label'])
clf = LinearSVC(random_state=42, tol=1e-5)
#----------------------------------------

#---------- CREATE VECTORIZER -----------
vectorizer = TfidfVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(df_train['Combined'])
#----------------------------------------

print('Running on test set...')
#------------Read the csv file and change the encoding----------------
df_test = pd.read_csv('../datasets/q1/test_without_labels.csv', encoding='utf-8')
df_test['Title'] = df_test['Title'].str.encode('ascii', 'ignore').str.decode('ascii')
df_test['Content'] = df_test['Content'].str.encode('ascii', 'ignore').str.decode('ascii')


df_test['Combined']  = df_test['Title'] + ' ' + df_test['Content']
#---------------------------------------------------------------------

print('Training Linear SVM...')
clf.fit(X, y)
print('Finished training...')

X = vectorizer.transform(df_test['Combined'])
predictions = clf.predict(X)

predictions = le.inverse_transform(predictions)

result = pd.DataFrame({'Id':df_test['Id'],'Predicted':predictions})
result.to_csv('testSet_categories.csv', sep=',', index=False)