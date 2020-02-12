import pandas as pd
import numpy as np

import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import preprocessing

#stop words don't help for some reason. w/e


#------------Read the csv file and change the encoding----------------
df_train  = pd.read_csv('../datasets/q2b/train.csv', encoding='utf-8')
df_train['Question1'] = df_train['Question1'].str.encode('ascii', 'ignore').str.decode('ascii').str.lower().str.replace('[^\w\s]','')
df_train['Question2'] = df_train['Question2'].str.encode('ascii', 'ignore').str.decode('ascii').str.lower().str.replace('[^\w\s]','')
df_train['Combined'] = df_train['Question1'] #+ ' ' + df_train['Question2']
df_train['Combined'].append(df_train['Question2'])
#---------------------------------------------------------------------


le = preprocessing.LabelEncoder()
y = le.fit_transform(df_train['IsDuplicate'])
clf = LinearSVC(random_state=42)
vectorizer = TfidfVectorizer()
vectorizer.fit(df_train['Combined'].values.astype(str))

X1 = vectorizer.transform(df_train['Question1'].values.astype(str))
X2 = vectorizer.transform(df_train['Question2'].values.astype(str))

#heuristic 1
#X = X1.multiply(X2)
#heuristic 2
#X = abs(X1 + X2)

#heuristic 3
#best so far X.power(0.7) acc = 78.08
X = abs(X1 - X2)
X = X.power(0.7)

#---------------------------------------------------------

df_test  = pd.read_csv('../datasets/q2b/test_without_labels.csv', encoding='utf-8')
df_test['Question1'] = df_test['Question1'].str.encode('ascii', 'ignore').str.decode('ascii').str.lower().str.replace('[^\w\s]','')
df_test['Question2'] = df_test['Question2'].str.encode('ascii', 'ignore').str.decode('ascii').str.lower().str.replace('[^\w\s]','')
df_test['Combined'] = df_test['Question1'] #+ ' ' + df_train['Question2']
df_test['Combined'].append(df_test['Question2'])

clf.fit(X, y)

X_test1 = vectorizer.transform(df_test['Question1'].values.astype(str))
X_test2 = vectorizer.transform(df_test['Question2'].values.astype(str))

X_test = abs(X_test1 - X_test2)
X_test = X_test.power(0.7)

predictions = clf.predict(X_test)


result = pd.DataFrame({'Id':df_test['Id'],'Predicted':predictions})
result.to_csv('2b_predictions.csv', sep=',', index=False)