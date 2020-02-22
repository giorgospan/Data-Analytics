import pandas as pd
import numpy as np

import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
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
df_train['Combined'] = df_train['Question1']
df_train['Combined'].append(df_train['Question2'])
#---------------------------------------------------------------------


le = preprocessing.LabelEncoder()
y = le.fit_transform(df_train['IsDuplicate'])
clf = LinearSVC(random_state=42)
vectorizer = TfidfVectorizer()

vectorizer.fit(df_train['Combined'].values.astype(str))

print(len(vectorizer.get_feature_names()))

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

print('Starting 5-fold')
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