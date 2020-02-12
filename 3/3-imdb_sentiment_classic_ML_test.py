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


#------------Read the csv file and change the encoding----------------
df_train  = pd.read_csv('../datasets/q3/train.csv', encoding='utf-8')
df_train['Content'] = df_train['Content'].str.encode('ascii', 'ignore').str.decode('ascii').str.lower().str.replace('<br />','')
#---------------------------------------------------------------------


le = preprocessing.LabelEncoder()
y_train = le.fit_transform(df_train['Label'])
clf = LinearSVC(random_state=42)
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(df_train['Content'].values.astype(str))

clf.fit(X_train, y_train)

#------------------------------------------
df_test  = pd.read_csv('../datasets/q3/test_without_labels.csv', encoding='utf-8')
df_test['Content'] = df_test['Content'].str.encode('ascii', 'ignore').str.decode('ascii').str.lower().str.replace('<br />','')

X_test = vectorizer.transform(df_test['Content'].values.astype(str))
predictions = clf.predict(X_test)

predictions = le.inverse_transform(predictions)

result = pd.DataFrame({'Id':df_test['Id'],'Predicted':predictions})
result.to_csv('imdb_test.csv', sep=',', index=False)