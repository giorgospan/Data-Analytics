import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

size_test = 5374

vectorizer = TfidfVectorizer()

df_train  = pd.read_csv('../datasets/q2a/corpusTrain.csv', encoding='utf-8')
df_train['Content'] = df_train['Content'].str.encode('ascii', 'ignore').str.decode('ascii')
train_vectors = vectorizer.fit_transform(df_train['Content'])

df_test  = pd.read_csv('../datasets/q2a/corpusTest.csv', encoding='utf-8')
df_test['Content'] = df_test['Content'].str.encode('ascii', 'ignore').str.decode('ascii')
test_vectors = vectorizer.transform(df_test['Content'])

print(test_vectors.shape[0])
#For a 16GB RAM machine, 1000 vectors at a time is the maximum we can handle. That is why we are going to
#get the results in batches of max 1000 test vectors at a time.
slices = [slice(0, 1000), slice(1000, 2000), slice(2000, 3000), slice(3000, 4000), slice(4000, 5000), slice(5000, 5374)]

num_duplicates = 0

for s in slices:
    Y = cosine_similarity(test_vectors[s], train_vectors)
    num_duplicates += len(np.where((Y > 0.8).any(axis=1))[0])

print('Duplicates: {}'.format(num_duplicates))