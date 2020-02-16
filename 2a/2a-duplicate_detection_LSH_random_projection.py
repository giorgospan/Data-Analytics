#The code for this random projection LSH hashtable can found at:
#https://gist.github.com/santhoshhari/52d8b7acd39c1b744736d7591497ae39#file-hashtable-py
import numpy as np
import pandas as pd
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

hash_size = 10


class HashTable:
    def __init__(self, hash_size, inp_dimensions):
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.hash_table = dict()
        self.projections = np.random.randn(self.hash_size, inp_dimensions)
        
    def generate_hash(self, inp_vector):
        bools = (np.dot(inp_vector, self.projections.T) > 0).astype('int')
        return ''.join(bools.astype('str'))

    def __setitem__(self, inp_vec, label):
        hash_value = self.generate_hash(inp_vec)
        self.hash_table[hash_value] = self.hash_table\
            .get(hash_value, list()) + [label]
        
    def __getitem__(self, inp_vec):
        hash_value = self.generate_hash(inp_vec)
        return self.hash_table.get(hash_value, [])

df_train  = pd.read_csv('../datasets/q2a/corpusTrain.csv', encoding='utf-8')
df_train['Content'] = df_train['Content'].str.encode('ascii', 'ignore').str.decode('ascii')

vectorizer = TfidfVectorizer()
train = vectorizer.fit_transform(df_train['Content'])


dim = len(vectorizer.get_feature_names())
print('dim = {}'.format(dim))
h = HashTable(hash_size=hash_size, inp_dimensions=dim)


print('Building index...')
total_time = time.time()
for i, v in enumerate(train):
    h[v.toarray()[0]] = i
print('Build time: {}'.format(time.time() - total_time)) 
print('Index finished...')

df_test = pd.read_csv('../datasets/q2a/corpusTest.csv', encoding='utf-8')
df_test['Content'] = df_test['Content'].str.encode('ascii', 'ignore').str.decode('ascii')
test = vectorizer.transform(df_test['Content'])

num_duplicates = 0
total_time = time.time()
for v in test:

    L = h[v.toarray()[0]]
    tmp = train[L]
    Y = cosine_similarity(v, tmp)
    num_duplicates += len(np.where((Y > 0.8).any(axis=1))[0])
print('Query time: {}'.format(time.time() - total_time)) 
print('Duplicates: {}'.format(num_duplicates))
