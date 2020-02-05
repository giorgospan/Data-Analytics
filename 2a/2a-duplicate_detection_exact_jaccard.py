import pandas as pd

from sklearn.metrics import jaccard_score
import numpy as np

from multiprocessing import Process


def worker(test, train, i):

    dups = 0
    for q1 in test:
        s1 = set(q1.split())
        
        for q2 in train:
            s2 = set(q2.split())
            if len(s1.intersection(s2))/len(s1.union(s2)) > 0.8:
                dups += 1
                break
    print(i, dups)

df_train  = pd.read_csv('../datasets/q2a/corpusTrain.csv', encoding='utf-8')
df_train['Content'] = df_train['Content'].str.encode('ascii', 'ignore').str.decode('ascii')
train = df_train['Content']

df_test  = pd.read_csv('../datasets/q2a/corpusTest.csv', encoding='utf-8')
df_test['Content'] = df_test['Content'].str.encode('ascii', 'ignore').str.decode('ascii')
test = df_test['Content']

fr = 0
to = 500

num_cores = 10
slices = []
for i in range(num_cores):

    slices.append(slice(fr, to))
    fr += 500
    to += 500
slices[-1] = slice(4500, 5374)

processes = []

for i in range(10):

    p = Process(target=worker, args=(test[slices[i]], train, i))
    p.start()
    processes.append(p)

for i in range(10):
    processes[i].join()