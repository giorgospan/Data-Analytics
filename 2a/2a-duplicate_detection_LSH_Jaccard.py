import pandas as pd

from datasketch import MinHash, MinHashLSH

num_perm = 16
threshold = 0.8

m_list = []

df  = pd.read_csv('../datasets/q2a/corpusTrain.csv', encoding='utf-8')
df['Content'] = df['Content'].str.encode('ascii', 'ignore').str.decode('ascii')
for q in df['Content']:
    q = set(q.split())
    m = MinHash(num_perm=num_perm)
    for i in q:
        m.update(i)
    
    m_list.append(m)

# Create LSH index
lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

for i, m in enumerate(m_list):
    lsh.insert(str(i), m)


df  = pd.read_csv('../datasets/q2a/corpusTest.csv', encoding='utf-8')
df['Content'] = df['Content'].str.encode('ascii', 'ignore').str.decode('ascii')

num_duplicates = 0

for q in df['Content']:
    q = set(q.split())
    m = MinHash(num_perm=num_perm)
    for i in q:
        m.update(i)
    
    if len(lsh.query(m)) > 0:
        num_duplicates += 1

print(num_duplicates)
