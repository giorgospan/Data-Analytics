import pandas as pd
import re

from wordcloud import WordCloud

import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

punkt = {',', '.', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '/', '\\', '-', '_', '+', '=', '"', '~', '`', '<', '>', '?', ':', ';', "'", '|', '[', ']', '{', '}', '*'}
stop_words = set(stopwords.words('english'))

stop_words = set(stopwords.words('english'))
with open('../extras/extra_stopwords.txt') as f:
    for line in f:
        stop_words.add(line[:-1])

#wordcloud dictionary
wordcloud_dict = {'Business': [], 'Entertainment': [], 'Health': [], 'Technology': []}

#------------Read the csv file and change the encoding----------------
df  = pd.read_csv('../datasets/q1/train.csv', encoding='utf-8')
#Uncomment to test without waiting too long...
# df = df[:100]
df['Title'] = df['Title'].str.encode('ascii', 'ignore').str.decode('ascii')
df['Content'] = df['Content'].str.encode('ascii', 'ignore').str.decode('ascii')
df['Label'] = df['Label'].str.encode('ascii', 'ignore').str.decode('ascii')
#---------------------------------------------------------------------

df['Content'] = 20*(df['Title'] + ' ') + df['Content']

df['Content'] = df['Content'].apply(lambda x:re.sub(r'[^\w\s]',' ',x))
df['Content'] = df['Content'].apply(word_tokenize)
df['Content'] = df['Content'].apply(lambda x:[word for word in x if word not in (stop_words) and len(word)>1])
df['Content'] = df['Content'].apply(lambda x:' '.join(x))
wordcloud_dict = df.groupby('Label')['Content'].apply(list).to_dict()

for category in wordcloud_dict:
    wordcloud = WordCloud(width=1920, height=1080).generate_from_text(' '.join(wordcloud_dict[category]))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    wordcloud.to_file(category+'.jpg')