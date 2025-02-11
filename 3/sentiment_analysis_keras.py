# -*- coding: utf-8 -*-
"""sentiment_analysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15cnl9aMU_IotOs3aHRrgr1aaYFZVwwwC
"""

# !wget -O "/content/drive/My Drive/datasets.tar.gz"  --user bigdata --password d@t@s3t  195.134.67.98/documents/BigData/datasets2020.tar.gz
# !tar -xvzf "/content/drive/My Drive/datasets" -C "/content/drive/My Drive/"

# !wget -O "/content/drive/My Drive/glove.6B.zip" http://nlp.stanford.edu/data/glove.6B.zip
# !unzip "/content/drive/My Drive/glove.6B.zip" -d "/content/drive/My Drive/glove"

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D

K.tensorflow_backend._get_available_gpus()

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
with open('/content/drive/My Drive/extra_stopwords.txt') as f:
    for line in f:
        stop_words.add(line[:-1])
stop_words = list(stop_words)

df_train = pd.read_csv('/content/drive/My Drive/datasets/q3/train.csv', encoding='utf-8')
df_train['Content'] = df_train['Content'].str.encode('ascii', 'ignore').str.decode('ascii').str.lower().str.replace('<br />','')
df_train['Content'] = df_train['Content'].apply(word_tokenize)
df_train['Content'] = df_train['Content'].apply(lambda x:[word for word in x if word not in (stop_words) and len(word)>1])
reviews  = df_train['Content'] 
labels = df_train['Label']

# prepare tokenizer
t = Tokenizer(oov_token=True)
t.fit_on_texts(reviews)
vocab_size = len(t.word_index) + 1

# integer encode the reviews
encoded_revs = t.texts_to_sequences(reviews)

# pad the sequences to maxlen
maxlen = 100
encoded_revs = pad_sequences(encoded_revs,maxlen=maxlen)
encoded_revs.shape

# load the whole embedding into memory
embeddings_index = dict()
with open('/content/drive/My Drive/glove/glove.6B.100d.txt','r') as f:
  for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

# Embedding
max_features = 20000
maxlen = 100
embedding_size = 100

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70

# Training (only 2 epochs are needed as the dataset is very small)
batch_size = 30
epochs = 2

def create_model():
  # construct a CNN model
  model = Sequential()
  model.add(Embedding(vocab_size, embedding_size, weights=[embedding_matrix], input_length=encoded_revs.shape[1]))
  model.add(Dropout(0.25))
  model.add(Conv1D(filters,
                  kernel_size,
                  padding='valid',
                  activation='relu',
                  strides=1))
  model.add(MaxPooling1D(pool_size=pool_size))
  model.add(LSTM(lstm_output_size))
  model.add(Dense(1))
  model.add(Activation('sigmoid'))
  # compile the model
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  # print(model.summary())
  return model

# Metrics have been removed from Keras core. We need to calculate them using sklearn.
print('===============')
print('Starting 5-fold')
print('===============')

kf = KFold(n_splits=5)
accuracy = 0
precision = 0
recall = 0
fmeasure = 0

for train_index, test_index in kf.split(encoded_revs):
  
  # Fetch train and test data
  X_train, X_test = encoded_revs[train_index], encoded_revs[test_index]
  y_train, y_test = labels[train_index], labels[test_index]

  # Create model
  model = None
  model = create_model()

  # Fit on the train data
  model.fit(X_train, y_train,batch_size=batch_size,epochs=epochs)

  # Make predictions for test data
  predictions  = model.predict_classes(X_test)
  predictions  = [item for sublist in predictions for item in sublist]
  
  # Calculate metrics
  accuracy += accuracy_score(y_test, predictions)
  precision += precision_score(y_test, predictions, average='macro')
  recall += recall_score(y_test, predictions, average='macro')
  fmeasure += f1_score(y_test, predictions, average='macro')

accuracy /= 5
precision /= 5
recall /= 5
fmeasure /= 5

print('accuracy = {}, precision = {}, recall = {}, f1-measure = {}'.format(round(accuracy, 4), round(precision,4), round(recall,4), round(fmeasure,4)))

df_test  = pd.read_csv('/content/drive/My Drive/datasets/q3/test_without_labels.csv', encoding='utf-8')
df_test['Content'] = df_test['Content'].str.encode('ascii', 'ignore').str.decode('ascii').str.lower().str.replace('<br />','')
test_reviews  = df_test['Content']

encoded_test = t.texts_to_sequences(test_reviews)
encoded_test = pad_sequences(encoded_test,maxlen=maxlen)

predictions  = model.predict_classes(encoded_test)
predictions  = [item for sublist in predictions for item in sublist]

result = pd.DataFrame({'Id':df_test['Id'],'Predicted':predictions})
result.to_csv('sentiment_predictions.csv', sep=',', index=False)

# history = model.fit(encoded_revs, labels, epochs=5)
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()