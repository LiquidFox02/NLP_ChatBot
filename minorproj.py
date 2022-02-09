#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 21:57:29 2022

@author: ranchhor
"""
import os
import re
import string

import nltk
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, LSTM, Dropout
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


df = pd.read_csv('~/Desktop/ML/NLP/tweet_emotions.csv', dtype=str)


""" preprocessing """

def tokenize(text):
    tokens = re.split('W+',text)
    return tokens

nltk.download('punkt')
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')    

def rm_stpwrds(text):
    op = [i for i in text if i not in stopwords]
    return op

stopwords = nltk.corpus.stopwords.words('english')
def clean_text(text):
    regex_html = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    remove_digits = str.maketrans('', '', string.digits + string.punctuation)
    text = re.sub(regex_html, '', text)
    text = text.translate(remove_digits)
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split()).lower()






df['text'] = (df.content).apply(clean_text)
df['tokenizedtext'] = df['text'].apply(lambda x:tokenize(x))
df['stpwrdless'] = df['tokenizedtext'].apply(lambda x:rm_stpwrds(x))

print(df['sentiment'].unique())
df.sentiment.replace({'empty':'sadness','worry':'sadness','hate':'anger','enthusiasm':'fun', 'neutral':'happiness' , 'relief':'happiness', 'surprise':'fun'},inplace=True)
train_df,test_df=train_test_split(df,test_size=0.20,random_state=123)
#d = [(j.split() for j in i) for i in train_df['stpwrdless']]
d = []
for i in df['stpwrdless']:
    
    if len(i) > 0:
        q = i[0].split()
        d += [q]

#<sentences = [text.split() for text in train_df['stpwrdless']]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(d)
#print(d)
#print(tokenizer.word_index)
model = Word2Vec(d, min_count=1 ,vector_size=300, window=5 , sg=1)
#vectors = model.wv.vocab.keys()
print(model.wv.most_similar('man'))
words = list(model.wv.index_to_key)
lenvoc = len(tokenizer.word_index)+1
print(lenvoc)


embedding_matrix = np.zeros((lenvoc, 300))
for word, i in tokenizer.word_index.items():
  if word in model.wv:
    embedding_matrix[i] = model.wv[word]
print(embedding_matrix.shape)
tokenizer.fit_on_texts(train_df)
X_train = pad_sequences(tokenizer.texts_to_sequences(train_df.content), maxlen=300)
X_test = pad_sequences(tokenizer.texts_to_sequences(test_df.content), maxlen=300)
print(X_test.dtype)


y_train=train_df.sentiment
y_test=test_df.sentiment
catlen = len(np.unique(y_train))
labelencoder = LabelEncoder()
y_train = labelencoder.fit_transform(y_train)
y_test=labelencoder.fit_transform(y_test)

embedding_layer = Embedding(lenvoc, 300, weights=[embedding_matrix], input_length=300, trainable=True)
model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.5))
model.add(LSTM(100, dropout=0.2, activation='tanh',recurrent_dropout=0.2))
model.add(Dense(catlen, activation='softmax'))
model.summary()


model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer="adam",
              metrics=['accuracy'])

model_history=model.fit(X_train, y_train,batch_size=512,epochs=5,validation_split=0.1,verbose=1)

from keras.models import load_model
model_history.save('minorlst.h5')
joblib.dump(labelencoder, 'lblencod.pkl')
pmodel = keras.models.load_model("minorlst.h5")
labelencoder = joblib.load('lblencod.pkl')



