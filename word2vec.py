"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

#-------Preprocessing------------------------------#


def onehot(word):
   word_vec = np.zeros(len(word_index))
   for i in word:
       if i == 0:
           continue
       word_vec[i-1] = 1
   return word_vec 

def training_data(seq , length , window_size):
    
    trainin_set = [] 
    for i in seq:
        for j in range(length):
            target = []
            neighbours = []
            target.append(i[j])   
            if(i[j] == 0):
                continue
            if(j == 0):
                for k in range(1, window_size +1):
                    neighbours.append(i[k])
            elif(j == length -1):
                for k in range ((length -(1 + window_size)) , j):
                    neighbours.append(i[k])
            else:
                for k in range(1,window_size+1):
                    #left neighbours
                    if j-k >= 0:
                        neighbours.append(i[j-k])
                    #right neighbours
                    if j+k <length:
                        neighbours.append(i[j+k])
            if(len(neighbours) < 2*window_size):
                for it in range(0,(2*window_size - len(neighbours))):
                    neighbours.append(0)
            temp = []
            target = onehot(target)
            neighbours = onehot(neighbours)
            temp.append(target)
            temp.append(neighbours)
            trainin_set.append(temp)
    trainin_set = np.array(trainin_set)
    return trainin_set



df = pd.read_csv("/home/ranchhor/Desktop/ML/NLP/train.csv") 
sentences = df['review']
sentences = sentences[:50]
#sentences = sentences[:100]
#sentences = ["the man was a boy", 
#             "the woman was a girl",
#             "he is a man",
#             "she is a woman",
#             "he is a king",
#             "she is the queen"]
#print(sentences)
# creating an instance of the tokenizer
tokenizer = Tokenizer(oov_token = "<OOV>")
tokenizer.fit_on_texts(sentences)
text_sequences = tokenizer.texts_to_sequences(sentences)
word_index = tokenizer.word_index
print(word_index)
#padding
padded = pad_sequences(text_sequences,padding = 'post')
#print(padded)
datasets = training_data(padded, len(padded[0]), 2)
print("the final")
#print(datasets)
y = list(i[0] for i in datasets)
x = list(i[1] for i in datasets)
x = np.array(x)
y = np.array(y)
#print(x,y)
print(x.shape)
print(y.shape)
print(len(y[0]))
vocab_size = len(word_index)
embedding_dim = 100
max_length = len(x[0]) 
#---------Initializing the neural network for word2vec------------#

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim ,\
                              input_length=max_length,name='embedding'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(8 , activation = 'relu'),
    tf.keras.layers.Dense(max_length , activation = 'softmax')
])
model.compile(optimizer='sgd' , loss = tf.keras.losses.CategoricalCrossentropy(),\
              metrics=["accuracy"])

num_epochs = 10
history = model.fit(x, y,epochs=num_epochs)
embedd_matrix = model.get_layer('embedding').get_weights()[0]
print(embedd_matrix)
"""


from gensim.models import Word2Vec
# define training data
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
			['this', 'is', 'the', 'second', 'sentence'],
			['yet', 'another', 'sentence'],
			['one', 'more', 'sentence'],
			['and', 'the', 'final', 'sentence']]
# train model
model = Word2Vec(sentences, min_count=1)
# summarize the loaded model
print(model)
# summarize vocabulary

# access vector for one word
print(model['sentence'])
# save model
model.save('model.bin')
# load model
new_model = Word2Vec.load('model.bin')
print(new_model)