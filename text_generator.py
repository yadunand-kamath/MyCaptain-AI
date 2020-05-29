# TEXT GENERATOR

import numpy
import sys
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import io
import pandas as pd

f = "/content/drive/My Drive/frankenstein.txt"
file = open(f,'r')

def tokenize_words(input):
  input = input.lower()
  tokenizer = RegexpTokenizer('\w+')
  tokens = tokenizer.tokenize(input)
  filtered = filter(lambda token: token not in stopwords.words('english'),tokens )
  return "".join(filtered)

processed_inputs = tokenize_words(file)

# chars to numbers
chars = sorted(list(set(processed_inputs)))
chars_to_num = dict((c,i) for i, c in enumerate(chars))

input_len = len(processed_inputs)
vocab_len = len(chars)
print("Toatal number of charcters:", input_len)
print("Total vocab:", vocab_len)

# Sequence length
seq_len = 100
x_data = []
y_data = []

for i in range(0, input_len - seq_len, 1):
  in_seq = processed_inputs[i:i + seq_len]
  out_seq = processed_inputs[i + seq_len]
  x_data.append([char_to_num[char] for char in in_seq])
  y_data.append(char_to_num[out_seq])

n_patterns = len(x_data)
print("Total Patterns:", n_patterns)

# Convert input sequence to np array
X = numpy.reshape(x_data, (n_patterns, seq_len, 1))
X = X/float(vocab_len)

# One-Hot Encoding
y = np_utils.to_categorical(y_data)

# Creating the Model
model = Sequential()
model.add(LSTM(256, input_shape = (X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# saving weights
filepath = "model_weights_saved.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose = 1, save_best_only=True, mode='min')
desired_callbacks = [checkpoint]

# fit model
model.fit(X,y, epochs=4, batch_size=256, callbacks=desired_callbacks)

# recompile model with saved weights
filename = "model_weights_saved.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

num_to_char = dict((i,c) for i,c in enumerate(chars))

# random seed to help generate
start = numpy.random.randint(0, len(x_data) - 1)
pattern = x_data[start]
print("Random Seeds: ")
print("\"",''.join([num_to_char[value] for value in pattern]), "\"")

# generate the text
for i in range(1000):
  x = numpy.reshape(pattern, (1,len(pattern),1))
  x = x/float(vocab_len)
  prediction = model.predict(x, verbose=0)
  index = numpy.argmax(prediction)
  result = num_to_char[index]
  seq_in = [num_to_char[value] for value in pattern]
  sys.stdout.write(result)
  pattern.append(index)
  pattern = pattern[1:len(pattern)]
