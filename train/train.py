import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import chardet

rawdata = open('spam.csv', 'rb').read()
result = chardet.detect(rawdata)
encoding = result['encoding']

df = pd.read_csv('spam.csv', encoding=encoding)

tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['v2'])
sequences = tokenizer.texts_to_sequences(df['v2']) 
padded = pad_sequences(sequences, padding='post')

df['v1'] = df['v1'].map({'ham': 0, 'spam': 1}) 

X_train, X_test, y_train, y_test = train_test_split(padded, df['v1'], test_size=0.2, random_state=42) 

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test), batch_size=32)

model.save('model.h5')
