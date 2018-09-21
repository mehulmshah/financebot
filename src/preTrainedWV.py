from gensim.models import KeyedVectors
from keras.models import Sequential
from keras.layers import Conv1D, GlobalMaxPooling1D, LSTM, Flatten, MaxPooling1D
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from nltk.corpus import stopwords
from pathlib import Path
import pickle
import numpy as np

THRESHOLD = 0.80
MAX_SEQUENCE_LENGTH = 20
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.25

EMBEDDING_FILE = "data/GoogleNews-vectors-negative300.bin"
category_index = {"balance": 0, "budgeting": 1, "housing": 2} # "unknown": 3}
category_reverse_index = dict((y, x) for (x, y) in category_index.items())
STOPWORDS = set(stopwords.words("english"))

trainingDataPickle = Path("data/trainingDataPickle")
pickledData = pickle.load(open('data/trainingDataPickle', 'rb'))
balance = pickledData['balance']
budgeting = pickledData['budgeting']
housing = pickledData['housing']
unknown = pickledData['unknown']

datasets = []
datasets.extend(balance)
datasets.extend(budgeting)
datasets.extend(housing)
# datasets.extend(unknown)

labels = [0 for i in range(len(balance))]
labels.extend([1 for i in range(len(budgeting))])
labels.extend([2 for i in range(len(housing))])
# labels.extend([3 for i in range(len(unknown))])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(datasets)
sequences = tokenizer.texts_to_sequences(datasets)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

xTrain = data[:-nb_validation_samples]
yTrain = labels[:-nb_validation_samples]
xVal = data[-nb_validation_samples:]
yVal = labels[-nb_validation_samples:]

print("Loading in word2vec pretrained w/ google")
model = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
embedding_layer = model.get_keras_embedding(train_embeddings=False)
embedding_layer.input_length = MAX_SEQUENCE_LENGTH

print("creating first model")
model1 = Sequential()
model1.add(embedding_layer)
model1.add(Conv1D(250,3,padding='valid',activation='relu',strides=1))
model1.add(GlobalMaxPooling1D())
model1.add(Dense(250))
model1.add(Dropout(0.2))
model1.add(Activation('relu'))
model1.add(Dense(len(category_index)))
model1.add(Activation('sigmoid'))
model1.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
model1.summary()
model1.fit(xTrain, yTrain, validation_data=(xVal, yVal), epochs=25, batch_size=10)

print("creating second model")
model2 = Sequential()
model2.add(embedding_layer)
model2.add(Conv1D(128, 5, activation='relu'))
model2.add(MaxPooling1D())
model2.add(Conv1D(128, 5, activation='relu'))
model2.add(MaxPooling1D())
model2.add(Flatten())
model2.add(Dense(128, activation='relu'))
model2.add(Dense(len(category_index), activation='softmax'))
model2.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
model2.summary()
model2.fit(xTrain, yTrain, validation_data=(xVal, yVal), epochs=25, batch_size=10)

print("creating third model")
model3 = Sequential()
model3.add(embedding_layer)
model3.add(Dropout(0.2))
model3.add(Conv1D(64, 5, activation='relu'))
model3.add(MaxPooling1D(pool_size=4))
model3.add(LSTM(100))
model3.add(Dense(len(category_index), activation='sigmoid'))
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model3.fit(xTrain, yTrain, validation_data=(xVal, yVal), epochs=25, batch_size=10)

model1.save('../logs/model1.h5')
model2.save('../logs/model2.h5')
model3.save('../logs/model3.h5')
pickle.dump({'tokenizer': tokenizer}, open('data/tokenizer', 'wb'))
