from gensim.models import KeyedVectors
from keras.models import Sequential
from keras.layers import Conv1D, GlobalMaxPooling1D, Flatten, MaxPooling1D
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from nltk.corpus import stopwords
from pathlib import Path
import pickle
import numpy as np

MAX_SEQUENCE_LENGTH = 15
EMBEDDING_DIM = 300
NUM_WORDS = 1000
VALIDATION_SPLIT = 0.25

EMBEDDING_FILE = "data/GoogleNews-vectors-negative300.bin"
category_index = {"balance": 0, "budgeting": 1, "housing": 2, "unknown": 3}
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
datasets.extend(unknown)

labels = [0 for i in range(len(balance))]
labels.extend([1 for i in range(len(budgeting))])
labels.extend([2 for i in range(len(housing))])
labels.extend([3 for i in range(len(unknown))])

tokenizer = Tokenizer(num_words=NUM_WORDS)
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

model = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
embedding_layer = model.get_keras_embedding(train_embeddings=False)
embedding_layer.input_length = MAX_SEQUENCE_LENGTH

model_1 = Sequential()
model_1.add(embedding_layer)
model_1.add(Conv1D(250,3,padding='valid',activation='relu',strides=1))
model_1.add(GlobalMaxPooling1D())
model_1.add(Dense(250))
model_1.add(Dropout(0.2))
model_1.add(Activation('relu'))
model_1.add(Dense(len(category_index)))
model_1.add(Activation('sigmoid'))
model_1.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
model_1.summary()

x = Sequential()
x.add(embedding_layer)
x.add(Conv1D(128, 5, activation='relu'))
x.add(MaxPooling1D())
x.add(Conv1D(128, 5, activation='relu'))
x.add(MaxPooling1D())
x.add(Flatten())
x.add(Dense(128, activation='relu'))
x.add(Dense(len(category_index), activation='softmax'))
x.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
x.summary()

# happy learning!
x.fit(xTrain, yTrain, validation_data=(xVal, yVal),
          epochs=30, batch_size=10)
score = x.evaluate(xVal, yVal, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model_1.fit(xTrain, yTrain, validation_data=(xVal, yVal), epochs=50, batch_size=10)
score = model_1.evaluate(xVal, yVal, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



def predict_x(sentence):
    seq = tokenizer.texts_to_sequences([sentence])
    pad_seq = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    print("-" * 10)
    print("Predicted category: ", category_reverse_index[model_1.predict_classes(pad_seq, verbose=0)[0]])
    print("-" * 10)
    probabilities = x.predict(pad_seq, verbose=0)
    probabilities = probabilities[0]
    print("Balance: {}\nBudgeting: {}\nHousing: {}\n".format(probabilities[category_index["balance"]],
                                                             probabilities[category_index["budgeting"]],
                                                             probabilities[category_index["housing"]]))

def predict_model_1(sentence):
    seq = tokenizer.texts_to_sequences([sentence])
    pad_seq = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    print("-" * 10)
    print("Predicted category: ", category_reverse_index[model_1.predict_classes(pad_seq, verbose=0)[0]])
    print("-" * 10)
    probabilities = model_1.predict(pad_seq, verbose=0)
    probabilities = probabilities[0]
    print("Balance: {}\nBudgeting: {}\nHousing: {}\n".format(probabilities[category_index["balance"]],
                                                             probabilities[category_index["budgeting"]],
                                                             probabilities[category_index["housing"]]))

def main():
    while True:
        sentence = input('--> ')
        predict_model_1(sentence)
        predict_x(sentence)

if __name__ == "__main__":
    main()