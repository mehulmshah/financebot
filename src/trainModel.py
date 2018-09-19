#!/usr/bin/env python3
# trainModel.py
# This file will load the conversational framework for the chatbot, and train
# a model in Tensorflow to recognize what category a user request is in.

import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from src.util.dataUtil import getBalanceData, getBudgetingData, getHousingData, getNegExampleData
from nltk.corpus import stopwords
from sklearn import model_selection
from sklearn import cross_validation
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle

stemmer = LancasterStemmer()
st = StanfordNERTagger('ner/english.muc.7class.distsim.crf.ser.gz', 'ner/stanford-ner.jar', encoding='utf-8')

with open('src/data/intents.json') as f:
    intents = json.load(f)

intents['categorySet'][0]['trainData'] = getBalanceData()
intents['categorySet'][1]['trainData'] = getBudgetingData()
intents['categorySet'][2]['trainData'] = getHousingData()
intents['categorySet'][3]['trainData'] = getNegExampleData()

words = []
sentences = []
categories = ['balance', 'budgeting', 'housing', 'unknown']

for category in intents['categorySet']:
    for sentence in category['trainData']:
        tokenized_sentence = word_tokenize(sentence)
        words.extend(tokenized_sentence)
        sentences.append((tokenized_sentence, category['name']))

stop_words = set(stopwords.words('english'))
words = [stemmer.stem(w.lower()) for w in words if w.isalnum() and not w in stop_words]
words = list(dict.fromkeys(words))

trainingSet = []
for sentence in sentences:
    bagOfWords = []
    tokenized_sentence = sentence[0]
    tokenized_sentence = [stemmer.stem(word.lower()) for word in tokenized_sentence]
    for w in words:
        bagOfWords.append(1) if w in tokenized_sentence else bagOfWords.append(0)
    output = [0,0,0,0]
    output[categories.index(sentence[1])] = 1
    trainingSet.append([bagOfWords, output])

classified_text = st.tag(word_tokenize(text))
print(classified_text)

random.shuffle(trainingSet)
trainingSet = np.array(trainingSet)
dataTrain, dataTest, labelTrain, labelTest = model_selection.train_test_split(trainingSet[:,0], trainingSet[:,1], test_size=0.25)
dataTest, dataVal, labelTest, labelVal = cross_validation.train_test_split(dataTest, labelTest, test_size=0.5)

tf.reset_default_graph()
net = tflearn.input_data(shape=[None, len(dataTrain[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, len(labelTrain[0]), activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')
model = tflearn.DNN(net, tensorboard_dir='logs')

model.fit(dataTrain, labelTrain, validation_set=(dataVal, labelVal), n_epoch=100, batch_size=8, show_metric=True, validation_batch_size=8)
model.save('logs/model')
pickle.dump({'words':words, 'categories':categories, 'train_x':train_x, 'train_y':train_y},open('src/data/training_data','wb'))
