#!/usr/bin/env python3
# trainModel.py
# This file will load the conversational framework for the chatbot, and train
# a model in Tensorflow to recognize what category a user request is in.

import nltk
from nltk.stem.lancaster import LancasterStemmer
from src.util.dataUtil import getBalanceData, getBudgetingData, getHousingData
import numpy as np
from sklearn import linear_model
import random
import json
import pickle

with open('src/data/conversation.json') as f:
    intents = json.load(f)

intents['categorySet'][0]['trainData'] += getBalanceData()
intents['categorySet'][1]['trainData'] += getBudgetingData()
intents['categorySet'][2]['trainData'] += getHousingData()

words = []
categories = []
sentences = []

print('creating set of sentences, words, and categories...')
for topic in intents['categorySet']:
    for datapoint in topic['trainData']:
        w = nltk.word_tokenize(datapoint)
        words.extend(w)
        sentences.append((w, topic['category']))
        if topic['category'] not in categories:
            categories.append(topic['category'])

stemmer = LancasterStemmer()
words = [stemmer.stem(w.lower()) for w in words if w.isalnum()]
words = list(dict.fromkeys(words))

print('creating training set...')
trainingSet = []
for sentence in sentences:
    bagOfWords = []
    tokenizedWords = sentence[0]
    tokenizedWords = [stemmer.stem(word.lower()) for word in tokenizedWords]
    for w in words:
        bagOfWords.append(1) if w in tokenizedWords else bagOfWords.append(0)
    output = [0 for i in range(len(categories))]
    output[categories.index(sentence[1])] = 1
    trainingSet.append([bagOfWords, output])

random.shuffle(trainingSet)
trainingSet = np.array(trainingSet)
train_x = list(trainingSet[:400,0])
train_y = []
for item in list(trainingSet[:400,1]):
    train_y.append(item.index(1))

test_x = list(trainingSet[401:,0])
test_y = []
for item in list(trainingSet[401:,1]):
    test_y.append(item.index(1))

logreg = linear_model.LogisticRegression(C=10)
logreg.fit(train_x, train_y)
z = list(logreg.predict(test_x))
numRight = 0
for i in range(len(z)):
    if z[i] == test_y[i]:
        numRight += 1

print("Accuracy:", 100*numRight/len(test_y))

model.save('logs/model')
pickle.dump({'words':words, 'categories':categories, 'train_x':train_x, 'train_y':train_y},open('src/data/training_data','wb'))
