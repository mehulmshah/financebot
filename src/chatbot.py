#!/usr/bin/env python3
# chatbot.py
# This file contains the main flow for the actual system. The trained model
# is loaded and functionality of the chatbot is set. Then the user can converse
# with the chatbot.

import nltk
from nltk.stem.lancaster import LancasterStemmer
from util.responseUtil import conversationFlow, unknownFlow
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle

data = pickle.load(open('src/data/training_data','rb'))
words = data['words']
categories = data['categories']
train_x = data['train_x']
train_y = data['train_y']
ERROR_THRESHOLD = 0.65
# import our chat-bot intents file
with open('src/data/conversation.json') as f:
    intents = json.load(f)

net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)
model = tflearn.DNN(net, tensorboard_dir='logs')
model.load('logs/model')

stemmer = LancasterStemmer()

def bagOfWords(sentence, words, debug=False):
    # tokenize the pattern
    tokenized = nltk.word_tokenize(sentence)
    [stemmer.stem(word.lower()) for word in tokenized]
    # bag of words
    bag = [0]*len(words)
    for s in tokenized:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if debug:
                    print ("found in: %s" % w)
    return(np.array(bag))

def classify(sentence, debug=False):
    results = model.predict([bagOfWords(sentence, words)])[0]
    if debug:
        print("results: {}".format(list(zip(categories, results))))
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1])
    return_list = []
    if results:
        return_list.append((categories[results[-1][0]], results[-1][1]))
    # return tuple of intent and probability
    return return_list


def response(sentence, debug=False):
    results = classify(sentence,debug)
    if results:
        for cat in intents['categorySet']:
            if cat['category'] == results[0][0]:
                if cat['category'] == 'balance' or cat['category'] == 'budgeting' or cat['category'] == 'housing':
                    botResponse = conversationFlow(cat['category'], sentence)
                else:
                    botResponse = random.choice(cat['responseSet'])
    else:
        botResponse = unknownFlow()

    return print('\033[94m' + botResponse + '\033[0m')

def main():
    output = [('temp','temp')]
    while (output[0][0] != 'exit'):
        userRequest = input('--> ')
        output = classify(userRequest)
        if not output:
            output = [('temp','temp')]
        response(userRequest, debug=True)

main()
