#!/usr/bin/env python3
# chatbot.py
# This file contains the main flow for the actual system. The trained model
# is loaded and functionality of the chatbot is set. Then the user can converse
# with the chatbot.

import nltk
from nltk.stem.lancaster import LancasterStemmer
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
ERROR_THRESHOLD = 0.5

# import our chat-bot intents file
with open('src/data/conversation.json') as f:
    intents = json.load(f)

net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)
model = tflearn.DNN(net, tensorboard_dir='logs')
model.load('logs/model')

stemmer = LancasterStemmer()
def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bagOfWords(sentence, words, debug=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if debug:
                    print ("found in: %s" % w)
    return(np.array(bag))

def classify(sentence):
    # generate probabilities from the model
    results = model.predict([bagOfWords(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((categories[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

context = {}
def response(sentence, userID='123', debug=False):
    results = classify(sentence)
    if results:
        for i in intents['categorySet']:
            if i['category'] == results[0][0]:
                # set context for this intent if necessary
                if 'context_set' in i:
                    if debug: print ('context:', i['context_set'])
                    context[userID] = i['context_set']
                # check if this intent is contextual and applies to this user's conversation
                if not 'context_filter' in i or \
                    (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                    if debug: print ('category:', i['category'])
                    # a random response from the intent
                    return print('\033[94m' + random.choice(i['responseSet']) + '\033[0m')


def main():
    output = [('temp','temp')]
    while (output[0][0] != 'exit'):
        userRequest = input('--> ')
        output = classify(userRequest)
        response(userRequest, debug=True)

main()
