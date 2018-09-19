#!/usr/bin/env python3
# chatbot.py
# This file contains the main flow for the actual system. The trained model
# is loaded and functionality of the chatbot is set. Then the user can converse
# with the chatbot.

import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from util.dataUtil import getBalanceData, getBudgetingData, getHousingData, getNegExampleData
from nltk.corpus import stopwords
from util.responseUtil import conversationFlow, unknownFlow
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle
import argparse

# init objects and load in training data
stemmer = LancasterStemmer()
processedData = pickle.load(open('src/data/processedDataPickle','rb'))
words = processedData['words']
categories = processedData['categories']
dataTrain = processedData['dataTrain']
labelTrain = processedData['labelTrain']

ERROR_THRESHOLD = 80

with open('src/data/intents.json') as f:
    intents = json.load(f)

net = tflearn.input_data(shape=[None, len(dataTrain[0])])
net = tflearn.fully_connected(net, 10)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 10)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, len(labelTrain[0]), activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')
model = tflearn.DNN(net, tensorboard_dir='logs')
model.load('logs/model')

# given a query, return an array of sentence in bag of words format
def bagOfWords(sentence, words, debug=False):
    stop_words = set(stopwords.words('english'))
    tokenized = nltk.word_tokenize(sentence)
    tokenized = [stemmer.stem(w.lower()) for w in tokenized if w.isalnum() and not w in stop_words]
    # bag of words
    bag = [0]*len(words)
    for toke_word in tokenized:
        for index,word in enumerate(words):
            if word == toke_word:
                bag[index] = 1
    return(np.array(bag))

# use DNN model to predict category for a query and return highest prob if above
# ERROR_THRESHOLD
def classify(sentence, debug=False):
    results = model.predict([bagOfWords(sentence, words)])[0]
    results = list(np.round(100*results, 2))
    if debug:
        print("results: {}".format(list(zip(categories, results))))
    if max(results) > ERROR_THRESHOLD:
        return results.index(max(results))
    else:
        return None

# classify a request, and then send it to responseUtil to obtain appropriate
# response
def response(sentence, debug):
    classIndex = classify(sentence,debug)
    if debug:
        print(categories[classIndex])
    #if classIndex === None or classIndex == 3:
    #    botResponse = unknownFlow()
    #else:
    #    botResponse = conversationFlow(intents['categorySet'][classIndex], sentence, debug)
    #return print('\033[94m' + botResponse + '\033[0m')

def validInput(request):
    while not (request and request.strip()):
        request = input('--> ')
    return request

# Load up chatbot and stay alive until user quits program or error is encountered
def main():
    parser = argparse.ArgumentParser(description='Finance chatbot!')
    parser.add_argument('--debug', type=bool, help='Toggle debug mode (default False)')
    args = parser.parse_args()
    if args.debug:
        DEBUG = args.debug
    else:
        DEBUG = False
    userRequest = validInput(input('--> '))
    while (userRequest != "exit"):
        response(userRequest, debug=DEBUG)
        userRequest = validInput(input('--> '))
    print("\033[94m Bye! See you next time. \033[0m\n")

if __name__ == '__main__':
    main()
