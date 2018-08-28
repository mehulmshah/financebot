#!/usr/bin/env python3
# chatbot.py
# This file contains the main flow for the actual system. The trained model
# is loaded and functionality of the chatbot is set. Then the user can converse
# with the chatbot.

import nltk
from nltk.stem.lancaster import LancasterStemmer
from util.responseUtil import conversationFlow, unknownFlow
import speech_recognition as sr
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle
import argparse

# init objects and load in training data
stemmer = LancasterStemmer()
r = sr.Recognizer()
data = pickle.load(open('src/data/training_data','rb'))
words = data['words']
categories = data['categories']
train_x = data['train_x']
train_y = data['train_y']

# confidence level needed to categorize request
ERROR_THRESHOLD = 0.65

# import intents file
with open('src/data/conversation.json') as f:
    intents = json.load(f)

# load up model from logs for prediction
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)
model = tflearn.DNN(net, tensorboard_dir='logs')
model.load('logs/model')

# given a query, return an array of sentence in bag of words format
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

# use DNN model to predict category for a query and return highest prob if above
# ERROR_THRESHOLD
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

# classify a request, and then send it to responseUtil to obtain appropriate
# response
def response(sentence, debug):
    results = classify(sentence,debug)
    if results:
        for cat in intents['categorySet']:
            if cat['category'] == results[0][0]:
                if cat['category'] == 'balance' or cat['category'] == 'budgeting' or cat['category'] == 'housing':
                    botResponse = conversationFlow(cat['category'], sentence, debug)
                else:
                    botResponse = random.choice(cat['responseSet'])
    else:
        botResponse = unknownFlow()

    return print('\033[94m' + botResponse + '\033[0m')

# Check to see if user wishes to type or speak into mic to chat
def getRequest(isTyping):
    if isTyping:
        return input('--> ')
    else:
        with sr.Microphone() as source:
            print("Say something --> ")
            audio = r.listen(source)

        try:
            print("--> " + r.recognize_google(audio))
        except sr.UnknownValueError:
            print("\033[94m I could not understand audio... please quit and try again via chat \033[0m")
        return r.recognize_google(audio)

# Load up chatbot and stay alive until user quits program or error is encountered
def main():
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--debug', type=bool, help='Toggle debug mode (default False)')
    args = parser.parse_args()
    if args.debug:
        DEBUG = args.debug
    else:
        DEBUG = False

    isTyping = int(input("Press 0 to talk via voice, 1 to talk via chat: "))
    userRequest = getRequest(isTyping)
    while (userRequest != "exit"):
        response(userRequest, debug=DEBUG)
        userRequest = getRequest(isTyping)

    print("\033[94m Bye! See you next time. \033[0m")

if __name__ == '__main__':
    main()
