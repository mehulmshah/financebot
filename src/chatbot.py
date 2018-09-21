#!/usr/bin/env python3
# chatbot.py
# This file contains the main flow for the actual system. The trained model
# is loaded and functionality of the chatbot is set. Then the user can converse
# with the chatbot.

from util.responseUtil import conversationFlow, unknownFlow
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import os
import pickle


category_index = {"balance": 0, "budgeting": 1, "housing": 2}
category_reverse_index = dict((y, x) for (x, y) in category_index.items())

THRESHOLD = 0.80
MAX_SEQUENCE_LENGTH = 20

model1 = load_model('../logs/model1.h5')
model2 = load_model('../logs/model2.h5')
model3 = load_model('../logs/model3.h5')
tokenDict = pickle.load(open('data/tokenizer', 'rb'))
tokenizer = tokenDict['tokenizer']


def classify(sentence, the_model):
    seq = tokenizer.texts_to_sequences([sentence])
    pad_seq = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    probabilities = the_model.predict(pad_seq, verbose=0)
    probabilities = probabilities[0]
    print("-" * 10)
    class_name = category_reverse_index[the_model.predict_classes(pad_seq, verbose=0)[0]]
    if max(probabilities) > THRESHOLD:
        print("Predicted category: ", class_name)
    else:
        class_name = "unknown"
        print("Predicted category: Unknown")
    print("-" * 10)
    probabilities = the_model.predict(pad_seq, verbose=0)
    probabilities = probabilities[0]
    print("Balance: {}\nBudgeting: {}\nHousing: {}\n".format(probabilities[category_index["balance"]],
                                                             probabilities[category_index["budgeting"]],
                                                             probabilities[category_index["housing"]]))
    return class_name


def response(sentence, the_model):
    class_name = classify(sentence, the_model)

    if class_name == "unknown":
        botResponse = unknownFlow()
    else:
        botResponse = conversationFlow(class_name, sentence)
    return print('\033[94m' + botResponse + '\033[0m')


def validInput(request):
    while not (request and request.strip()):
        request = input('--> ')
    return request


# Load up chatbot and stay alive until user quits program or error is encountered
def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    userRequest = validInput(input('--> '))
    while userRequest != "exit":
        response(userRequest, model1)
        response(userRequest, model2)
        response(userRequest, model3)
        userRequest = validInput(input('--> '))
    print("\033[94m Bye! See you next time. \033[0m\n")


if __name__ == '__main__':
    main()
