#!/usr/bin/env python3
# responseUtil.py
# This file contains helper functions to return a bot response given a query
# and a category

from nltk import word_tokenize, pos_tag, ne_chunk
import json

PERSONAL_EQUITY = {'income':6000,
                  'rent':-3000,
                  'utilGroceries':-1000,
                  'shopping':-1000,
                  'dining':-500}

BANK_ACCOUNTS = {'boaChecking':{'lastFour':9898,'balance':9000},
                 'boaSavings':{'balance':100000},
                 'chaseSavings':{'lastFour':9898,'balance':80000},
                 'creditCard'{'balance':-15000}
                }

with open('src/data/conversation.json') as f:
    intents = json.load(f)

def conversationFlow(category, userRequest):
    if category == 'balance':
        botResponse = balanceFlow(userRequest,category)
    elif category == 'budgeting':
        botResponse = budgetingFlow(userRequest,category)
    elif category == 'housing':
        botResponse = housingFlow(userRequest,category)
    return botResponse

def balanceFlow(userRequest,category):
    partsOfSpeech = pos_tag(word_tokenize(userRequest))

def  budgetingFlow(userRequest):


def housingFlow(userRequest):
