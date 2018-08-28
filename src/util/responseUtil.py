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

BANK_ACCOUNTS = {'boa':{
                        'checking':{
                                    'lastFour':9898,
                                    'balance':9000
                                    },
                        'savings':{
                                    'balance':100000
                                  }
                       },
                 'chase':{
                          'savings':{
                                     'lastFour':9898,
                                     'balance':80000
                                     }
                         },
                 'creditCard':-15000
                }

with open('src/data/conversation.json') as f:
    intents = json.load(f)

def conversationFlow(category, userRequest):
    words = word_tokenize(userRequest.lower().replace('bank of america','boa'))
    POS = pos_tag(words)
    if category == 'balance':
        botResponse = balanceFlow(words)
    elif category == 'budgeting':
        botResponse = budgetingFlow(words, POS)
    elif category == 'housing':
        botResponse = housingFlow(words, POS)
    return botResponse

def balanceFlow(words):
    bank = account = ""
    if 'boa' in words:
        bank = 'boa'
    elif 'chase' in words:
        bank = 'chase'
        account = 'savings'
    if 'checking' in words:
        account = 'checking'
        bank = 'boa'
    if 'savings' in words:
        account = 'savings'
    while not bank or not account:
        if bank and not account:
            account = input('\033[94m' + intents['categorySet'][0]['responseSet'][0]['whichAccount'] + '\033[0m\n--> ')
        if account and not bank:
            bank = input('\033[94m' + intents['categorySet'][0]['responseSet'][0]['whichBank'] + '\033[0m\n--> ').lower()
    return intents['categorySet'][0]['responseSet'][0]['balance'].format(BANK_ACCOUNTS[bank][account]['balance'])

def budgetingFlow(words, POS):
    print('c')

def housingFlow(words, POS):
    for index,item in enumerate(POS):
        if 'CD' in item:
            cost = item[0]



def unknownFlow():
    return random.choice(intents['categorySet'][3]['responseSet'])
