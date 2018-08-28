#!/usr/bin/env python3
# responseUtil.py
# This file contains helper functions to return a bot response given a query
# and a category

from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tag.stanford import StanfordNERTagger
import json
import random

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

PRICE_CONVERSION = {'million': 1000000,
                    'm': 1000000,
                    'thousand': 1000,
                    'k': 1000
                   }

with open('src/data/conversation.json') as f:
    intents = json.load(f)

def conversationFlow(category, userRequest):
    word_tokens = word_tokenize(userRequest)
    if category == 'balance':
        botResponse = balanceFlow(word_tokens)
    elif category == 'budgeting':
        botResponse = budgetingFlow(word_tokens)
    elif category == 'housing':
        botResponse = housingFlow(word_tokens)
    return botResponse

def balanceFlow(word_tokens):
    bank = account = ""
    if 'checking' in word_tokens:
        account = 'checking'
        bank = 'BoA'
    elif 'savings' in word_tokens:
        account = 'savings'

    bank = getBank(word_tokens)

    if not account:
        account = input('\033[94m' + intents['categorySet'][0]['responseSet'][0]['whichAccount'] + '\033[0m\n--> ')
    while not bank:
        bank = getBank(input('\033[94m' + intents['categorySet'][0]['responseSet'][0]['whichBank'] + '\033[0m\n--> ').split())

    if bank == 'chase' and account == 'checking':
        return "You don't have a checking account with Chase."
    else:
        return intents['categorySet'][0]['responseSet'][0]['balance'].format(bank, account, BANK_ACCOUNTS[bank.lower()][account]['balance'])

def budgetingFlow(word_tokens):
    return (intents['categorySet'][1]['responseSet'][0] +
           "income: +{}/mo\n".format(PERSONAL_EQUITY['income']) +
           "rent: -{}/mo\n".format(PERSONAL_EQUITY['rent']) +
           "utilities and groceries: {}/mo\n".format(PERSONAL_EQUITY['utilGroceries']) +
           "shopping: {}/mo\n".format(PERSONAL_EQUITY['shopping']) +
           "dining: {}/mo\n".format(PERSONAL_EQUITY['dining']) +
           "Remaining budget: {}/mo! (Try to put this into your savings account)".format(sum(PERSONAL_EQUITY.values())))

def housingFlow(word_tokens):
    housePrice = getPrice(word_tokens)
    if not housePrice:
        housePrice = getPrice(input("I'm sorry, I didn't detect a price... how much does your dream place cost? "))

    if housePrice > 785000:
        response = intents['categorySet'][2]['responseSet'][0]['no']
        response += " You can put a 20% downpayment of {}.".format(.2*housePrice)
    else:
        response = intents['categorySet'][2]['responseSet'][0]['yes']
        response += " That will require a 20% downpayment of {}.".format(.2*housePrice)
        response += " At your current rate of savings (+500/mo), you can afford the downpayment in {} months!".format((.2*housePrice - 157000)/500)

    return response
    
def getBank(word_tokens):
    bank = ""
    st = StanfordNERTagger('src/data/bank-ner-model.ser.gz', '../stanford-ner-2018-02-27/stanford-ner.jar')
    tagged_words = st.tag(word_tokens)
    print(tagged_words)
    for tag in tagged_words:
        if 'C-ORG' in tag:
            bank = 'chase'
        elif 'B-ORG' in tag:
            bank = 'boa'

    return bank

def getPrice(word_tokens):
    price = []
    multiplier = 1
    housePrice = 0

    st = StanfordNERTagger('src/data/price-ner-model.ser.gz', '../stanford-ner-2018-02-27/stanford-ner.jar')
    tagged_words = st.tag(word_tokens)
    for tag in tagged_words:
        if 'B-PRICE' in tag:
            price.append(tag[0])

    for item in price:
        item = item.lower()
        if '$' in item:
            item = item.replace('$','')
        for key in PRICE_CONVERSION:
            if key in item:
                multiplier = PRICE_CONVERSION[key]
                item = item.replace(key,'')
        if item.replace('.','',1).isdigit():
            housePrice = float(item)

    return housePrice * multiplier

def unknownFlow():
    responseSet = ["Sorry, I did not understand that. Can you try re-phrasing?", "I'm still in v0.1, I don't think I can help with that..."]
    return random.choice(responseSet)
