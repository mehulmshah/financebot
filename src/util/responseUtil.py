#!/usr/bin/env python3
# responseUtil.py
# This file contains helper functions to return a bot response given a query
# and a category

from nltk import word_tokenize
from nltk.tag import StanfordNERTagger
import json
import random

# highest house price Mary can afford a downpayment for (see writeup)
MAX_AFFORDABLE_PRICE = 758000

# dict containing Mary's income and expenses for budgeting
PERSONAL_EQUITY = {'income':6000,
                  'rent':-3000,
                  'utilGroceries':-1000,
                  'shopping':-1000,
                  'dining':-500}

# dict containing Mary's bank account balances (should ideally be via bank apis)
BANK_ACCOUNTS = {'BoA':{
                        'checking':{
                                    'lastFour':9898,
                                    'balance':9000
                                    },
                        'savings':{
                                    'balance':100000
                                  }
                       },
                 'Chase':{
                          'savings':{
                                     'lastFour':9898,
                                     'balance':80000
                                     }
                         },
                 'creditCard':-15000
                }

# dict containing conversions of written number to int
PRICE_CONVERSION = {'million': 1000000,
                    'm': 1000000,
                    'thousand': 1000,
                    'k': 1000
                   }

# load up intents json
with open('src/data/intents.json') as f:
    intents = json.load(f)

# go to correct conversational flow submethod
def conversationFlow(category, userRequest, debug):
    word_tokens = word_tokenize(userRequest)
    if category == 'balance':
        botResponse = balanceFlow(word_tokens, debug)
    elif category == 'budgeting':
        botResponse = budgetingFlow(word_tokens)
    elif category == 'housing':
        botResponse = housingFlow(word_tokens, debug)
    return botResponse

# if a balance question, obtain bank and account, and then fetch balance from dict if available
def balanceFlow(word_tokens, debug):
    bank = getBank(word_tokens, debug)
    account = getAccount(word_tokens)

    while account != 'checking' and account != 'savings':
        account = getAccount(input('\033[94m' + intents['categorySet'][0]['responseSet'][0]['whichAccount'] + '\033[0m\n--> ').split())
    while not bank:
        bank = getBank(input('\033[94m' + intents['categorySet'][0]['responseSet'][0]['whichBank'] + '\033[0m\n--> ').split(),debug)

    if bank == 'Chase' and account == 'checking':
        return "You don't have a checking account with Chase."
    else:
        return intents['categorySet'][0]['responseSet'][0]['balance'].format(bank, account, BANK_ACCOUNTS[bank][account]['balance'])

# Function containing the budget response
def budgetingFlow(word_tokens):
    return (intents['categorySet'][1]['responseSet'][0] +
           "income: +{}/mo\n".format(PERSONAL_EQUITY['income']) +
           "rent: {}/mo\n".format(PERSONAL_EQUITY['rent']) +
           "utilities and groceries: {}/mo\n".format(PERSONAL_EQUITY['utilGroceries']) +
           "shopping: {}/mo\n".format(PERSONAL_EQUITY['shopping']) +
           "dining: {}/mo\n".format(PERSONAL_EQUITY['dining']) +
           "Remaining budget: {}/mo! (Try to put this into your savings account)".format(sum(PERSONAL_EQUITY.values())))

# Function containing house affordability flow
def housingFlow(word_tokens, debug):
    housePrice = getPrice(word_tokens, debug)
    if not housePrice:
        housePrice = int(input("I'm sorry, I didn't detect a price... how much does your dream place cost? ").replace('$',''))

    if housePrice < MAX_AFFORDABLE_PRICE:
        response = intents['categorySet'][2]['responseSet'][0]['yes']
        response += " You can put a 20% downpayment of ${}.".format(int(.2*housePrice))
    else:
        response = intents['categorySet'][2]['responseSet'][0]['no']
        response += " A ${} home will require a 20% downpayment of ${}.".format(int(housePrice), int(.2*housePrice))
        response += " At your current rate of savings ($500/mo), you can afford the downpayment in {} years!".format(int((.2*housePrice - 157000)/(500*12)))

    return response

# Helper function using NER model to obtain Bank name from query
def getBank(word_tokens, debug):
    bank = ""
    st = StanfordNERTagger('ner/bank-ner-model.ser.gz', 'ner/stanford-ner.jar')
    tagged_words = st.tag(word_tokens)
    if debug:
        print(tagged_words)
    for tag in tagged_words:
        if 'C-ORG' in tag:
            bank = 'Chase'
        elif 'B-ORG' in tag:
            bank = 'BoA'

    return bank

# Helper function to extract account type from user query
def getAccount(word_tokens):
    if 'checking' in word_tokens:
        return 'checking'
    elif 'savings' in word_tokens:
        return 'savings'
    else:
        return ''
# Helper function using NER model to obtain price of house from query
def getPrice(word_tokens, debug):
    price = []
    multiplier = 1
    housePrice = 0

    st = StanfordNERTagger('ner/price-ner-model.ser.gz', 'ner/stanford-ner.jar')
    tagged_words = st.tag(word_tokens)
    for tag in tagged_words:
        if 'B-PRICE' in tag:
            price.append(tag[0])

    if debug:
        print(price)

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

# Default answers if the query does not fit any of the above 3 categories
def unknownFlow():
    responseSet = ["Sorry, I did not understand that. Can you try re-phrasing?", "I'm still in v0.1, I don't think I can help with that..."]
    return random.choice(responseSet)
