#!/usr/bin/env python3
# responseUtil.py
# This file contains helper functions to return a bot response given a query
# and a category

from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
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

st = StanfordNERTagger('../ner/english.muc.7class.distsim.crf.ser.gz', '../ner/stanford-ner.jar', encoding='utf-8')


# load up intents json
with open('data/intents.json') as f:
    intents = json.load(f)


# go to correct conversational flow submethod
def conversationFlow(category, userRequest):
    word_tokens = word_tokenize(userRequest)
    if category == 'balance':
        botResponse = balanceFlow(word_tokens)
    elif category == 'budgeting':
        botResponse = budgetingFlow(word_tokens)
    else:
        botResponse = housingFlow(word_tokens)
    return botResponse


# if a balance question, obtain bank and account, and then fetch balance from dict if available
def balanceFlow(word_tokens):
    bank = getBank(word_tokens)
    account = getAccount(word_tokens)

    while account != 'checking' and account != 'savings':
        account = getAccount(input('\033[94m' + "Can you specify an account please? (checking or savings)" + '\033[0m\n--> ').split())
    while not bank:
        bank = getBank(input('\033[94m' + "Can you specify a bank please? + "'\033[0m\n--> ').split())

    if bank == 'Chase' and account == 'checking':
        return "You don't have a checking account with Chase."
    else:
        return "Your {} {} account balance is {}".format(bank, account, BANK_ACCOUNTS[bank][account]['balance'])


# Helper function using NER model to obtain Bank name from query
def getBank(word_tokens):
    bank = ""
    bank_st = StanfordNERTagger('../ner/bank-ner-model.ser.gz', '../ner/stanford-ner.jar')
    tagged_words = bank_st.tag(word_tokens)
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


# Function containing the budget response
def budgetingFlow(word_tokens):
    return ("Your necessities + discretionary budget:\n" +
           "income: +{}/mo\n".format(PERSONAL_EQUITY['income']) +
           "rent: {}/mo\n".format(PERSONAL_EQUITY['rent']) +
           "utilities and groceries: {}/mo\n".format(PERSONAL_EQUITY['utilGroceries']) +
           "shopping: {}/mo\n".format(PERSONAL_EQUITY['shopping']) +
           "dining: {}/mo\n".format(PERSONAL_EQUITY['dining']) +
           "Remaining budget: {}/mo! (Try to put this into your savings account)".format(sum(PERSONAL_EQUITY.values())))


# Function containing house affordability flow
def housingFlow(word_tokens):
    housePrice = getPrice(word_tokens)
    if not housePrice:
        housePrice = getPrice(input("How much is the place?").split())

    if housePrice < MAX_AFFORDABLE_PRICE:
        response = "You definitely can!"
        response += " You can put a 20% downpayment of ${}.".format(int(.2*housePrice))
    else:
        response = "Unfortunately, I don't think that's a good idea."
        response += " A ${} home will require a 20% downpayment of ${}.".format(int(housePrice), int(.2*housePrice))
        response += " At your current rate of savings ($500/mo), you can afford the downpayment in {} years!".format(int((.2*housePrice - 157000)/(500*12)))

    return response


# Helper function using NER model to obtain price of house from query
def getPrice(word_tokens):
    price = []
    tagged_words = st.tag(word_tokens)
    for tag in tagged_words:
        if 'MONEY' in tag:
            price.append(tag[0])

    print("Price:", price)
    if len(price) == 0:
        return None
    else:
        amt = price[1]
        if 'M' in amt:
            return float(price[1].replace('M', ''))*1000000
        if 'K' in amt:
            return float(price[1].replace('K', ''))*1000
        else:
            return float(price[1])


# Default answers if the query does not fit any of the above 3 categories
def unknownFlow():
    responseSet = ["Sorry, I did not understand that. Can you try re-phrasing?", "I'm still in v0.1, I don't think I can help with that..."]
    return random.choice(responseSet)
