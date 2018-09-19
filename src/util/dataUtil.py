#!/usr/bin/env python3
# getTrainingData.py
# This file will obtain training data for categorization of user requests.
# Two categories will obtain data from Reddit, and the third (check balance)
# will be provided by me, as it is not something that is often asked online,
# and is less variable in terms of how it is asked.

import praw
import os

# init reddit object for scraping from subreddits
redditObj = praw.Reddit(client_id='9wHjRUw5P54JpA', \
                     client_secret='SYx98S03esOePq05LGLwcLcxf50', \
                     user_agent='financeScraper', \
                     username='financeScraperBot', \
                     password='botSCRAPERfinance')

# helper method to actually scrape from Reddit
def getRedditData(flair, queryList):
    titles = []
    for query in queryList:
        for post in redditObj.subreddit('personalfinance').search('flair:"{}" {}'.format(flair, query),limit=None):
            titles.append(post.title)
    return titles

# dict containing params for reddit searches
queryDict = {'budgetingDict': {'flair':'budgeting','queries':['what should my budget look like?', 'how much can I budget']},
             'housingDict': {'flair':'housing','queries':['afford house']},
             'entityDict': {'queries':['boa', 'Chase']},
             'negExampleDict': {'flair':'auto','queries':['can I afford']}
            }

# path to manually annotated datasets
BALANCE_DATA_PATH = 'src/data/checkBalanceData.txt'
MORTGAGE_FAQ_PATH = 'src/data/mortgageFaqData.txt'
SPENDING_DATA_PATH = 'src/data/spendingData.txt'
NEG_DATA_PATH = 'src/data/negData.txt'

def getNegExampleData():
    autoData = getRedditData(queryDict['negExampleDict']['flair'], queryDict['negExampleDict']['queries'])
    with open(NEG_DATA_PATH) as f:
        otherNegData = [line.rstrip('\n') for line in f]
    return autoData + otherNegData

def getBalanceData():
    with open(BALANCE_DATA_PATH) as f:
        balanceData = [line.rstrip('\n') for line in f]
    return balanceData

def getBudgetingData():
    budgetingData = getRedditData(queryDict['budgetingDict']['flair'], queryDict['budgetingDict']['queries'])
    for post in budgetingData:
        if 'budget' not in post and 'budgeting' not in post:
            budgetingData.remove(post)
    with open(SPENDING_DATA_PATH) as f:
        spendingData = [line.rstrip('\n') for line in f]
    return budgetingData + spendingData

def getHousingData():
    housingData = getRedditData(queryDict['housingDict']['flair'], queryDict['housingDict']['queries'])
    with open(MORTGAGE_FAQ_PATH) as f:
        mortgageFaqData = [line.rstrip('\n') for line in f]
    return housingData + mortgageFaqData

def getEntityData(limit):
    titles = []
    queryList = queryDict['entityDict']['queries']
    for query in queryList:
        for post in redditObj.subreddit('personalfinance').search('{}'.format(query),limit=limit):
            titles.append(post.title)
    return titles
