#!/usr/bin/env python3
# getTrainingData.py
# This file will obtain training data for categorization of user requests.
# Two categories will obtain data from Reddit, and the third (check balance)
# will be provided by me, as it is not something that is often asked online,
# and is less variable in terms of how it is asked.

import praw
import os

redditObj = praw.Reddit(client_id='9wHjRUw5P54JpA', \
                     client_secret='SYx98S03esOePq05LGLwcLcxf50', \
                     user_agent='financeScraper', \
                     username='financeScraperBot', \
                     password='botSCRAPERfinance')

queryDict = {'budgetingDict': {'flair':'budgeting','queries':['what should my budget look like?', 'how much can I budget']},
             'housingDict': {'flair':'housing','queries':['afford house', 'can I buy a home']},
             'entityDict': {'queries':['boa', 'Chase']}
            }
BALANCE_DATA_PATH = 'src/data/checkBalanceData.txt'

def getBalanceData():
    with open(BALANCE_DATA_PATH) as f:
        balanceData = [line.rstrip('\n') for line in f]
    return balanceData

def getBudgetingData():
    budgetingData = getRedditData(queryDict['budgetingDict']['flair'], queryDict['budgetingDict']['queries'])
    for post in budgetingData:
        if 'budget' not in post and 'budgeting' not in post:
            budgetingData.remove(post)
    return budgetingData[:50]

def getHousingData():
    housingData = getRedditData(queryDict['housingDict']['flair'], queryDict['housingDict']['queries'])
    return housingData[:50]

def getTrainData():
    return getBalanceData() + getHousingData() + getBudgetingData()

def getRedditData(flair, queryList):
    titles = []
    for query in queryList:
        for post in redditObj.subreddit('personalfinance').search('flair:"{}" {}'.format(flair, query),limit=None):
            titles.append(post.title)

    return titles

def getEntityData(limit):
    titles = []
    queryList = queryDict['entityDict']['queries']
    for query in queryList:
        for post in redditObj.subreddit('personalfinance').search('{}'.format(query),limit=limit):
            titles.append(post.title)
    return titles

def getPriceData(limit):
    titles = []
    for post in redditObj.subreddit('budgetnerds').hot(limit=limit):
        titles.append(post.title)
    return titles
