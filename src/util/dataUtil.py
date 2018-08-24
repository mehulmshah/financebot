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

BALANCE_DATA_PATH = 'src/data/checkBalanceData.txt'

def getBalanceData():
    with open(BALANCE_DATA_PATH) as f:
        balanceData = [line.rstrip('\n') for line in f]

    return balanceData

def getBudgetingData():
    budgetingDict = {'flair':'budgeting','queries':['what is my budget', 'how much can I budget']}
    budgetingData = getRedditData(redditObj, budgetingDict['flair'], budgetingDict['queries'])
    return budgetingData

def getHousingData():
    housingDict = {'flair':'housing','queries':['afford house', 'can I buy a house']}
    housingData = getRedditData(redditObj, housingDict['flair'], housingDict['queries'])
    return housingData

def getTrainData():
    return getBalanceData() + getHousingData() + getBudgetingData()

def getRedditData(redditObj, flair, queryList):
    titles = []
    for query in queryList:
        for post in redditObj.subreddit('personalfinance').search('flair:"{}" {}'.format(flair, query),limit=None):
            titles.append(post.title)

    return titles
