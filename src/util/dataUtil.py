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

# dict containing params for reddit searches
queryDict = {'budgetingDict': {'flair':'budgeting','queries':['what should my budget look like?', 'how much can I budget']},
             'housingDict': {'flair':'housing','queries':['afford house', 'can I buy a home']},
             'entityDict': {'queries':['boa', 'Chase']}
            }

# path to manually annotated dataset for checking balance
BALANCE_DATA_PATH = 'src/data/checkBalanceData.txt'

# read in created dataset for training data for balance category
def getBalanceData():
    with open(BALANCE_DATA_PATH) as f:
        balanceData = [line.rstrip('\n') for line in f]
    return balanceData

# scrape /r/personalfinance for posts about budgeting
def getBudgetingData():
    budgetingData = getRedditData(queryDict['budgetingDict']['flair'], queryDict['budgetingDict']['queries'])
    for post in budgetingData:
        if 'budget' not in post and 'budgeting' not in post:
            budgetingData.remove(post)
    return budgetingData[:50]

# scrape /r/personalfinance for posts about housing affordability
def getHousingData():
    housingData = getRedditData(queryDict['housingDict']['flair'], queryDict['housingDict']['queries'])
    return housingData[:50]

def getTrainData():
    return getBalanceData() + getHousingData() + getBudgetingData()

# helper method to actually scrape from Reddit
def getRedditData(flair, queryList):
    titles = []
    for query in queryList:
        for post in redditObj.subreddit('personalfinance').search('flair:"{}" {}'.format(flair, query),limit=None):
            titles.append(post.title)

    return titles

# get data for training NER model to recognize banks
def getEntityData(limit):
    titles = []
    queryList = queryDict['entityDict']['queries']
    for query in queryList:
        for post in redditObj.subreddit('personalfinance').search('{}'.format(query),limit=limit):
            titles.append(post.title)
    return titles

# get data for training NER model to recognize prices in queries
def getPriceData(limit):
    titles = []
    for post in redditObj.subreddit('budgetnerds').hot(limit=limit):
        titles.append(post.title)
    return titles
