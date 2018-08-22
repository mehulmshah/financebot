#!/usr/bin/env python3
# getTrainingData.py
# This file will obtain training data for categorization of user requests.
# Two categories will obtain data from Reddit, and the third (check balance)
# will be provided by me, as it is not something that is often asked online,
# and is less variable in terms of how it is asked.

import praw
from getRedditData import getData

balanceFileName = 'checkBalanceData.txt'
housingDict = {'flair':'housing','queries':['afford house', 'can I buy a house']}
budgetingDict = {'flair':'budgeting','queries':['how much can I save', 'how much can I budget']}
redditObj = praw.Reddit(client_id='9wHjRUw5P54JpA', \
                     client_secret='SYx98S03esOePq05LGLwcLcxf50', \
                     user_agent='financeScraper', \
                     username='financeScraperBot', \
                     password='botSCRAPERfinance')


housingData = getData(redditObj, housingDict['flair'], housingDict['queries'])
budgetingData = getData(redditObj, budgetingDict['flair'], budgetingDict['queries'])

with open(balanceFileName) as f:
    
