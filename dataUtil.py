#!/usr/bin/env python3
# getTrainingData.py
# This file will obtain training data for categorization of user requests.
# Two categories will obtain data from Reddit, and the third (check balance)
# will be provided by me, as it is not something that is often asked online,
# and is less variable in terms of how it is asked.

import praw

def getTrainData():
    # init variables for text file, and Reddit object
    # PRAW docs: https://praw.readthedocs.io/en/latest/code_overview/reddit_instance.html
    balanceFileName = 'checkBalanceData.txt'
    housingDict = {'flair':'housing','queries':['afford house', 'can I buy a house']}
    budgetingDict = {'flair':'budgeting','queries':['what is my budget', 'how much can I budget']}
    redditObj = praw.Reddit(client_id='9wHjRUw5P54JpA', \
                         client_secret='SYx98S03esOePq05LGLwcLcxf50', \
                         user_agent='financeScraper', \
                         username='financeScraperBot', \
                         password='botSCRAPERfinance')
    # Three lists of data points, one for each category. A data point is a String
    # representing a user request
    housingData = getRedditData(redditObj, housingDict['flair'], housingDict['queries'])
    budgetingData = getRedditData(redditObj, budgetingDict['flair'], budgetingDict['queries'])
    with open(balanceFileName) as f:
        balanceData = [line.rstrip('\n') for line in f]

    return balanceData + budgetingData[:100] + housingData[:100]


def getRedditData(redditObj, flair, queryList):
    titles = []
    for query in queryList:
        for post in redditObj.subreddit('personalfinance').search('flair:"{}" {}'.format(flair, query),limit=None):
            titles.append(post.title)

    return titles
