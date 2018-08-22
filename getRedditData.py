#!/usr/bin/env python3
# getRedditData.py
# This file will scrape relevant subreddits on Reddit (i.e. /r/personalfinance, /r/finance)
# for posts that ask questions relevant to the three categories my system will handle. These
# will be used as training data to generate a model which can classify incoming user requests.

import praw
import pandas as pd
import numpy as np
import datetime as dt

redditObj = praw.Reddit(client_id='9wHjRUw5P54JpA', \
                     client_secret='SYx98S03esOePq05LGLwcLcxf50', \
                     user_agent='financeScraper', \
                     username='financeScraperBot', \
                     password='botSCRAPERfinance')

houseAffordabilityTitles = []
for post in redditObj.subreddit('personalfinance').search('flair:"housing" afford house',limit=None):
    houseAffordabilityTitles.append(post.title)

print(len(houseAffordabilityTitles))
