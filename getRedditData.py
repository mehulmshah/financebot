#!/usr/bin/env python3
# getRedditData.py
# This file will scrape /r/personalfinance (a subreddit on Reddit for posts
# that ask questions relevant to 'house affordability' and 'budgeting'

import praw

def getData(redditObj, flair, queryList):
    titles = []
    for query in queries:
        for post in redditObj.subreddit('personalfinance').search('flair:"{}" {}'.format(flair, query),limit=None):
            titles.append(post.title)

    return titles
