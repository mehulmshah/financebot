#!/usr/bin/env python3
# categorizeRequests.py
# This file will train a model to take incoming user requests and categorize
# them in one of three ways:
# 1. Check balance (i.e. how much is in my bank account)
# 2. Budgeting (i.e. how much room is in my budget this month?)
# 3. House Affordability (i.e. can I afford a $2 million house?)

from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from getTrainingData import getTrainData

n_samples = 246
n_features = 1000
n_components = 3
n_top_words = 5

# get vector of training data from getTrainingData.py
trainingData = getTrainData()
print("received training data")

# Use tf-idf features for NMF.
print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95,
                                   max_features=n_features,
                                   stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(trainingData)

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95,
                                max_features=n_features,
                                stop_words='english')

tf = tf_vectorizer.fit_transform(trainingData)

# Fit the NMF model
nmf = NMF(n_components=n_components, random_state=1,
          beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
          l1_ratio=.5)

nmf.fit(tfidf)



lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda.fit(tf)

userRequest = input("Please enter a request: ")
trainingData.append(userRequest)
test_tf = tf_vectorizer.fit_transform(trainingData)
lda.transform(test_tf)
