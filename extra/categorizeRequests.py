#!/usr/bin/env python3
# categorizeRequests.py
# This file will train a model to take incoming user requests and categorize
# them in one of three ways:
# 1. Check balance (i.e. how much is in my bank account)
# 2. Budgeting (i.e. how much room is in my budget this month?)
# 3. House Affordability (i.e. can I afford a $2 million house?)

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from dataUtil import getTrainData

n_samples = 254
n_features = 400
n_components = 3
n_top_words = 5

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

trainingData = getTrainData()
trainingData.append(input("Please enter a request for the bot: "))
# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')

tf = tf_vectorizer.fit_transform(trainingData)

lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0).fit(tf[:-1])

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)

print("\n\nIndices for each topic for LDA:\n")
requestProbabilities = lda.transform(tf)
topicNum = requestProbabilities[-1].argmax(axis=0)
print('Looks like you are asking about Topic {}'.format(topicNum))
