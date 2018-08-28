#!/usr/bin/env python3
# trainPriceNER.py
# This file will pull data, tokenize it, and train it using Stanford's NER
# library. I will be manually tagging the dataset. This will be used for
# questions involving price (i.e $2.5M house, $700k, etc.)

import nltk
import subprocess
from util.dataUtil import getPriceData

# create train and test datasets into TSVs
path = 'data/'
trainfile = 'price.tsv'
testfile = 'priceTest.tsv'
traindata = getPriceData(25)
testdata = getPriceData(50)

# save to file and then manually annotate w/ NER tags
with open(path + trainfile, 'w') as f:
    for item in traindata:
        for word in item.split():
            f.write("%s 0\n" % word)

with open(path + testfile, 'w') as f:
    for item in testdata:
        for word in item.split():
            f.write("%s 0\n" % word)

# train Stanford NER on the train set
subprocess.run('java -cp ../stanford-ner-2018-02-27/stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -prop src/data/price.prop',shell=True)
