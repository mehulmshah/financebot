#!/usr/bin/env python3
# trainNER.py
# This file will pull data, tokenize it, and train it using Stanford's NER
# library. I will be manually tagging the dataset. This will be used for
# questions involving banks (i.e BoA, Bank of America, Chase)

import nltk
import subprocess
from util.dataUtil import getEntityData

# create train and test datasets into TSVs
path = 'src/data/'
trainfile = 'entity.txt'
testfile = 'entityTest.txt'
traindata = getEntityData(25)
testdata = getEntityData(50)

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
subprocess.run('java -cp ../stanford-ner-2018-02-27/stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -prop src/data/entity.prop',shell=True)

# run Stanford NER on test set to see performance for personal use
subprocess.run('java -cp ../stanford-ner-2018-02-27/stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier src/data/bank-ner-model.ser.gz -testFile src/data/entityTest.tsv',shell=True)
