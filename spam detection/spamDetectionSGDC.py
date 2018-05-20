#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 09:42:20 2018
"""


import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
import numpy as np 


data = pd.read_excel('training_data.xlsx')
#print(data.head())
data = data.dropna()
data = data.sample(frac=1)
X = data.text
y= data.label
training_document, testing_document, training_label, testing_label = train_test_split(X, y, test_size=0.10)

text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(max_iter=10,loss='log')),])
text_clf = text_clf.fit(training_document, training_label)


predicted = text_clf.predict(testing_document)
print(np.mean(predicted == testing_label))