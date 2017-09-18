#!/usr/bin/python
# -*- coding:utf-8 -*-
#  20Newsgroup.py
#  Created by HenryLee on 2017/9/2.
#  Copyright © 2017年. All rights reserved.
#  Description :

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from pprint import pprint
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

twenty_train = load_files('train', categories=['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian'])
pprint(list(twenty_train.target_names))

print twenty_train.filenames[68:69]
print twenty_train.target

count_vect = CountVectorizer(decode_error='ignore')
X_train_counts = count_vect.fit_transform(twenty_train.data)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

import pandas as pd
print pd.DataFrame(X_train_counts.toarray(), columns=count_vect.get_feature_names()).head(10)

print '\nNearestCentroid :'
from sklearn.neighbors.nearest_centroid import NearestCentroid

clf = NearestCentroid().fit(X_train_tfidf, twenty_train.target)

doc_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(doc_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predict = clf.predict(X_new_tfidf)

for doc, category in zip(doc_new, predict):
    print '%s -> %s' % (doc, twenty_train.target_names[category])


print '\nMultinomialNB :'
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

predict = clf.predict(X_new_tfidf)

for doc, category in zip(doc_new, predict):
    print '%s -> %s' % (doc, twenty_train.target_names[category])


print '\nKNeighborsClassifier :'
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(15).fit(X_train_tfidf, twenty_train.target)

predict = clf.predict(X_new_tfidf)

for doc, category in zip(doc_new, predict):
    print '%s -> %s' % (doc, twenty_train.target_names[category])


print '\nsvm :'
from sklearn import svm

clf = svm.SVC(kernel='linear').fit(X_train_tfidf, twenty_train.target)

predict = clf.predict(X_new_tfidf)

for doc, category in zip(doc_new, predict):
    print '%s -> %s' % (doc, twenty_train.target_names[category])


print '\nNMF'
from sklearn.decomposition import TruncatedSVD, NMF

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic #%d:" % topic_idx
        print ", ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])

n_topics = 10
n_top_words = 10
n_features = 1000

tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=n_features, decode_error='ignore',stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(twenty_train.data)

nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)

print '\nTruncatedSVD'
svd = TruncatedSVD(n_components=n_topics, random_state=1, n_iter=10)
svd.fit(tfidf)
svd_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(svd, tfidf_feature_names, n_top_words)
