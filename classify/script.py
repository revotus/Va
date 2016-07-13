#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""Script to classify."""

from __future__ import unicode_literals
from sys import argv
import os
# from collections import defaultdict
from kitchen.text.converters import to_bytes
from va.containers import Verbatims
from va.features.freq import IdfFeatureExtractor, CountFeatureExtractor
from va.vectors.freq import IdfVectorizer, CountVectorizer
from va.classify.svm import SVMClassifier
from va.stop_words import ENGLISH_STOP_WORDS


def classify(verbs, starting_label=0):
    """Main script for classifying verbs."""
    # See va.containers for other Verbatims params
    stop_words = set(ENGLISH_STOP_WORDS)
    # Add other stop words here
    added_to_stop_words = ['stryker', 'good', 'need']
    stop_words.update(added_to_stop_words)
    verbs.stop_words = stop_words
    # verbs.stemmed = False

    verbs.preprocess()

    train_verbs = verbs[:int(round(len(verbs) * 0.75))]
    test_verbs = verbs[len(train_verbs)]

    # See va.features.freq for details
    feat_extractor = IdfFeatureExtractor(threshold=300)
    feats = feat_extractor.get_feats(train_verbs)

    # See va.vectors.freq for details
    vectorizer = IdfVectorizer(feats)
    dtm = vectorizer.create_vectors(train_verbs)
    # feats_list = vectorizer.get_feature_names()

    # See va.classify.svm and sci-learn kit python package for details
    svm = SVMClassifier()
    codes = [verb.code for verb in train_verbs]
    svm.train(dtm, codes)
    predictions = svm.classify(test_verbs)

    return predictions

    # Current method for output
    # delete previous contents if starting_label is default
    # if starting_label == 0:
    #     folder = 'out'
    #     for file_ in os.listdir(folder):
    #         path = os.path.join(folder, file_)
    #         if os.path.isfile(path):
    #             os.unlink(path)

    # all_outfile = open('out/all.txt', 'w+')
    # all_outfile.write('\nTotal Verbatims: %d\n\n' %
    #                   sum([len(vs) for vs in kmeans.clustering_.values()]))
    # for cluster_label in kmeans.clustering_:
    #     print_label = cluster_label + starting_label
    #     single_outfile = open('out/cluster%s.txt' % print_label, 'w+')

    #     all_outfile.write('Cluster %s:\n' % print_label)
    #     single_outfile.write('Cluster %s:\n' % print_label)

    #     for feat in kmeans.top_n_feats(cluster_label, feats_list, n=5):
    #         all_outfile.write('%s\n' % feat)
    #         single_outfile.write('%s\n' % feat)

    #     all_outfile.write('\nVerbatims: %d\n\n' %
    #                       len(kmeans.clustering_[cluster_label]))
    #     single_outfile.write('\nVerbatims: %d\n\n' %
    #                          len(kmeans.clustering_[cluster_label]))

    #     for verb in kmeans.clustering_[cluster_label]:
    #         try:
    #             all_outfile.write('%s: %s\n' % (verb,
    #                                             to_bytes(verb.orig_text,
    #                                                      errors='ignore')))
    #             single_outfile.write('%s: %s\n' % (verb,
    #                                                to_bytes(verb.orig_text,
    #                                                         errors='ignore')))
    #         except UnicodeDecodeError:
    #             pass
    #     all_outfile.write('\n\n\n')
    #     single_outfile.write('\n\n\n')

    # return kmeans.clustering_


if __name__ == '__main__':
    # See va.containers for other Verbatims params
    verbs = Verbatims.from_files([argv[1]])

    classify(verbs)
