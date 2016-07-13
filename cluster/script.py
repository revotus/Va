#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""Script to cluster."""

from __future__ import unicode_literals, division
from sys import argv
# from math import exp
import csv
import re
from collections import defaultdict
from kitchen.text.converters import to_bytes
from va.containers import Verbatims
from va.features.freq import IdfFeatureExtractor, CountFeatureExtractor
from va.vectors.freq import IdfVectorizer, CountVectorizer
from va.cluster.kmeans import KMeansClusterer
from va.stop_words import ENGLISH_STOP_WORDS


def cluster(verbs, n_clusters=10, starting_label=0):
    """Main script for clustering verbs."""
    # See va.containers for other Verbatims params
    stop_words = set(ENGLISH_STOP_WORDS)
    # Add other stop words here
    added_to_stop_words = ['stryker', 'good', 'need', 'different', 'difference',
                           'best', "'s", 'day', 'like', "n't", 'high', 'value',
                           'everybody', 'market', 'member', 'matter',
                           'required', 'requires', 'require', 'wonderful',
                           'accomplish', 'group', 'true', 'paying',
                           'provides', 'provide', 'provided', 'providing',
                           'better', 'basis', 'office', 'incredicle', 'basis',
                           'outside', 'long', 'neccessary', 'especially',
                           'grateful', 'love', 'valued', 'feel', 'great',
                           'ability', 'way', 'highest', 'does', 'values',
                           'ready', 'think', 'abilities', 'meet', 'look',
                           'looking', 'looks', 'want', 'wants', 'company', 'lot'
                           'lots', 'companies', "company's", 'companys', "'re",
                           "'ve", 'make', "'m", 'appreciate', 'thing', 'things',
                           'amazing', 'making', 'makes', 'know', 'knows',
                           'knowing', 'just', 'time', 'work', 'job', 'working',
                           'really', 'constantly', 'come', 'employee',
                           'employees', 'say', 'improve', 'increase',
                           'change', 'build', 'develop', 'program', 'need',
                           'level', 'workers', 'possible', 'stop', 'continue',
                           'have', 'having', 'issue', 'issues', 'focus', 'year',
                           'new', 'use', 'week']
    stop_words.update(added_to_stop_words)
    verbs.stop_words = stop_words
    # verbs.stop_words = None
    verbs.stemmed = False

    verbs.preprocess()

    # See va.features.freq for details
    feat_extractor = IdfFeatureExtractor(threshold=1000)
    feats = feat_extractor.get_feats(verbs)

    # See va.vectors.freq for details
    vectorizer = IdfVectorizer(feats)
    dtm = vectorizer.create_vectors(verbs)
    feats_list = vectorizer.get_feature_names()

    # See va.cluster.kmeans and sci-learn kit python package for details
    kmeans = KMeansClusterer(n_clusters=n_clusters)
    kmeans.cluster_vectorspace(dtm, verbs)
    distances = kmeans.distances_from_centroid(dtm)

    for i, label in enumerate(kmeans.model.labels_):
        verbs[i].code = label + starting_label
        verbs[i].distance = distances[i]

    top_n_feats = kmeans.top_n_feats(feats_list, 5, starting_label)

    return top_n_feats


def recode(verbs, old_code, new_code):
    """Change code."""
    tmp = [v for v in verbs if v.code == old_code]
    for v in tmp:
        v.code = new_code


def make_other(verbs, fts, cluster_ind):
    """Set code for cluster to other code: 99."""
    clustr = [v for v in verbs if v.code in cluster_ind]
    for v in clustr:
        v.code = -1
    fts[-1] = ['other']


def sub_cluster(verbs, fts, sub_inds=None, reg=None, org=False,
                start=None, n_clusters=1):
    """Combine all verbs with label in sub_inds and recluster."""
    sub_vs = []
    if sub_inds:
        if sub_inds == 'all':
            sub_vs.extend([v for v in verbs])
        else:
            sub_vs.extend([v for v in verbs if v.code in sub_inds])
    if reg:
        if org:
            sub_vs = [v for v in sub_vs if re.search(reg, v.orig_text)]
        else:
            sub_vs = [v for v in sub_vs if re.search(reg, v.text, re.I)]

    for v in sub_vs:
        v.text = v.orig_text
    sub_vs = Verbatims(sub_vs)
    if not start:
        start = max([v.code for v in verbs]) + 1
    sub_fts = cluster(sub_vs, n_clusters, start)
    fts.update(sub_fts)
    return sub_fts


def load_feats(filename):
    """Load feats dictionary from csv file."""
    with open(filename, 'r') as csvfile:
        featsin = csv.DictReader(csvfile)
        feats = {int(row['Label']): row['Feats'].split('-') for row in featsin}
    return feats


def write_clusters(verbs, top_feats):
    """writeclustering from kmeans."""
    with open('out/verbs.csv', 'w+') as verbfile, \
         open('out/feats.csv', 'w+') as featfile:
        verbout = csv.DictWriter(verbfile, ['Id', 'Code', 'Distance',
                                            'Orig_Text', 'Top_N_Feats',
                                            'Cluster_Size', 'Verb_ID'])
        featout = csv.DictWriter(featfile, ['Label', 'Feats', 'Size'])
        verbout.writeheader()
        featout.writeheader()

        clustering = defaultdict(list)
        for verb in verbs:
            clustering[verb.code].append(verb)

        for label in sorted(clustering):
            try:
                top_feats_com = '-'.join([str(ft) for ft in top_feats[label]])
            except UnicodeEncodeError:
                pass
            cluster_size = len(clustering[label])

            featout.writerow({'Label': label, 'Feats': top_feats_com,
                              'Size': cluster_size})

            for i, verb in enumerate(clustering[label]):
                try:
                    verbout.writerow({'Id': i, 'Code': verb.code, 'Distance':
                                      verb.distance, 'Orig_Text':
                                      to_bytes(verb.orig_text, errors='ignore'),
                                      'Top_N_Feats': top_feats_com,
                                      'Cluster_Size': cluster_size, 'Verb_ID':
                                      verb.sql_id})
                except UnicodeDecodeError:
                    pass


if __name__ == '__main__':
    # See va.containers for other Verbatims params
    arg_vs = Verbatims.from_files([argv[1]])

    cluster(arg_vs)
