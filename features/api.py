#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Api for class that extracts features from a Verbatims instance."""
from __future__ import unicode_literals
import os
import csv
from itertools import takewhile
from kitchen.text.converters import to_bytes
from va.containers import Ngrams
from va.util import logged


@logged
class FeatureExtractor(object):

    """Class that takes verbs and extracts feats from them."""

    def __init__(self, threshold=0, filter_type='size'):
        """Do not overide default params for no feat filtering."""
        self.threshold = threshold
        self.filter_type = filter_type

    def get_feats(self, verbs):
        """Convenience call to extract features and then rank and select."""
        feats = self.extract_feats(verbs)
        filtered_feats = self.filter_feats(verbs, feats)

        return filtered_feats

    def extract_feats(self, verbs):
        """Extract features from verbs."""
        raise NotImplementedError()

    def score_feats(self, verbs, feats):
        """Rank features according to some criteria."""
        raise NotImplementedError()

    def filter_feats(self, verbs, feats):
        """Keep only the feats w/ score above a certain threshold."""
        self.logger.info('Filtering %d features...', len(feats))

        if self.threshold == 0:
            return feats

        scored_feats = self.score_feats(verbs, feats)
        if self.filter_type == 'size':
            pool = _pool_by_size(scored_feats, self.threshold)
        if self.filter_type == 'score':
            pool = _pool_by_score(scored_feats, self.threshold)

        filtered_feats = {feat: ind for ind, (feat, val) in enumerate(pool)}

        self.logger.debug('%d features left after filtering',
                          len(filtered_feats))

        return filtered_feats

    def write_feats(self, feats, out_dir=None):
        """Write features and their corresponding scores to output file."""
        out_dir = out_dir or os.getcwd()
        feat_filename = os.path.join(out_dir, 'feats.csv')
        self.logger.info('Writing %d features to "%s"...',
                         len(feats), feat_filename)

        with open(feat_filename, 'w') as feat_file:
            fieldnames = ['FEATURE', 'FREQUENCY']
            feat_csv = csv.DictWriter(feat_file, fieldnames)
            feat_csv.writeheader()
            for feat, freq in feats.most_common():
                feat_csv.writerow({'FEATURE': to_bytes(feat),
                                   'FREQUENCY': freq})


def _pool_by_size(feats, size):
    """Keep only the top scored feats s.t. len(top_feats) > size."""
    if size > len(feats):
        size = len(feats)
    pool = feats.most_common(size)

    return pool


def _pool_by_score(feats, score):
    """Keep only ranked feats with score above threshold score."""
    pool = takewhile(lambda x: x[1] >= score, feats.most_commmon())

    return pool


@logged
class BOWFeatureExtractor(FeatureExtractor):

    """Feature extractor that treats doc terms as standalone feats."""

    def __init__(self, ngram_range=(1, 1), threshold=300, filter_type='size'):
        """init."""
        super(BOWFeatureExtractor, self).__init__(threshold, filter_type)
        self.ngram_range = ngram_range

    def get_feats(self, verbs, ngram_range=(1, 1)):
        """Get feats and filter."""
        feats = self.extract_feats(verbs)
        filtered_feats = self.filter_feats(verbs, feats)
        feats = Ngrams(filtered_feats)

        return feats

    def extract_feats(self, verbs):
        """Extract feature set from Verbatims instance.

        Ngrams are used instead of tokens for the ability to choose a larger
        feature set if desired. Set ngram_size=1 to get only single tokens.
        """
        self.logger.info('Extracting %d-gram features from %s...',
                         self.ngram_range, verbs)
        feats = verbs.vocab(self.ngram_range)

        self.logger.debug('Extracted %d features from %s', len(feats), verbs)
        return feats
