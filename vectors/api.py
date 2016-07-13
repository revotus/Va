#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Api for vector creators."""
from __future__ import division, unicode_literals
# import os
# import csv
# from kitchen.text.converters import to_bytes
from math import sqrt
from operator import itemgetter
# from va.containers import Ngrams
from va.util import logged


@logged
class VectorizerMixin(object):

    """Create a vector for each verb given a set of features."""

    def create_vectors(self, verbs):
        """Create vector for each verb given a dict of features."""
        raise NotImplementedError()

    def get_feature_names(self):
        """Array mapping from feature integer indices to feature name."""
        return [t for t, _ in sorted(self.feats.iteritems(), key=itemgetter(1))]


def normalize(matrix):
    """Normalize sparse matrix."""
    n_samples = matrix.shape[0]
    for i in xrange(n_samples):
        sum_ = 0.0

        for j in xrange(matrix.indptr[i], matrix.indptr[i + 1]):
            sum_ += (matrix.data[j] * matrix.data[j])

        if sum_ == 0.0:
            # do not normalize empty rows (can happen if CSR is not pruned
            # correctly)
            continue

        sum_ = sqrt(sum_)

        for j in xrange(matrix.indptr[i], matrix.indptr[i + 1]):
            matrix.data[j] /= sum_

    # def modify_vectors(self, verbs, vectors, feats):
    #     """Modify vectors."""
    #     feats_to_add = [feat for feat in feats if feat not in vectors]
    #     feats_to_del = [feat for feat in vectors if feat not in feats]

    #     self.logger.info('Modifying vectors by adding %d new features and '
    #                      'removing %d old features for a total of %d '
    #                      'features...',
    #                      len(feats_to_add), len(feats_to_del),
    #                      len(vectors) +
    #                      len(feats_to_add) - len(feats_to_del))

    #     if feats_to_add:
    #         add_vectors = self.create_vectors(verbs)

    #     for add_feat in feats_to_add:
    #         vectors[add_feat] = add_vectors[add_feat]
    #     for del_feat in feats_to_del:
    #         del vectors[del_feat]

    #     return vectors

    # def write_vectors(self, verbs, vectors, feats, out_dir=None):
    #     """Write vector for each verb to external file."""
    #     out_dir = out_dir or os.getcwd()
    #     vect_filename = os.path.join(out_dir, 'vectors.csv')
    #     self.logger.info('Writing %d feature vectors to "%s"...',
    #                      len(vectors), vect_filename)

    #     feat_list = [feat for feat, _ in feats.most_common()]
    #     fieldnames = ['SQL_ID', 'CODE'] + [to_bytes(feat) for
    #                                        feat in feat_list]
    #     with open(vect_filename, 'w') as vect_file:
    #         vect_csv = csv.DictWriter(vect_file, fieldnames)
    #         vect_csv.writeheader()
    #         for verb in verbs:
    #             row = {'SQL_ID': verb.sql_id,
    #                    'CODE': verb.code}
    #             row.update(dict([(to_bytes(feat),
    #                               '{:.2f}'.format(vectors[feat][verb]))
    #                              for feat in feat_list]))
    #             vect_csv.writerow(row)

    # @classmethod
    # def feats_from_file(cls, feat_filename):
    #     """Read features from external files."""
    #     cls.logger.info('Getting features from "%s"...', feat_filename)

    #     feats = Ngrams()
    #     with open(feat_filename) as feat_file:
    #         feat_csv = csv.DictReader(feat_file)
    #         for row in feat_csv:
    #             feat = row['FEATURE']
    #             freq = row['FREQUENCY']
    #             feats[feat] = freq

    #     cls.logger.debug('Found %d features in "%s"', len(feats), feat_filename)
    #     return feats
