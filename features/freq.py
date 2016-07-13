#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feature Extractors that use Bag-of-Words approach."""
from __future__ import division, unicode_literals
from math import log
from va.containers import Ngrams, NgramCounter
from va.features.api import BOWFeatureExtractor
from va.util import logged


@logged
class CountFeatureExtractor(BOWFeatureExtractor):

    """Extract and score simple count feats."""

    def score_feats(self, verbs, feats):
        """Return ngrams because ngrams are mapped to counts."""
        ngrams = verbs.ngrams(self.ngram_range)
        scored_feats = NgramCounter({feat: ngrams[feat] for feat in feats})
        # scored_feats = Ngrams({ngram: count for ngram, count
        #                        in verbs.ngrams(self.ngram_range).iteritems()
        #                        if ngram in feats})

        return scored_feats


@logged
class IcfFeatureExtractor(BOWFeatureExtractor):

    """Extract and score tf*icf features."""

    def score_feats(self, verbs, feats):
        """Score features according to tf-icf value.

        Tf = total frequency of term in document collection
        Icf = inverse category frequency - The total number of categories in
            the document collection divided by the number of categories
            in which the term occurs, log-scaled.
        """
        self.logger.info('scoreing %d features with icf values...', len(feats))

        cat_freq = verbs.cat_freq()
        C = len(verbs.codes())
        scored_feats = Ngrams()

        for feat in feats:
            tf = feats[feat]
            cf = cat_freq[feat]
            icf = log(C/cf, 10)
            tf_icf = tf * icf
            scored_feats[feat] = tf_icf

        self.logger.debug('scored %d features', len(scored_feats))
        return scored_feats


@logged
class IdfFeatureExtractor(BOWFeatureExtractor):

    """Extract and score tf*idf features."""

    def score_feats(self, verbs, feats):
        """score features according to tf-idf value.

        Tf - total frequency of term in document collection
        Idf - inverse document frequency - total number of documents divided
            by the number of documents in which the term occurs, log-scaled.
        """
        self.logger.info('scoreing %d features with idf values...', len(feats))

        doc_freq = verbs.doc_freq()
        D = len(verbs)
        ngrams = verbs.ngrams(self.ngram_range)
        feat_counts = Ngrams({feat: ngrams[feat] for feat in feats})
        scored_feats = NgramCounter()

        for feat in feats:
            tf = feat_counts[feat]
            df = doc_freq[feat]
            idf = log(D/df, 10)
            tf_idf = tf * idf
            scored_feats[feat] = tf_idf

        self.logger.debug('scored %d features', len(scored_feats))
        return scored_feats


@logged
class MIFeatureExtractor(BOWFeatureExtractor):

    """Extract and score mutual information features."""

    def score_feats(self, verbs, feats):
        """score features accoridng to mutual information criteria."""
        # TODO: finish desciption of Mutual information criteria
        feats = feats if feats is not None else verbs.ngrams(self.ngram_range)
        self.logger.info('scoreing %d features with MI values...', len(feats))

        code_ngrams = verbs.code_ngrams()
        codes = verbs.codes()
        N = len(verbs)
        scored_feats = Ngrams()

        for feat in feats:
            MI = 0.0
            feat_freq = feats[feat]
            for code in code_ngrams:
                cooccur = code_ngrams[code][feat]
                code_freq = codes[code]
                try:
                    I = log(cooccur * N / (feat_freq * code_freq), 2)
                    MI += code_freq/N * I
                except ValueError:
                    pass
            scored_feats[feat] = MI

        self.logger.debug('scored %d features', len(scored_feats))
        return scored_feats


@logged
class ChiFeatureExtractor(BOWFeatureExtractor):

    """Extract and score chi squared features."""

    def score_feats(self, verbs, feats):
        """score features with chi squared value."""
        # TODO: finish chi squared description
        feats = feats if feats is not None else verbs.ngrams(self.ngram_range)
        self.logger.info('scoreing %d features with Chi2 values...', len(feats))

        code_ngrams = verbs.code_ngrams()
        codes = verbs.codes()
        N = len(verbs)
        scored_feats = Ngrams()

        for feat in feats:
            chi = 0.0
            for code in code_ngrams:
                A = code_ngrams[code][feat]
                B = feats[feat] - A
                C = codes[code] - A
                D = N - A - B - C
                x = N * (A*D-C*B)**2 / ((A+C)*(B+D)*(A+B)*(C+D))
                chi += C/N * x

            scored_feats[feat] = chi

        self.logger.debug('scored %d features', len(scored_feats))
        return scored_feats


@logged
class IGFeatureExtractor(BOWFeatureExtractor):

    """Extract and score information gain features."""

    def score_feats(self, verbs, feats):
        """score features according to information gain."""
        # TODO: Finish ig description.
        feats = feats if feats is not None else verbs.ngrams(self.ngram_range)
        self.logger.info('scoreing %d features with IG values...', len(feats))

        codes = verbs.codes()
        code_ngrams = verbs.code_ngrams()
        n = sum(feats.values())
        N = len(verbs)
        scored_feats = Ngrams()

        for feat in feats:
            first = 0.0
            second = 0.0
            third = 0.0

            t = feats[feat]
            p_t = t / n
            p_not_t = 1 - p_t

            for code in code_ngrams:
                c_t = code_ngrams[code][feat]
                c_all = sum(code_ngrams[code].values())

                p_c = codes[code]/N
                p_c_t = c_t / t
                p_c_not_t = (c_all - c_t) / (n - t)

                try:
                    first += p_c * log(p_c, 2)
                except ValueError:
                    pass
                try:
                    second += p_c_t * log(p_c_t, 2)
                except ValueError:
                    pass
                try:
                    third += p_c_not_t * log(p_c_not_t, 2)
                except ValueError:
                    pass

            IG = -first + p_t * second + p_not_t * third
            scored_feats[feat] = IG

        self.logger.debug('scored %d features', len(scored_feats))
        return scored_feats
