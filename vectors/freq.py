#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vector creators based on term frequencies."""
from __future__ import division, unicode_literals
from math import log
import array
import numpy as np
from scipy import sparse
from va.vectors.api import normalize, VectorizerMixin
# from va.util import logged

# lda = importr("lda")


class CountVectorizer(VectorizerMixin):

    """Uses simple frequency features to create vectors."""

    def __init__(self, feats=None, normalize=True):
        """Init."""
        self.feats = feats if feats is not None else {}
        self.should_normalize = normalize

    def create_vectors(self, verbs):
        """Create vectors with simple frequency."""
        self.logger.info('Creating frequency vectors for %d features with '
                         '%s...', len(self.feats), verbs)

        j_indices = array.array(str('i'))
        indptr = array.array(str('i'))
        indptr.append(0)
        values = array.array(str('i'))

        for verb in verbs:
            verb_ngrams = verb.ngrams()
            for ngram in verb_ngrams:
                try:
                    j_indices.append(self.feats[ngram])
                except KeyError:
                    pass
                else:
                    values.append(verb_ngrams[ngram])
            indptr.append(len(j_indices))

        j_indices = np.frombuffer(j_indices, dtype=np.intc)
        indptr = np.frombuffer(indptr, dtype=np.intc)
        values = np.frombuffer(values, dtype=np.intc)

        dtm = sparse.csr_matrix((values, j_indices, indptr),
                                shape=(len(indptr) - 1, len(self.feats)))

        dtm.sum_duplicates()

        if self.should_normalize:
            normalize(dtm)

        return dtm


class IcfVectorizer(VectorizerMixin):

    """Uses tf-icf features to create vectors."""

    def __init__(self, feats=None, normalize=True):
        """Init."""
        self.feats = feats if feats is not None else {}
        self.should_normalize = normalize

    def create_vectors(self, verbs):
        """Create tf-icf vector for each verb."""
        self.logger.info('Creating icf vectors for %d features with %s..',
                         len(self.feats), verbs)

        j_indices = array.array(str('i'))
        indptr = array.array(str('i'))
        indptr.append(0)
        values = array.array(str('f'))
        C = len(verbs.codes())
        cat_freq = verbs.cat_freq()

        for verb in verbs:
            verb_ngrams = verb.ngrams()
            for ngram in verb_ngrams:
                try:
                    j_indices.append(self.feats[ngram])
                except KeyError:
                    pass
                else:
                    tf = verb_ngrams[ngram]
                    cf = cat_freq[ngram]
                    icf = log(C/cf, 10)
                    tf_icf = tf * icf
                    values.append(tf_icf)

            indptr.append(len(j_indices))

        j_indices = np.frombuffer(j_indices, dtype=np.intc)
        indptr = np.frombuffer(indptr, dtype=np.intc)
        values = np.frombuffer(values, dtpe=np.float32)

        dtm = sparse.csr_matrix((values, j_indices, indptr),
                                shape=(len(indptr) - 1, len(self.feats)))

        dtm.sum_duplicates()

        if self.should_normalize:
            normalize(dtm)

        return dtm


class IdfVectorizer(VectorizerMixin):

    """Uses tf-idf features to create vectors."""

    def __init__(self, feats=None, normalize=True):
        """Init."""
        self.feats = feats if feats is not None else {}
        self.should_normalize = normalize

    def create_vectors(self, verbs):
        """Create tf-idf vector for each verb."""
        self.logger.info('Creating idf vectors for %d features with %s..',
                         len(self.feats), verbs)

        D = len(verbs)
        doc_freq = verbs.doc_freq()

        j_indices = array.array(str('i'))
        indptr = array.array(str('i'))
        indptr.append(0)
        values = array.array(str('f'))

        for verb in verbs:
            verb_ngrams = verb.ngrams()
            for ngram in verb_ngrams:
                try:
                    j_indices.append(self.feats[ngram])
                except KeyError:
                    pass
                else:
                    tf = verb_ngrams[ngram]
                    df = doc_freq[ngram]
                    idf = log(D/df, 10)
                    tf_idf = tf * idf
                    values.append(tf_idf)

            indptr.append(len(j_indices))

        j_indices = np.frombuffer(j_indices, dtype=np.intc)
        indptr = np.frombuffer(indptr, dtype=np.intc)
        values = np.frombuffer(values, dtype=np.float32)

        dtm = sparse.csr_matrix((values, j_indices, indptr),
                                shape=(len(indptr) - 1, len(self.feats)))

        dtm.sum_duplicates()

        if self.should_normalize:
            normalize(dtm)

        return dtm


# class WNTMVectorCreator(VectorCreator):

#     def create_vectors(self, verbs, feats, min_freq=1, iterations=2000,
#                         alpha=0.5, beta=0.01, **kwds):
#         self.logger.info('Creating wntm vectors for %d features with %s..',
#                             len(feats), verbs)

#         train_verbs, words = self._prep_train_verbs(verbs, min_freq)
#         i_words = {word: i for i, word in enumerate(words, 1)}

#         docs = self._docs_from_word_list(i_words, train_verbs)
#         vocab = ro.StrVector(words)

#         lda_results = lda.lda_collapsed_gibbs_sampler(docs, len(feats), vocab,
#                                                     iterations, alpha, beta)
#         for word in sorted(i_words, key=i_words.get):
#             print i_words[word], word
# print lda.top_topic_words(lda_results.rx2('topics'), 5, by_score='TRUE')

#         doc_sums = lda_results.rx2('document_sums')
#         topic_totals = self

#         vectors = defaultdict(lambda: defaultdict(int))

#         for verb in verbs:
#             verb_tokens = verb.tokens
#             for topic in feats:
#                 for token in verb_tokens:
#                     topic_prob = (doc_sums.rx(topic, i_words[token])[0] /
#                                         topic_totals[token])
#                     word_prob = verb_tokens.count(token) / len(verb_tokens)
#                     vectors[topic][verb] += topic_prob * word_prob

#         return vectors

#     @staticmethod
#     def _prep_train_verbs(verbs, min_freq):
#         train_verbs = Verbatims([verb for verb in verbs
#               if verb.cat =='train'])
#         unigrams = [unigram for unigram, freq in
#                         pool = takewhile(lambda x: x[1] >= min_freq,
#           feats.most_commmon())
#         train_verbs.ngrams(1).most_common() if freq >= min_freq]
#         unigrams = unigrams[5:]
#         for verb in verbs:
#             verb.tokens = [token for token in verb.tokens
#                             if Ngram([token]) in unigrams]

#         words = [unicode(unigram) for unigram in unigrams]

#         return train_verbs, words


#     @staticmethod
#     def _docs_from_word_list(i_words, train_verbs):
#         docs = ro.r['list']()
#         for word, i in i_words.iteritems():
#             word_matrix = {}
#             for list_word, freq in train_verbs.word_net[word].iteritems():
#                 word_vector = ro.IntVector([i_words[list_word]-1, freq])
#                 word_matrix[to_bytes(list_word)] = word_vector

#             if not word_matrix:
#                 word_matrix[word] = ro.IntVector([i_words[word]-1, 1])
#             docs.rx2[i] = ro.r['as.matrix'](ro.DataFrame(word_matrix))

#         return docs

#     @staticmethod
#     def _get_topic_totals(doc_sums, i_words):
#         topic_totals = {}
#         for word in i_words:
#             word_vector = doc_sums.rx(True, i_words[word])
#             topic_totals[word] = sum(word_vector)

#         return topic_totals
