#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Methods to transform verb tokens."""

from __future__ import unicode_literals

import re
import logging
from nltk.corpus import stopwords
from va.preprocess.stemming import PorterStemmer

logger = logging.getLogger(__name__)


def remove_stopwords(verbs, stoplist=stopwords.words('english')):
    """Remove words in stoplist from verbs' tokens.

    NOTE: must be done before stemming as stemming is not reverseable.
    """
    logger.info('Removing stoplisted tokens from %s...', verbs)

    if stoplist is None:
        return verbs

    for verb in verbs:
        verb.text = ' '.join([token for token in verb.tokens()
                              if token.lower() not in stoplist])

    return verbs


def remove_punc_words(verbs):
    """Remove all non alphanumeric tokensi from verbs."""
    logger.info('Removing all non alphanumeric tokens')

    for verb in verbs:
        verb.text = ' '.join([token for token in verb.tokens()
                              if not re.match(r'^\W+$', token)])
    return verbs


def stem(verbs):
    """Use Porter stemmer to stem verbs' tokens.

    NOTE: Not a reversable operation. Make sure to filter tokens before.
    """
    logger.info('Stemming verbs for %s...', verbs)

    stemmer = PorterStemmer()
    for verb in verbs:
        stemmed_tokens = []
        for token in verb.tokens():
            stemmed_tokens.append(stemmer.stem(token.lower(), 0, len(token)-1))
        verb.text = ' '.join(stemmed_tokens)

    return verbs

# def write_verbs(self, verbs, out_dir=None):
#     out_dir = out_dir or os.getcwd()
#     train_filename = os.path.join(out_dir, 'train.csv')
#     test_filename = os.path.join(out_dir, 'test.csv')
#     fieldnames = ['SQL_ID', 'CODE', 'VERB_TEXT']

#     train_verbs = Verbatims([verb for verb in verbs if verb.cat == 'train'])
#     test_verbs = Verbatims([verb for verb in verbs if verb.cat == 'test'])

#     self.logger.info('Writing preprocessed train verbs '
#                      'for %s to train file "%s"...',
#                      train_verbs, train_filename)

#     with open(train_filename, 'w') as train_file:
#         train_csv = csv.DictWriter(train_file, fieldnames,
#                                    quoting=csv.QUOTE_ALL)
#         train_csv.writeheader()
#         for verb in sorted(train_verbs):
#             train_csv.writerow({'SQL_ID': verb.sql_id,
#                                 'CODE': verb.code,
#                                 'VERB_TEXT': to_bytes(verb.text)})

#     self.logger.info('Writing preprocessed test verbs '
#                      'for %s to test file "%s"...',
#                      test_verbs, test_filename)

#     with open(test_filename, 'w') as test_file:
#         test_csv = csv.DictWriter(test_file, fieldnames,
#                                   quoting=csv.QUOTE_ALL)
#         test_csv.writeheader()
#         for verb in sorted(test_verbs):
#             test_csv.writerow({'SQL_ID': verb.sql_id,
#                                'CODE': verb.code,
#                                'VERB_TEXT': to_bytes(verb.text)})
