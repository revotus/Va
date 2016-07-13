#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Containers for verbatims."""

from __future__ import unicode_literals

# import codecs
import csv
import re
from collections import Counter, defaultdict, MutableSequence, MutableMapping
from abc import ABCMeta
from functools import total_ordering
# from copy import deepcopy
from nltk import word_tokenize, sent_tokenize
from kitchen.text.converters import to_unicode, to_bytes
from va.preprocess import tokentools, trimming
from va.stop_words import ENGLISH_STOP_WORDS
from va.util import (AutoLabeler, logged, NulObsDesc, TypNulObsDesc,
                     TypSubjDesc, TypSubjNulObsDesc)


class Ngram(tuple):

    """To represent Ngram tuples better."""

    def __new__(cls, iterable):
        """Create new and convert to unicode.

        If string is passed, create tuple with one string entry rather than
        iterate through string like tuple.__new__.
        """
        if isinstance(iterable, str):
            tokens = [unicode(iterable)]
        else:
            tokens = [unicode(item) for item in iterable]
        return super(Ngram, cls).__new__(cls, tokens)

    def __repr__(self):
        """Represent with name 'Ngram'."""
        return '{}{!r}'.format(self.__class__.__name__, tuple(self))

    def __str__(self):
        """Join tokens with underscore."""
        return '_'.join(token for token in self)


class AutoLabeledABC(AutoLabeler, ABCMeta):

    """Combines the two metaclasses so that others can inherit from it."""

    pass


class Ngrams(MutableMapping):

    """Mapping for Ngram objects."""

    __metaclass__ = AutoLabeledABC
    _ngram_range = TypNulObsDesc(typ=tuple)
    _unigrams = TypNulObsDesc(typ=MutableMapping, subtyp=Ngram)
    _bigrams = TypNulObsDesc(typ=MutableMapping, subtyp=Ngram)
    _trigrams = TypNulObsDesc(typ=MutableMapping, subtyp=Ngram)
    _ngrams = TypSubjDesc(typ=MutableMapping,
                          observers=[_ngram_range, _unigrams, _bigrams,
                                     _trigrams])

    def __init__(self, *args, **kwds):
        """Initalize Ngrams instance from mapping of Ngram objects.

        Initializes with same signature as dict. All keys are changed to
        Ngram objects on initalization.
        """
        ngrams = dict(*args, **kwds)
        self._ngrams = {Ngram(key): val for key, val in ngrams.iteritems()}
        self._ngram_range = None
        self._unigrams = None
        self._bigrams = None
        self._trigrams = None

    def ngrams(self):
        """Return all ngrams of length <= 'n'.

        Preferred usage is to call object directly.
        Included for so that _ngrams can be accessed publicly in Ngrams's
        update method without infinite recursion.
        """
        return self._ngrams

    def ngram_range(self):
        """Return ngram length range."""
        if self._ngram_range is None:
            try:
                min_n = min([len(ngram) for ngram in self._ngrams])
                max_n = max([len(ngram) for ngram in self._ngrams])
                self._ngram_range = (min_n, max_n)
            except ValueError:
                pass

        return self._ngram_range

    def unigrams(self):
        """Return all unigrams in mapping."""
        if self._unigrams is None:
            self._unigrams = self.__class__({ngram: val for ngram, val
                                             in self._ngrams.iteritems()
                                             if len(ngram) == 1})
        return self._unigrams

    def bigrams(self):
        """Return all bigrams in mapping."""
        if self._bigrams is None:
            self._bigrams = self.__class__({ngram: val for ngram, val
                                            in self._ngrams.iteritems()
                                            if len(ngram) == 2})
        return self._bigrams

    def trigrams(self):
        """Return all trigrams in mapping."""
        if self._trigrams is None:
            self._trigrams = self.__class__({ngram: val for ngram, val
                                             in self._ngrams.iteritems()
                                             if len(ngram) == 3})
        return self._trigrams

    def __repr__(self):
        """Represent with class name and convert ngrams to bytes."""
        try:
            return '{}({!r})'.format(self.__class__.__name__,
                                     {to_bytes(ngram): val for ngram, val
                                      in self._ngrams.iteritems()})
        except AttributeError:
            return'{}({})'.format(self.__class__.__name__, {})

    # def __str__(self):
    #     return '{}({})'.format(self.__class__.__name__, self.ngrams)

    # def __call__(self, n=3):
    #     """Return all ngrams of len <= 'n' when instance is called."""
    #     ngrams = self.__class__()
    #     ngrams.update(self.unigrams())
    #     if n > 1:
    #         ngrams.update(self.bigrams())
    #         if n > 2:
    #             ngrams.update(self.trigrams())
    #     return ngrams

    def __call__(self, ngram_range=(1, 1)):
        """Return all ngrams w/ min_n <= len <= max_n when called."""
        min_n, max_n = ngram_range
        ngrams = self.__class__({ngram: val for ngram, val in
                                 self._ngrams.iteritems()
                                 if min_n <= len(ngram) <= max_n})
        return ngrams

    def __len__(self):
        """Return number of Ngram keys in mapping."""
        return len(self._ngrams) if self._ngrams is not None else 0

    def __getitem__(self, key):
        """Return value of key after converting to Ngram."""
        return self._ngrams[Ngram(key)]

    def __setitem__(self, key, val):
        """Convert key to Ngram and set to val."""
        self._ngrams[Ngram(key)] = val

    def __delitem__(self, key):
        """Convert key to Ngram and delete from mapping."""
        del self._ngrams[Ngram(key)]

    def __iter__(self):
        """Iterate over keys in mapping."""
        for key in self._ngrams:
            yield key


class NgramDefault(Ngrams):

    """Same as Ngrams, but with a default like defaultdict."""

    def __init__(self, default_factory=None, *args, **kwds):
        """Initilialize with same signature as defaultdict."""
        super(NgramDefault, self).__init__(*args, **kwds)
        self._ngrams = defaultdict(default_factory, self._ngrams)


class NgramCounter(Ngrams, Counter):

    """Ngrams mapping with added Counter methods."""

    def __init__(self, iterable=None, **kwds):
        """Initalize like Counter then call Ngrams init then back to Counter."""
        counts = Counter(iterable, **kwds)
        super(NgramCounter, self).__init__(counts, **kwds)
        self._ngrams = Counter(self._ngrams)

    def update(self, other):
        """Update mapping with other.

        Was going to use 'self._ngrams.update(other)'
        or Counter.update(self, other) only, but
        if Counter self._ngrams is empty, Counter's update method calls
        dict.update(self, other) and that does not seem to work with
        Ngrams and NgramCounter. Not sure why.
        """
        if isinstance(other, Ngrams):
            self._ngrams.update(other.ngrams())
        else:
            self._ngrams.update(other)


@logged
class Document(object):

    """Container for a text document."""

    __metaclass__ = AutoLabeler
    _ngrams = TypNulObsDesc(typ=NgramCounter)
    _word_net = NulObsDesc()
    _tokens = TypSubjNulObsDesc(typ=list, subtyp=unicode,
                                observers=[_ngrams, _word_net])
    text = TypSubjDesc(typ=unicode, observers=[_tokens])

    def __init__(self, text):
        """Init."""
        self.text = text
        self._tokens = None
        self._ngrams = None
        self._word_net = None

    def ngrams(self, ngram_range=(1, 1)):
        """Return doc's ngrams w min_n <= len < max_n."""
        min_n, max_n = ngram_range
        if self._ngrams is None:
            self._ngrams = self._get_ngrams(min_n, max_n)
        elif self._ngrams != {}:
            old_min_n, old_max_n = self._ngrams.ngram_range()
            if min_n < old_min_n or max_n > old_max_n:
                self._ngrams = self._get_ngrams(min_n, max_n)
        return self._ngrams(ngram_range)

    def tokens(self):
        """Tokenize verb's text using NLTK's tokenize methods."""
        if self._tokens is None:
            self._tokens = []
            for sent in sent_tokenize(self.text):
                # tokens.append(u'<s>')
                self._tokens.extend(word_tokenize(sent))
                # tokens.append(u'</s>')
        return self._tokens

    def _get_ngrams(self, min_n, max_n):
        """Get ngrams: count NgramCounter."""
        tokens = self.tokens()
        n_tokens = len(tokens)
        ngrams = NgramCounter()

        for n in xrange(min_n, min(max_n + 1, n_tokens + 1)):
            for i in xrange(n_tokens - n + 1):
                ngrams[Ngram(tokens[i: i + n])] += 1

        return ngrams

    def word_net(self):
        """Construct word net from verb's tokens."""
        word_net = defaultdict(Counter)
        tokens = self.tokens
        for index in range(len(tokens)):
            token_1 = tokens[index]
            for token_2 in tokens[index+1: index+10]:
                if token_1 != token_2:
                    word_net[token_1][token_2] += 1
                    word_net[token_2][token_1] += 1

        return word_net


@logged
@total_ordering
class Verbatim(Document):

    """Container for a verbatim."""

    def __init__(self, sql_id, text, code=None, cat=None):
        """Initialize with SQL ID, code, and text of verb."""
        super(Verbatim, self).__init__(text)
        self.sql_id = sql_id
        self.code = code
        self.orig_text = self.text
        self.cat = cat
        self.distance = None

    def __repr__(self):
        """Respresent intance with class name and sql_id, code, and text."""
        return '{}(sql id: {!r}, code: {!r}, text: "{!r}")'.format(
            self.__class__.__name__, self.sql_id,
            self.code, to_bytes(self.text))

    def __str__(self):
        """Return sql_id for string representation."""
        return '{}({})'.format(self.__class__.__name__, self.sql_id)

    def __hash__(self):
        """Hash by sql_id."""
        return hash(self.sql_id)

    def __eq__(self, other):
        """Compare equality by sql_id."""
        return self.sql_id == other.sql_id

    def __lt__(self, other):
        """Compare by sql_id."""
        return self.sql_id < other.sql_id

    def __int__(self):
        """Return int version of sql_id."""
        return int(self.sql_id)


class Corpus(MutableSequence):

    """A corpus of documents."""

    __metaclass__ = AutoLabeledABC
    _word_net = NulObsDesc()
    _cat_freq = TypNulObsDesc(typ=Ngrams)
    _doc_freq = TypNulObsDesc(typ=Ngrams)
    _ngram_codes = TypSubjNulObsDesc(typ=NgramDefault, observers=[_cat_freq])
    _code_ngrams = TypNulObsDesc(typ=defaultdict)
    _codes = TypNulObsDesc(typ=Counter)
    _ngrams = TypNulObsDesc(typ=NgramCounter)
    _vocab = TypNulObsDesc(typ=Ngrams)
    _docs = TypSubjDesc(typ=list,
                        observers=[_vocab, _ngrams, _codes, _ngram_codes,
                                   _code_ngrams, _doc_freq, _word_net])

    def __init__(self, docs=None, stop_words=None, stemmed=True, lower=True):
        """Initialize with list of Document instances."""
        self._docs = docs if docs is not None else []
        # self._preprocessed = False
        self.stop_words = stop_words
        self.stemmed = stemmed
        self.lower = lower
        # self._test_verbs = None
        # self._train_verbs = None
        self._vocab = None
        self._ngrams = None
        self._codes = None
        self._ngram_codes = None
        self._code_ngrams = None
        self._doc_freq = None
        self._cat_freq = None
        self._word_net = None

    # def train(self):
    #     """Return Verbatims object with verbs that have cat = 'train'."""
    #     return self.__class__([verb for verb in self._verbs
    #                            if verb.cat == 'train'])

    # def test(self):
    #     """Return Verbatims object with verbs that have cat = 'test'."""
    #     return self.__class__([verb for verb in self._verbs
    #                            if verb.cat == 'test'])

    def vocab(self, ngram_range=(1, 1)):
        """Return mapping of ngrams to indices."""
        min_n, max_n = ngram_range
        if self._vocab is None:
            self._vocab = self._get_vocab(min_n, max_n)
        else:
            old_min_n, old_max_n = self._vocab.ngram_range()
            if min_n != old_min_n or max_n != old_max_n:
                self._vocab = self._get_vocab(min_n, max_n)
        return self._vocab(ngram_range)

    def _get_vocab(self, min_n, max_n):
        """Return mapping of docs' ngrams: index."""
        vocab = set()
        for doc in self:
            for ngram in doc.ngrams((min_n, max_n)):
                vocab.add(ngram)
        vocab = Ngrams({ngram: ind for ind, ngram in enumerate(vocab)})

        return vocab

    def ngrams(self, ngram_range=(1, 1)):
        """Return frequency mapping for all docs' ngram_counts."""
        min_n, max_n = ngram_range
        if self._ngrams is not None:
            old_min_n, old_max_n = self._ngrams.ngram_range()
            if min_n != old_min_n or max_n != old_max_n:
                self._ngrams = self._get_ngrams(min_n, max_n)
        else:
            self._ngrams = self._get_ngrams(min_n, max_n)
        return self._ngrams(ngram_range)

    def _get_ngrams(self, min_n, max_n):
        """Return ngram counter built from docs' ngrams."""
        ngrams = NgramCounter()
        for doc in self:
            ngrams.update(doc.ngrams((min_n, max_n)))

        return ngrams

    def trim_verbs(self, count_type='total', count=0):
        """Trim doc list down to size."""
        self.logger.info('Trimming %s...', self._verbs)

        # TODO: integrate trim_verbs_per
        trim_funcs = {'total': trimming.trim_verbs_tot,
                      'prop': trimming.trim_verbs_prop}
        trim_func = trim_funcs[count_type]

        split_ratio = len(self.train())/len(self)
        train_count = int(round(split_ratio * count))
        test_count = count - train_count

        train_verbs = trim_func(self.train(), train_count)
        test_verbs = trim_func(self.test(), test_count)

        self._docs = list(train_verbs + test_verbs)

    def preprocess(self):
        """Preprocess docs' text."""
        self._docs = tokentools.remove_punc_words(self._docs)
        if self.lower:
            for doc in self._docs:
                doc.text = doc.text.lower()
        stop_list = self.get_stop_list()
        self._docs = tokentools.remove_stopwords(self._docs, stop_list)
        if self.stemmed:
            self._docs = tokentools.stem(self._docs)

    def get_stop_list(self):
        """Retunr stop list."""
        if self.stop_words == "english":
            return ENGLISH_STOP_WORDS
        elif self.stop_words is None:
            return None
        else:               # assume it's a collection
            return frozenset(self.stop_words)

    def codes(self):
        """Return Counter of docs' codes."""
        if self._codes is None:
            self._codes = Counter()
            for doc in self:
                self._codes[doc.code] += 1
        return self._codes

    # TODO: Check to see if n is necessary for following methods
    def code_ngrams(self):
        """Return mapping with code keys and ngram values."""
        if self._code_ngrams is None:
            self._code_ngrams = defaultdict(NgramCounter)
            for doc in self:
                self._code_ngrams[doc.code].update(doc.ngrams())
        return self._code_ngrams

    def ngram_codes(self):
        """Return mapping with ngram keys and code values."""
        if self._ngram_codes is None:
            self._ngram_codes = NgramDefault(Counter)
            for doc in self:
                for ngram in doc.ngrams():
                    self._ngram_codes[ngram][doc.code] += doc.ngrams()[ngram]
        return self._ngram_codes

    def cat_freq(self):
        """Return mapping with number of unique codes for each ngram."""
        if self._cat_freq is None:
            self._cat_freq = Ngrams({ngram: len(codes) for ngram, codes
                                     in self.ngram_codes().iteritems()})
        return self._cat_freq

    def doc_freq(self):
        """Return doc freq for idf computations."""
        if self._doc_freq is None:
            self._doc_freq = NgramCounter()
            for doc in self:
                for ngram in doc.ngrams():
                    self._doc_freq[ngram] += 1
        return self._doc_freq

    def word_net(self):
        """Return word net docs."""
        word_net = defaultdict(Counter)
        for doc in self:
            for token in doc.word_net:
                word_net[token].update(doc.word_net[token])
        return word_net

    def __len__(self):
        """Return number of docs."""
        return len(self._docs)

    def __getitem__(self, val):
        """Return Corpus slice or single Document."""
        if isinstance(val, slice):
            return self.__class__(self._docs[val])
        return self._docs[val]

    def __setitem__(self, index, doc):
        """Set index of Corpus to doc."""
        self._docs[index] = doc

    def __delitem__(self, index):
        """Delete doc with index."""
        del self._docs[index]

    def insert(self, index, doc):
        """Insert doc at index point."""
        return self.__class__(self._docs[:index] + [doc] + self._docs[index:])

    def __add__(self, other):
        """Add two Corpus instances together."""
        return self.__class__(self._docs + other)


@logged
class Verbatims(Corpus):

    """MutableSequence of Verbatim instances."""

    @classmethod
    def from_files(cls, filenames, *args, **kwds):
        """Create Verbatims instance from verbfiles."""
        cls.logger.info('Getting verbatims from file(s):\t%s...', filenames)
        verbs = []
        for filename in filenames:
            with open(filename) as infile:
                incsv = csv.DictReader(infile)

                sql_id_fieldname = None
                code_fieldname = None
                text_fieldname = None
                for fieldname in incsv.fieldnames:
                    if re.search(r'(verb|dc).*id', fieldname, re.I):
                        sql_id_fieldname = fieldname

                    elif re.search(r'\b(code|label)', fieldname, re.I):
                        digit = re.search(r'\d+', fieldname)
                        if not digit or int(digit.group(0)) == 1:
                            code_fieldname = fieldname

                    elif re.search(r'(verb)?.*(text|original)',
                                   fieldname, re.I):
                        text_fieldname = fieldname

                for i, row in enumerate(incsv):
                    verb = Verbatim(
                        sql_id=row.get(sql_id_fieldname, i),
                        code=row.get(code_fieldname, None),
                        text=to_unicode(row[text_fieldname]))
                    verbs.append(verb)
        verbs = cls(verbs, *args, **kwds)

        cls.logger.debug('Retrieved %d verbatims', len(verbs))
        return verbs

    def __repr__(self):
        """Represent with class name and verbs."""
        try:
            return '{}({!r})'.format(self.__class__.__name__, self._verbs)
        except AttributeError:
            return'{}()'.format(self.__class__.__name__)

    def __str__(self):
        """Return string with class name and number of verbs."""
        try:
            return '{}({} verbs)'.format(self.__class__.__name__,
                                         len(self._docs))
        except AttributeError:
            return'{}()'.format(self.__class__.__name__)
