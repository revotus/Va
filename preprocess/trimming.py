#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Different methods of trimming verb sets."""
from __future__ import unicode_literals
import random
import logging
from collections import Counter

logger = logging.getLogger(__name__)
# TODO: decide if shuffle belongs in defs


def trim_verbs_per(verbs, code_count=0):
    """"Trim Verbatims instance so that all codes are equally represented."""
    random.shuffle(verbs)

    if not code_count or code_count > min(verbs.codes()):
        code_count = min(verbs.codes())
    logger.info('Trimming verbs to %d verbs per code', code_count)

    codes_seen = Counter()

    for verb in verbs:
        if codes_seen[verb.code] < code_count:
            codes_seen[verb.code] += 1
        else:
            del verbs[verb]

    return verbs


def trim_verbs_tot(verbs, verb_count=0):
    """Trim Verbatims instance to a total number of verbs."""
    random.shuffle(verbs)

    if not verb_count or verb_count > len(verbs):
        verb_count = len(verbs)
    logger.info('Trimming verbs to %d total verbs...', verb_count)

    verbs = verbs[:verb_count]

    return verbs


def trim_verbs_prop(verbs, verb_count=0):
    """Trim Verbatim instance so that trimmed set has same code proportion."""
    random.shuffle(verbs)

    if not verb_count or verb_count > len(verbs):
        verb_count = len(verbs)
    logger.info('Trimming verbs to %d total verbs...', verb_count)

    code_count = verbs.codes()

    # reduce total count while keeping same proportion
    code_sum = sum(code_count.values())
    for code in code_count:
        code_count[code] /= code_sum
        code_count[code] *= verb_count

    codes_seen = Counter()

    for verb in verbs:
        if codes_seen[verb.code] <= code_count[verb.code]:
            codes_seen[verb.code] += 1
        else:
            del verbs[verb]

    return verbs
