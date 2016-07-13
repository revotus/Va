"""Employing word net topic model."""
from collections import Counter
from va.features.api import FeatureExtractor


class WNTMFeatureExtractor(FeatureExtractor):

    """Class that uses word net topic model to extract features from docs."""

    def extract_feats(self, verbs, num_topics=100, **kwds):
        """Extract word net topic model features.

        Feats are number of topics.
        """
        topics = range(1, num_topics + 1)
        feats = Counter({topic: topic for topic in topics})
        return feats

    def rank_feats(self, feats, *args, **kwds):
        """No way to rank wntp features. All ae retained."""
        return feats

    def filter_feats(self, feats, *args, **kwds):
        """No need to filter since cannot be ranked."""
        return feats
