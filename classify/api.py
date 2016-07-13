"""
Interfaces for labeling vectors with category labels (or "class labels").

``ClassifierI`` is a standard interface for "single-category
classification", in which the set of categories is known, the number
of categories is finite, and each text belongs to exactly one
category.

``MultiClassifierI`` is a standard interface for "multi-category
classification", which is like single-category classification except
that each text belongs to zero or more categories.
"""


class ClassifierI(object):

    """Main api for classifiers.

    A processing interface for labeling tokens with a single category
    label (or "class").  Labels are typically strs or
    ints, but can be any immutable type.  The set of labels
    that the classifier chooses from must be fixed and finite.
    Subclasses must define:
        - ``labels()``
        - either ``classify()`` or ``classify_many()`` (or both)
    Subclasses may define:
        - either ``prob_classify()`` or ``prob_classify_many()`` (or both)
    """

    def labels(self):
        """Return list of labels.

        :return: the list of category labels used by this classifier.
        :rtype: list of (immutable)
        """
        raise NotImplementedError()

    def classify(self, vector):
        """Classify vector given model.

        :return: the most appropriate label for the given vector.
        :rtype: label
        """
        if self.classify_many:
            return self.classify_many([vector])[0]
        else:
            raise NotImplementedError()

    def prob_classify(self, vector):
        """Classify with a probability distribution of labels for a vector.

        :return: a probability distribution over labels for the given
        vector.
        :rtype: ProbDistI
        """
        if self.prob_classify_many:
            return self.prob_classify_many([vector])[0]
        else:
            raise NotImplementedError()

    def classify_many(self, vectors):
        """Classify multiple vectors.

        Apply ``self.classify()`` to each element of ``vectors``.  I.e.:
        return [self.classify(vec) for vec in vectors]
        :rtype: list(label)
        """
        return [self.classify(vec) for vec in vectors]

    def prob_classify_many(self, vectors):
        """Classify with prob dist for each vector in vectors.

        Apply ``self.prob_classify()`` to each element of ``vectors``.  I.e.:
        return [self.prob_classify(vec) for vec in vectors]
        :rtype: list(ProbDistI)
        """
        return [self.prob_classify(vec) for vec in vectors]


class MultiClassifierI(object):

    """Same as above, but for vectors that have 0 or more labels.

    A processing interface for labeling vectors with zero or more
    category labels (or "labels").  Labels are typically strs
    or ints, but can be any immutable type.  The set of labels
    that the multi-classifier chooses from must be fixed and finite.
    Subclasses must define:
        - ``labels()``
        - either ``classify()`` or ``classify_many()`` (or both)
    Subclasses may define:
        - either ``prob_classify()`` or ``prob_classify_many()`` (or both)
    """

    def labels(self):
        """Return list of labels.

        :return: the list of category labels used by this classifier.
        :rtype: list of (immutable)
        """
        raise NotImplementedError()

    def classify(self, vector):
        """Classify vector with 0 or more labels.

        :return: the most appropriate set of labels for the given vector.
        :rtype: set(label)
        """
        if self.classify_many:
            return self.classify_many([vector])[0]
        else:
            raise NotImplementedError()

    def prob_classify(self, vector):
        """Classify a vector with a prob distribution set of labels.

        :return: a probability distribution over sets of labels for the
        given vector.
        :rtype: ProbDistI
        """
        if self.prob_classify_many:
            return self.prob_classify_many([vector])[0]
        else:
            raise NotImplementedError()

    def classify_many(self, vectors):
        """Classify multiple vectors.

        Apply ``self.classify()`` to each element of ``vectors``.  I.e.:
        return [self.classify(vec) for vec in vectors]
        :rtype: list(set(label))
        """
        return [self.classify(vec) for vec in vectors]

    def prob_classify_many(self, vectors):
        """Classify multiple vectors with prob distribution.

        Apply ``self.prob_classify()`` to each element of ``vectors``.  I.e.:
        return [self.prob_classify(vec) for vec in vectors]
        :rtype: list(ProbDistI)
        """
        return [self.prob_classify(vec) for vec in vectors]
