""" Wrapper for SVM from scikit-learn."""
from __future__ import unicode_literals

import numpy as np
from sklearn.svm.classes import SVC
from va.classify.api import ClassifierI

__all__ = ['SVMClassifier']


class SVMClassifier(ClassifierI):

    """Wrapper for scikit-learn svm classifier."""

    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0,
                 shrinking=True, probability=False, tol=1e-3, cache_size=200,
                 class_weight=None, verbose=False, max_iter=-1,
                 decision_function_shape=None, random_state=None):
        """Init. See scikit-learn."""
        self._clf = SVC(C=1, kernel=kernel, degree=degree, gamma=gamma,
                        coef0=coef0, shrinking=shrinking,
                        probability=probability, tol=tol, cache_size=cache_size,
                        class_weight=class_weight, verbose=verbose,
                        max_iter=max_iter,
                        decision_function_shape=decision_function_shape,
                        random_state=random_state)
        self.classes_ = None

    def __repr__(self):
        return "<SVMClassifier(%r)>" % self._clf

    def classify_many(self, vectors):
        """Classify a batch of verbs.

        :param vectors: An doc term array of vectors
        :return: The predicted class label for each input sample.
        :rtype: list
        """
        classes = self.classes_
        return [classes[i] for i in self._clf.predict(vectors)]

    def prob_classify_many(self, vectors):
        """Compute per-class probabilities for a batch of samples.
        :param vectors: A doc term array of vectors
        :rtype: list of ``ProbDistI``
        """
        y_proba_list = self._clf.predict_proba(vectors)
        return [self._make_probdist(y_proba) for y_proba in y_proba_list]

    def labels(self):
        """The class labels learned by this classifier.
        :rtype: list
        """
        return list(self.classes_)

    def train(self, vectors, labels):
        """
        Train (fit) the scikit-learn svm classifier.
        :param vectors: a doc-term array of vectors to learn from
        :param labels: a list of labels corresponding to the rows
        of the doc term array.
        """
        self.classes_, labels = np.unique(labels, return_inverse=True)
        self._clf.fit(vectors, labels)

        return self

    def _make_probdist(self, y_proba):
        classes = self.classes_
        return dict((classes[i], p) for i, p in enumerate(y_proba))
