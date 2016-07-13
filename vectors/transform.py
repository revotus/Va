"""Classes for transforming vectors and vector spaces."""
from __future__ import division
import logging
import numpy as np
from math import sqrt

logger = logging.getLogger(__name__)


class SVD(object):

    """Uses Singular Value Decomposition to transform vector space."""

    def __init__(self):
        """Init."""
        self._Tt = None

    def transform_space(self, vectors, dimensions=None):
        """Transform vector space using SVD."""
        if dimensions and dimensions < len(vectors[0]):
            [u, d, vt] = np.linalg.svd(np.transpose(np.array(vectors)))
            S = d[:dimensions] * np.identity(dimensions, np.float64)
            T = u[:, :dimensions]
            Dt = vt[:dimensions, :]
            vectors = np.transpose(np.dot(S, Dt))
            self._Tt = np.transpose(T)

        return vectors

    def transform_vector(self, vector):
        """Transform new vector using info from SVD of space."""
        if self._Tt is not None:
            return np.dot(self._Tt, vector)


def normalize(vector):
    """Normalize vector to unit length."""
    return vector / sqrt(np.dot(vector, vector))


def create_doc_term_mat(vectors):
    """Create doc term numpy array matrix."""
    logger.info('Transforming %d vectors into np doc-term matrix...',
                len(vectors))

    doc_term_mat = np.array([[vector[feat] for feat in vector]
                             for vector in vectors])

    return doc_term_mat
