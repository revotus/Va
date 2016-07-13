#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Clustering."""
from __future__ import unicode_literals
from collections import defaultdict
import numpy as np
from sklearn import cluster
# from sklearn import mixture
from va.cluster.api import VectorSpaceClusterer
from va.containers import Verbatims
from va.util import logged


@logged
class KMeansClusterer(VectorSpaceClusterer):

    """Uses KMeans to cluster vectors.

    The K-means clusterer starts with k arbitrary chosen means then allocates
    each vector to the cluster with the closest mean. It then recalculates the
    means of each cluster as the centroid of the vectors in the cluster. This
    process repeats until the cluster memberships stabilise. This is a
    hill-climbing algorithm which may converge to a local maximum. Hence the
    clustering is often repeated with random initial means and the most
    commonly occurring output means are chosen.
    """

    def __init__(self, n_clusters=8, init='k-means++', n_init=10,
                 max_iter=300, tol=0.0001, precompute_distances='auto',
                 verbose=0, random_state=None, copy_x=True, n_jobs=1,
                 normalise=False, transform='svd', dimensions=None):
        """Init."""
        super(KMeansClusterer, self).__init__(normalise, transform, dimensions)
        self.n_clusters = n_clusters
        self.model = cluster.KMeans(n_clusters=n_clusters, init=init,
                                    n_init=n_init, max_iter=max_iter, tol=tol,
                                    precompute_distances=precompute_distances,
                                    verbose=verbose, random_state=random_state,
                                    copy_x=copy_x, n_jobs=n_jobs)
        self.feats_list = None

    def cluster_vectorspace(self, vectors, samples=None):
        """Fit model to vectors."""
        self.model.fit(vectors)
        self.clustering_ = self.get_clustering(samples)
        return self

    def classify_vectorspace(self, vector):
        """Classify vector using fitted model."""
        prediction = self.model.predict(vector)
        return prediction

    def distances_from_centroid(self, vectors):
        """Compute distance from centroid for each vector."""
        distances = []
        for i, label in enumerate(self.model.labels_):
            distance = np.linalg.norm(vectors[i] -
                                      self.model.cluster_centers_[label])
            distances.append(distance)
        return distances

    def num_clusters(self):
        """Return number of clusters in model."""
        return len(self.model.cluster_centers_)

    def get_clustering(self, samples):
        """Return clustering of verbs."""
        if samples is None:
            samples = range(len(self.model.labels_))
        clustering = defaultdict(list)
        for idx, label in enumerate(self.model.labels_):
            clustering[label].append(samples[idx])
        for label in clustering:
            clustering[label] = Verbatims(clustering[label])
        return clustering

    def top_n_feats(self, feats, n=5, starting_label=0):
        """Change labels of clusters to top n feature names for cluster."""
        centroids_feat_indices = self.model.cluster_centers_.argsort()
        top_n_feats = dict()
        for label in self.clustering_:
            ordered_centroid = centroids_feat_indices[label, ::-1]
            top_n_feats[label + starting_label] = [feats[ind] for ind
                                                   in ordered_centroid[:n]]

        return top_n_feats
