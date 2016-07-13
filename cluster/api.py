"""Api for Clusterer objects."""
from va.vectors.transform import normalize, SVD


class ClusterI(object):

    """Interface covering basic clustering functionality."""

    def cluster(self, vectors, assign_clusters=False):
        """Assign the vectors to clusters.

        Assign the vectors to clusters, learning clustering parameters
        from the data. Returns a cluster identifier for each vector.
        """
        raise NotImplementedError()

    def classify(self, vector):
        """Classify the vector into a cluster.

        Classify the vector into a cluster, setting the token's CLUSTER
        parameter to that cluster identifier.
        """
        raise NotImplementedError()

    def likelihood(self, vector, label):
        """Return the likelihood of the vector having the cluster label."""
        if self.classify(vector) == label:
            return 1.0
        else:
            return 0.0

    def classification_probdist(self, vector):
        """Classify the vector into a cluster.

        Classify the vector into a cluster, returning
        a probability distribution over the cluster identifiers.
        """
        likelihoods = {}
        sum_ = 0.0
        for cluster in self.cluster_names():
            likelihoods[cluster] = self.likelihood(vector, cluster)
            sum_ += likelihoods[cluster]
            for cluster in self.cluster_names():
                likelihoods[cluster] /= sum_

        # TODO: look into nltk's DictionaryProbDist
        return likelihoods

    def num_clusters(self):
        """ Return the number of clusters."""
        raise NotImplementedError()

    def cluster_names(self):
        """Return the names of the clusters."""
        return list(range(self.num_clusters()))

    def cluster_name(self, index):
        """Return the names of the cluster at index."""
        cluster_names = self.cluster_names()
        return cluster_names[index]


class VectorSpaceClusterer(ClusterI):

    """Abstract clusterer which takes tokens and maps them into a vector space.

    Optionally performs singular value decomposition to reduce the
    dimensionality.
    """

    transforms = {'svd': SVD}

    def __init__(self, normalise=False, transform='svd', dimensions=None):
        """Init."""
        self._should_normalise = normalise
        self._transform = self.transforms[transform]()
        self._dimensions = dimensions

    def cluster(self, vectors, assign_clusters=False):
        """Cluster with optional normalization and transform."""
        assert len(vectors) > 0

        if self._should_normalise:
            vectors = [normalize(vector) for vector in vectors]

        vectors = self._transform.transform_space(vectors, self._dimensions)

        # call abstract method to cluster vectors
        self.cluster_vectorspace(vectors)

        # assign the vectors to clusters
        if assign_clusters:
            return [self.classify(vector) for vector in vectors]

    def cluster_vectorspace(self, vectors):
        """Find the clusters using the given set of vectors."""
        raise NotImplementedError()

    def classify(self, vector):
        """Classify vector into cluster."""
        vector = self.vector(vector)
        cluster = self.classify_vectorspace(vector)
        return self.cluster_name(cluster)

    def classify_vectorspace(self, vector):
        """Return the index of the appropriate cluster for the vector."""
        raise NotImplementedError()

    def likelihood(self, vector, label):
        """Return of vector having label."""
        vector = self.vector(vector)

    def likelihood_vectorspace(self, vector, cluster):
        """Return the likelihood of vector belong to a cluster."""
        predicted = self.classify_vectorspace(vector)
        return 1.0 if cluster == predicted else 0.0

    def vector(self, vector):
        """Return vector after normalization and deimensionality reduction."""
        if self._should_normalise:
            vector = normalize(vector)
        vector = self._transform.transform_vector(vector)
        return vector
