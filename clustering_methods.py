from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, Birch, AgglomerativeClustering, MiniBatchKMeans, DBSCAN, SpectralClustering
import numpy as np
from utils import RANDOM_STATE
import skfuzzy as fuzz


# functions return [clustering_method], [labels]
def gmm(data, n_clusters=16):
    gmm_method = GaussianMixture(n_components=n_clusters, random_state=RANDOM_STATE)
    gmm_method.fit(data)
    return gmm_method, gmm_method.predict(data)


def kmeans(data, n_clusters=16):
    kmeans_method = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
    kmeans_method.fit(data)
    return kmeans_method, kmeans_method.labels_


def birch(data, n_clusters=16):
    np.random.seed(RANDOM_STATE)
    birch_method = Birch(n_clusters=n_clusters)
    birch_method.fit(data)
    return birch_method, birch_method.labels_


def agglomerative(data, n_clusters=16):
    np.random.seed(RANDOM_STATE)
    agglomerative_method = AgglomerativeClustering(n_clusters=n_clusters)
    agglomerative_method.fit(data)
    return agglomerative_method, agglomerative_method.labels_


def hierarchical(data, n_clusters=16):
    # This is another name for AgglomerativeClustering
    np.random.seed(RANDOM_STATE)
    hierarchical_method = AgglomerativeClustering(n_clusters=n_clusters)
    hierarchical_method.fit(data)
    return hierarchical_method, hierarchical_method.labels_


def miniBatchKmeans(data, n_clusters=16):
    mini_batch_kmeans_method = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=100,         # Updated from 50
        max_iter=100,
        init='k-means++',       # Updated from 'random'
        tol=0.0001,
        max_no_improvement=20,  # Updated from 10
        reassignment_ratio=0.01,
        random_state=RANDOM_STATE
    )
    mini_batch_kmeans_method.fit(data)
    return mini_batch_kmeans_method, mini_batch_kmeans_method.labels_


def dbscan(data, eps=0.5, min_samples=5):
    dbscan_method = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan_method.fit_predict(data)
    return dbscan_method, labels


def fuzzy_kmeans(data, n_clusters=16, m=1.1, error=0.0001, maxiter=100):
    # Transpose data for skfuzzy compatibility
    data_t = data.T

    # Initialize fuzzy c-means
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data_t, n_clusters, m, error, maxiter, seed=RANDOM_STATE
    )

    # Get the labels from the membership matrix
    labels = np.argmax(u, axis=0)

    # Create a method object to return (similar structure to other methods)
    class FuzzyCMeans:
        def __init__(self, centers, memberships):
            self.cluster_centers_ = centers
            self.u = memberships  # Membership matrix

    fuzzy_method = FuzzyCMeans(cntr, u)
    return fuzzy_method, labels


def spectral(data, n_clusters=16, affinity='rbf'):
    spectral_method = SpectralClustering(
        n_clusters=n_clusters,
        affinity=affinity,
        random_state=RANDOM_STATE
    )
    labels = spectral_method.fit_predict(data)
    return spectral_method, labels


CLUSTERING_METHODS_FUNCTIONS_DICT = {
    'gmm': gmm,
    'kmeans': kmeans,
    'birch': birch,
    'minibatchkmeans': miniBatchKmeans,
    'hierarchical': hierarchical,
    'agglomerative': agglomerative,
    'dbscan': dbscan,
    'fuzzy_kmeans': fuzzy_kmeans,
    'spectral': spectral
}

CLUSTERING_METHODS_OPTIMAL_CLUSTERS_NUM = {
    'gmm': 29,
    'kmeans': 14,
    'birch': 32,
    'minibatchkmeans': 20,
    'hierarchical': 15,
    'agglomerative': 15,
    'dbscan': None,  # DBSCAN automatically determines the number of clusters
    'fuzzy_kmeans': 14,
    'spectral': 16
}

CLUSTERING_METHODS_OPTIMAL_DIMS_NUM = {
    'gmm': 70,
    'kmeans': 50,
    'birch': 20,
    'minibatchkmeans': 70,
    'hierarchical': 50,
    'agglomerative': 50,
    'dbscan': 50,
    'fuzzy_kmeans': 50,
    'spectral': 50
}

# For methods that don't use n_clusters, you might want to add default parameters
CLUSTERING_METHODS_DEFAULT_PARAMS = {
    'dbscan': {'eps': 0.5, 'min_samples': 5},
    'spectral': {'affinity': 'rbf'}
}