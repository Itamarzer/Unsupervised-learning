from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import KMeans, Birch, MiniBatchKMeans

##### CONSTANTS
CLUSTERING_METHODS_NAMES_LIST = ['gmm', 'kmeans', 'birch', 'minibatchkmeans']

CLUSTERING_METHODS_PLOT_NAMES_DICT = {
    'gmm': 'GMM',
    'kmeans': 'KMeans',
    'birch': 'Birch',
    'minibatchkmeans': 'Minibatch KMeans'
}

CLUSTERING_METHODS_DICT = {
    'gmm': GaussianMixture(),
    'kmeans': KMeans(),
    'birch': Birch(),
    'minibatchkmeans': MiniBatchKMeans()
}

CLUSTERING_COLORS = ['royalblue', 'slategrey', 'limegreen', 'deeppink']

ANOMALY_DETECTION_MODELS = ['isolation_forest']

DIMENSIONALITY_REDUCTIONS_NAMES_LIST = ['']
DATA_SIZE = 11024  # not used here
TRAIN_DATA_PERCENTAGE = 0.6
TEST_DATA_PERCENTAGE = 0.4
RANDOM_STATE = 42

CLUSTERS_NUM_LIST = list(range(2, 21, 3))  # from 2 to 20, step 3
DIMS_NUM_LIST = [5, 10, 20, 50, 70, 90, 128]
FEATURES_AMOUNT = 128  # not used here

