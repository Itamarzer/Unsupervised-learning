import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA, FastICA
from sklearn.manifold import LocallyLinearEmbedding, TSNE
import umap
from sklearn.metrics import silhouette_score
from clustering_methods import kmeans, gmm, miniBatchKmeans, fuzzy_kmeans, birch
from tqdm import tqdm
from data_import import download_emotion_csv


# Load and preprocess data
def load_and_preprocess_data():
    df = download_emotion_csv()
    df = df.dropna()
    df = df.iloc[:, :-1]  # Drop last column
    df = df.iloc[:int(0.6 * len(df))]  # Take top 60%
    return df.values


# Dimension reduction methods
def get_dimension_reduction_methods():
    return {
        'PCA': lambda n: PCA(n_components=n, random_state=0),
        'IPCA': lambda n: IncrementalPCA(n_components=n),
        'ICA': lambda n: FastICA(n_components=n, random_state=0, max_iter=1000),
        'UMAP': lambda n: umap.UMAP(n_components=n, random_state=0),
        'LLE': lambda n: LocallyLinearEmbedding(n_components=n, random_state=0),
        'TSNE': lambda n: TSNE(n_components=n, random_state=0, n_iter=1000)
    }


# Clustering methods
CLUSTERING_METHODS = {
    'KMeans': (kmeans, 8),
    'GMM': (gmm, 8),
    'MiniBatchKMeans': (miniBatchKmeans, 10),
    'FuzzyKMeans': (fuzzy_kmeans, 7),
    'Birch': (birch, 6),
}


# Main function
def find_optimal_dimensions():
    print("=====starting dimensionality reduction analysis=====")
    data = load_and_preprocess_data()
    reduction_methods = get_dimension_reduction_methods()

    dims_list = list(range(2, 21)) + list(range(25, 126, 5))

    # Track the best method for each clustering algorithm
    best_methods = {}

    for cluster_name, (cluster_func, n_clusters) in CLUSTERING_METHODS.items():
        print(f"\n=== Clustering Method: {cluster_name} ===")
        best_overall_score = -1
        best_overall_method = None
        best_overall_dim = None

        for reduction_name, reduction_func in tqdm(reduction_methods.items(), desc=f"Cluster: {cluster_name}",
                                                   position=0):
            best_score = -1
            best_dim = None

            for n_dim in tqdm(dims_list, desc=f"{reduction_name}", leave=False, position=1):
                try:
                    reducer = reduction_func(n_dim)
                    reduced_data = reducer.fit_transform(data)

                    # Some methods like TSNE may ignore n_dim
                    if reduced_data.shape[1] != n_dim and reduction_name != 'TSNE':
                        continue

                    # Clustering
                    clusterer, labels = cluster_func(reduced_data, n_clusters=n_clusters)

                    # Silhouette score
                    if len(np.unique(labels)) > 1:
                        score = silhouette_score(reduced_data, labels)
                        if score > best_score:
                            best_score = score
                            best_dim = n_dim

                        # Track the best overall method
                        if score > best_overall_score:
                            best_overall_score = score
                            best_overall_method = reduction_name
                            best_overall_dim = n_dim
                except Exception as e:
                    # Skip if failed
                    continue

            if best_dim is not None:
                print(f"{reduction_name}: Best dimension = {best_dim} with Silhouette Score = {best_score:.4f}")
            else:
                print(f"{reduction_name}: No valid clustering result.")

        # Store the best method for this clustering algorithm
        if best_overall_method is not None:
            best_methods[cluster_name] = (best_overall_method, best_overall_dim, best_overall_score)

    # Print the best dimension reduction method for each clustering algorithm
    print("\n=== Best Dimension Reduction Method for Each Clustering Algorithm ===")
    for cluster_name, (best_method, best_dim, best_score) in best_methods.items():
        print(f"{cluster_name}: {best_method} with {best_dim} dimensions (Silhouette Score: {best_score:.4f})")

