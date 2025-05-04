import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, IncrementalPCA, FastICA
from sklearn.manifold import LocallyLinearEmbedding
import umap
from sklearn.cluster import DBSCAN, SpectralClustering, AgglomerativeClustering
import hdbscan
from sklearn.metrics import silhouette_score
import warnings
from data_import import download_emotion_csv

warnings.filterwarnings('ignore')


# Function to perform dimension reduction
def apply_dim_reduction(data, method, n_components):
    if method == 'pca':
        model = PCA(n_components=n_components, random_state=42)
    elif method == 'ipca':
        model = IncrementalPCA(n_components=n_components)
    elif method == 'ica':
        model = FastICA(n_components=n_components, random_state=42)
    elif method == 'lle':
        model = LocallyLinearEmbedding(n_components=n_components, n_neighbors=15, random_state=42)
    elif method == 'umap':
        model = umap.UMAP(n_components=n_components, random_state=42)

    return model.fit_transform(data)


# Function to perform clustering
def apply_clustering(data, method, params):
    if method == 'dbscan':
        eps = params.get('eps', 0.5)
        min_samples = params.get('min_samples', 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif method == 'hdbscan':
        min_cluster_size = params.get('min_cluster_size', 5)
        min_samples = params.get('min_samples', None)
        model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    elif method == 'spectral':
        n_clusters = params.get('n_clusters', 8)
        model = SpectralClustering(n_clusters=n_clusters, random_state=42)
    elif method == 'hierarchical':
        n_clusters = params.get('n_clusters', 8)
        linkage = params.get('linkage', 'ward')
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)

    labels = model.fit_predict(data)

    # Count valid clusters (excluding noise points labeled as -1)
    valid_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Calculate silhouette score if there are at least 2 valid clusters and no single cluster
    if valid_clusters >= 2 and len(set(labels)) > 1 and min(np.bincount(labels[labels >= 0])) > 1:
        try:
            # Use only non-noise points for silhouette calculation
            if -1 in labels:
                valid_points = labels != -1
                score = silhouette_score(data[valid_points], labels[valid_points])
            else:
                score = silhouette_score(data, labels)
            return score, labels, valid_clusters
        except:
            return -1, labels, valid_clusters
    else:
        return -1, labels, valid_clusters


try:
    # Read the CSV file
    df = download_emotion_csv()

    # Remove the last column
    df = df.iloc[:, :-1]

    # Remove null values
    df = df.dropna()

    # Take the top 60% of the data
    top_60_percent = int(len(df) * 0.6)
    df = df.iloc[:top_60_percent, :]

    # Extract features for dimension reduction and clustering
    X = df.values

    # Define dimension reduction methods
    dim_reduction_methods = ['umap', 'ipca', 'pca', 'ica', 'lle']
    dim_components_range = list(range(2, 11))  # 2 to 10 dimensions

    # Define clustering methods and their parameters
    clustering_methods = {
        'dbscan': {
            'eps': [0.1, 0.3, 0.5, 0.7, 1.0],
            'min_samples': [3, 5, 10, 15, 20]
        },
        'hdbscan': {
            'min_cluster_size': [3, 5, 10, 15, 20],
            'min_samples': [None, 3, 5, 10]
        },
        'spectral': {
            'n_clusters': list(range(2, 11))  # 2 to 10 clusters
        },
        'hierarchical': {
            'n_clusters': list(range(2, 11)),  # 2 to 10 clusters
            'linkage': ['ward', 'complete', 'average']
        }
    }

    # Store results
    results_dict = {}
    best_overall = {
        'silhouette': -1,
        'method': None,
        'params': {},
        'dim_method': '',
        'n_components': 0,
        'cluster_method': '',
        'n_clusters': 0
    }

    # For each dimension reduction method
    for dim_method in dim_reduction_methods:
        results_dict[dim_method] = {}

        # For each clustering method
        for cluster_method in clustering_methods:
            results_dict[dim_method][cluster_method] = {}

            # Create matrices to store best results and number of clusters
            silhouette_matrix = np.ones((len(dim_components_range), 0)) * -1
            clusters_matrix = np.zeros((len(dim_components_range), 0))
            param_combinations = []

            # Generate parameter combinations for each clustering method
            if cluster_method == 'dbscan':
                for eps in clustering_methods[cluster_method]['eps']:
                    for min_samples in clustering_methods[cluster_method]['min_samples']:
                        param_combinations.append({'eps': eps, 'min_samples': min_samples})
            elif cluster_method == 'hdbscan':
                for min_cluster_size in clustering_methods[cluster_method]['min_cluster_size']:
                    for min_samples in clustering_methods[cluster_method]['min_samples']:
                        param_combinations.append({'min_cluster_size': min_cluster_size, 'min_samples': min_samples})
            elif cluster_method == 'spectral':
                for n_clusters in clustering_methods[cluster_method]['n_clusters']:
                    param_combinations.append({'n_clusters': n_clusters})
            elif cluster_method == 'hierarchical':
                for n_clusters in clustering_methods[cluster_method]['n_clusters']:
                    for linkage in clustering_methods[cluster_method]['linkage']:
                        param_combinations.append({'n_clusters': n_clusters, 'linkage': linkage})

            # Initialize matrices with proper shapes
            silhouette_matrix = np.ones((len(dim_components_range), len(param_combinations))) * -1
            clusters_matrix = np.zeros((len(dim_components_range), len(param_combinations)), dtype=int)

            # For each dimension
            for i, n_comp in enumerate(dim_components_range):
                print(f"Processing {dim_method} with {n_comp} dimensions for {cluster_method}...")

                # Perform dimension reduction
                reduced_data = apply_dim_reduction(X, dim_method, n_comp)

                # For each parameter combination
                for j, params in enumerate(param_combinations):
                    # Apply clustering
                    score, labels, n_clusters = apply_clustering(reduced_data, cluster_method, params)

                    # Store results
                    silhouette_matrix[i, j] = score
                    clusters_matrix[i, j] = n_clusters

                    # Store detailed results
                    param_key = ', '.join([f"{k}={v}" for k, v in params.items()])
                    if n_comp not in results_dict[dim_method][cluster_method]:
                        results_dict[dim_method][cluster_method][n_comp] = {}
                    results_dict[dim_method][cluster_method][n_comp][param_key] = {
                        'silhouette': score,
                        'n_clusters': n_clusters
                    }

                    # Update best overall result
                    if score > best_overall['silhouette']:
                        best_overall['silhouette'] = score
                        best_overall['method'] = f"{dim_method}_{cluster_method}"
                        best_overall['params'] = params
                        best_overall['dim_method'] = dim_method
                        best_overall['n_components'] = n_comp
                        best_overall['cluster_method'] = cluster_method
                        best_overall['n_clusters'] = n_clusters

            # Create parameter labels for heatmap
            param_labels = []
            for params in param_combinations:
                label = ', '.join([f"{k}={v}" for k, v in params.items()])
                param_labels.append(label)

            # Create heatmap for silhouette scores
            plt.figure(figsize=(18, 10))

            # Replace -1 values with NaN for better visualization
            silhouette_matrix_display = np.where(silhouette_matrix == -1, np.nan, silhouette_matrix)

            # Create heatmap
            ax = sns.heatmap(silhouette_matrix_display, annot=True, fmt=".3f", cmap="viridis",
                             xticklabels=param_labels, yticklabels=dim_components_range,
                             cbar_kws={'label': 'Silhouette Score'})

            plt.title(f"{dim_method.upper()} + {cluster_method.upper()} - Silhouette Scores")
            plt.xlabel("Parameters")
            plt.ylabel("Number of Dimensions")
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.show()

            # Create heatmap for number of clusters
            plt.figure(figsize=(18, 10))
            ax = sns.heatmap(clusters_matrix, annot=True, fmt="d", cmap="YlGnBu",
                             xticklabels=param_labels, yticklabels=dim_components_range,
                             cbar_kws={'label': 'Number of Clusters'})

            plt.title(f"{dim_method.upper()} + {cluster_method.upper()} - Number of Clusters")
            plt.xlabel("Parameters")
            plt.ylabel("Number of Dimensions")
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.show()

            # Now create a simplified heatmap showing only the best silhouette score for each dimension and number of clusters
            # First, find the best configuration for each dimension
            best_scores = {}
            for i, n_comp in enumerate(dim_components_range):
                for j, params in enumerate(param_combinations):
                    score = silhouette_matrix[i, j]
                    n_clusters = clusters_matrix[i, j]

                    if score <= 0:  # Skip invalid scores
                        continue

                    key = (n_comp, n_clusters)
                    if key not in best_scores or score > best_scores[key]['score']:
                        best_scores[key] = {
                            'score': score,
                            'params': params
                        }

            # Create a dimension vs clusters matrix
            if best_scores:  # Check if we have valid results
                max_clusters = max([k[1] for k in best_scores.keys()]) if best_scores else 10
                dim_vs_clusters = np.ones((len(dim_components_range), max_clusters + 1)) * -1

                for (n_comp, n_clusters), data in best_scores.items():
                    if n_clusters > 0:  # Skip invalid clusters
                        dim_idx = dim_components_range.index(n_comp)
                        dim_vs_clusters[dim_idx, n_clusters] = data['score']

                # Create heatmap
                plt.figure(figsize=(14, 8))

                # Replace -1 values with NaN for better visualization
                dim_vs_clusters_display = np.where(dim_vs_clusters == -1, np.nan, dim_vs_clusters)

                ax = sns.heatmap(dim_vs_clusters_display, annot=True, fmt=".3f", cmap="viridis",
                                 xticklabels=range(max_clusters + 1), yticklabels=dim_components_range,
                                 cbar_kws={'label': 'Silhouette Score'})

                plt.title(
                    f"{dim_method.upper()} + {cluster_method.upper()} - Best Silhouette Scores by Dimension and Clusters")
                plt.xlabel("Number of Clusters")
                plt.ylabel("Number of Dimensions")
                plt.tight_layout()
                plt.show()

    # Print optimal parameters for each method combination
    print("\n===== OPTIMAL PARAMETERS FOR EACH METHOD =====")

    for dim_method in dim_reduction_methods:
        for cluster_method in clustering_methods:
            best_score = -1
            best_config = {}

            for n_comp in results_dict[dim_method][cluster_method]:
                for param_key, result in results_dict[dim_method][cluster_method][n_comp].items():
                    if result['silhouette'] > best_score:
                        best_score = result['silhouette']
                        best_config = {
                            'n_components': n_comp,
                            'params': param_key,
                            'n_clusters': result['n_clusters']
                        }

            if best_score > -1:
                print(f"{dim_method.upper()} (dims={best_config['n_components']}) + "
                      f"{cluster_method.upper()} (clusters={best_config['n_clusters']}) - "
                      f"Parameters: {best_config['params']}, Silhouette Score: {best_score:.4f}")

    # Print overall best result
    print("\n===== BEST OVERALL COMBINATION =====")
    print(f"Dimension Reduction: {best_overall['dim_method'].upper()} with {best_overall['n_components']} dimensions")
    print(f"Clustering Method: {best_overall['cluster_method'].upper()} with {best_overall['n_clusters']} clusters")
    print(f"Parameters: {', '.join([f'{k}={v}' for k, v in best_overall['params'].items()])}")
    print(f"Silhouette Score: {best_overall['silhouette']:.4f}")

except Exception as e:
    print(f"Error: {str(e)}")