import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from kneed import KneeLocator
from data_import import download_emotion_csv
from clustering_methods import gmm, kmeans, birch, miniBatchKmeans, fuzzy_kmeans
import matplotlib.pyplot as plt
import umap
import os


def format_p_value(p_value):
    """Format p-values to avoid displaying them as zero."""
    if p_value < 1e-6:
        return "p-value < 1e-6"  # For extremely small p-values
    elif p_value < 0.0001:
        return f"{p_value:.2e}"  # Use scientific notation for very small values
    else:
        return f"{p_value:.6f}"  # Show 6 decimal places for other values


def load_and_preprocess_data():
    # Read the CSV file
    try:
        df = download_emotion_csv()
        print(f"Data loaded successfully. Original shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    # Remove rows with null values
    df_clean = df.dropna()
    print(f"Data shape after removing null values: {df_clean.shape}")

    # Separate features and labels (last column)
    features = df_clean.iloc[:, :-1].values
    labels = df_clean.iloc[:, -1].values

    # Take top 60% of the data
    n_samples = len(features)
    train_size = int(n_samples * 0.6)
    features_train = features[:train_size]
    labels_train = labels[:train_size]

    print(f"Train data shape (60% of cleaned data): {features_train.shape}")

    return features_train, labels_train


def find_optimal_k(clustering_func, data, k_range=range(2, 21)):
    """
    Function to find optimal k for clustering methods using the elbow method
    """
    silhouette_scores = []
    inertias = []
    valid_ks = []

    for k in tqdm(k_range, desc=f"Finding optimal k for {clustering_func.__name__}"):
        try:
            model, labels = clustering_func(data, n_clusters=k)

            if len(np.unique(labels)) > 1:
                score = silhouette_score(data, labels)
                silhouette_scores.append(score)
                valid_ks.append(k)

                # Calculate inertia (either from model attribute or manually)
                if hasattr(model, 'inertia_'):
                    inertias.append(model.inertia_)
                else:
                    try:
                        if hasattr(model, 'cluster_centers_'):
                            centers = model.cluster_centers_
                        elif hasattr(model, 'means_'):
                            centers = model.means_
                        else:
                            centers = np.array([data[labels == i].mean(axis=0) for i in range(k)
                                                if np.sum(labels == i) > 0])

                        pseudo_inertia = 0
                        for i in range(len(np.unique(labels))):
                            if i == -1:
                                continue
                            cluster_points = data[labels == i]
                            if len(cluster_points) > 0 and i < len(centers):
                                squared_distances = np.sum((cluster_points - centers[i]) ** 2, axis=1)
                                pseudo_inertia += np.sum(squared_distances)

                        inertias.append(pseudo_inertia)
                    except Exception as e:
                        print(f"Error calculating pseudo-inertia: {e}")
                        inertias.append(np.nan)
        except Exception as e:
            print(f"Error in clustering with k={k}: {e}")
            continue

    best_k_silhouette = valid_ks[np.argmax(silhouette_scores)] if silhouette_scores else None
    best_score = max(silhouette_scores) if silhouette_scores else 0

    best_k_elbow = None
    if inertias and not np.isnan(inertias).all() and len(valid_ks) > 2:
        try:
            kn = KneeLocator(valid_ks, inertias, curve="convex", direction="decreasing")
            best_k_elbow = kn.knee
        except Exception as e:
            print(f"Error in elbow analysis: {e}")

    print(f"Best silhouette k for {clustering_func.__name__}: {best_k_silhouette} (score={best_score:.4f})")
    print(f"Elbow k for {clustering_func.__name__}: {best_k_elbow}")

    return best_k_elbow, best_score


def find_best_clustering_method(data):
    """
    Find the best clustering method based on silhouette score.
    Returns the best method, optimal k, and score.
    """
    clustering_methods = {
        'kmeans': kmeans,
        'gmm': gmm,
        'birch': birch,
        'miniBatchKmeans': miniBatchKmeans,
        'fuzzy_kmeans': fuzzy_kmeans
    }

    results = {}

    for method_name, method_func in clustering_methods.items():
        print(f"\nProcessing {method_name}...")
        try:
            optimal_k, score = find_optimal_k(method_func, data)

            if optimal_k:
                model, labels = method_func(data, n_clusters=optimal_k)
                actual_score = silhouette_score(data, labels) if len(np.unique(labels)) > 1 else 0

                results[method_name] = {
                    'optimal_k': optimal_k,
                    'silhouette_score': actual_score,
                    'model': model,
                    'labels': labels
                }

                print(f"Validated silhouette score for {method_name} with k={optimal_k}: {actual_score:.4f}")
            else:
                print(f"No optimal k found for {method_name}")
        except Exception as e:
            print(f"Error processing {method_name}: {e}")

    best_method = None
    best_k = None
    best_score = -1
    best_labels = None
    best_model = None

    for method_name, result in results.items():
        if result['silhouette_score'] > best_score:
            best_score = result['silhouette_score']
            best_method = method_name
            best_k = result['optimal_k']
            best_labels = result['labels']
            best_model = result['model']

    print(f"\nBest clustering method: {best_method} with k={best_k} (score={best_score:.4f})")
    return best_method, best_k, best_score, clustering_methods[best_method], best_labels, best_model


def visualize_clusters_with_umap(data, labels, method_name, k, output_dir='plots'):
    """
    Visualize clustering results using UMAP for dimensionality reduction
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Apply UMAP for dimensionality reduction
    print("\nApplying UMAP dimensionality reduction...")
    reducer = umap.UMAP(random_state=42)
    try:
        embedding = reducer.fit_transform(data)

        # Plot the embedding with cluster colors
        plt.figure(figsize=(12, 10))

        # Get unique cluster labels
        unique_labels = np.unique(labels)

        # Create a colormap
        cmap = plt.cm.get_cmap('tab10', len(unique_labels))

        # Plot each cluster
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(embedding[mask, 0], embedding[mask, 1],
                        c=[cmap(i)], label=f'Cluster {label}',
                        alpha=0.7, s=10)

        plt.title(f'{method_name} Clustering (k={k}) with UMAP Visualization')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # Save the plot
        output_path = os.path.join(output_dir, f'{method_name}_k{k}_umap_visualization.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
        plt.close()

        return True
    except Exception as e:
        print(f"Error in UMAP visualization: {e}")
        return False


def test_split_clusters_hypothesis(features, best_method_func, optimal_k):
    print("\n--- Testing Hypothesis on Split Data ---")

    n_samples, n_features = features.shape
    half_features = n_features // 2

    first_half = features[:, :half_features]
    second_half = features[:, half_features:2 * half_features] if n_features % 2 == 0 else features[:, half_features:]

    print(f"Split features into two halves:")
    print(f"First half shape: {first_half.shape}")
    print(f"Second half shape: {second_half.shape}")

    print("\nClustering first half...")
    _, labels_first = best_method_func(first_half, n_clusters=optimal_k)

    print("Clustering second half...")
    _, labels_second = best_method_func(second_half, n_clusters=optimal_k)

    contingency_table = np.zeros((optimal_k, optimal_k))
    for i in range(n_samples):
        contingency_table[labels_first[i], labels_second[i]] += 1

    print("\n--- Contingency Table of Cluster Assignments ---")
    print("Rows: First half clusters, Columns: Second half clusters")
    print(contingency_table)

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    print(f"\nChi-square test:")
    print(f"Chi2={chi2}, p={format_p_value(p_value)}, dof={dof}")
    print(f"Conclusion: {'Reject' if p_value < 0.05 else 'Fail to reject'} H0 at α=0.05")

    same_cluster_normalized = 0
    for i in range(n_samples):
        if labels_first[i] == labels_second[i] % optimal_k:
            same_cluster_normalized += 1

    print(f"\nSamples where halves are in corresponding clusters: {same_cluster_normalized} "
          f"({same_cluster_normalized / n_samples * 100:.2f}%)")

    print("\n--- Cluster Composition Analysis ---")

    cluster_compositions = []

    combined_data = np.vstack((first_half, second_half))
    combined_labels = np.concatenate((labels_first, np.array([f"second_{i}" for i in labels_second])))

    _, combined_clusters = best_method_func(combined_data, n_clusters=optimal_k)

    half_labels = np.array(['first'] * n_samples + ['second'] * n_samples)

    for cluster in range(optimal_k):
        mask = combined_clusters == cluster
        cluster_half_counts = np.bincount(np.array([1 if half == 'second' else 0 for half in half_labels[mask]]))

        if len(cluster_half_counts) > 0:
            total = np.sum(cluster_half_counts)
            if total > 0:
                first_percent = cluster_half_counts[0] / total * 100 if len(cluster_half_counts) > 0 else 0
                second_percent = cluster_half_counts[1] / total * 100 if len(cluster_half_counts) > 1 else 0

                print(f"Cluster {cluster}: {total} samples - "
                      f"First half: {first_percent:.1f}%, Second half: {second_percent:.1f}%")

                cluster_compositions.append((first_percent, second_percent))

    dominance_percent = np.mean([max(comp) for comp in cluster_compositions])
    print(f"\nAverage percentage of dominant half in clusters: {dominance_percent:.1f}%")

    above_80_percent = sum(1 for comp in cluster_compositions if max(comp) >= 80)

    binom_test = stats.binomtest(above_80_percent, len(cluster_compositions), p=0.5)

    print(f"\nClusters with ≥80% dominance by one half: {above_80_percent}/{len(cluster_compositions)} "
          f"({above_80_percent / len(cluster_compositions) * 100:.1f}%)")
    print(f"Binomial test p-value: {format_p_value(binom_test.pvalue)}")
    print(f"Conclusion: {'Reject' if binom_test.pvalue < 0.05 else 'Fail to reject'} H0 at α=0.05")

    return {
        'chi2_test': {'chi2': chi2, 'p_value': p_value},
        'same_cluster_percentage': same_cluster_normalized / n_samples * 100,
        'dominance_percent': dominance_percent,
        'binomial_test': {'p_value': binom_test.pvalue}
    }


def explore1():
    # Load and preprocess data
    features, labels = load_and_preprocess_data()

    if features is None:
        print("Error: Could not load or process the data.")
        return

    # Split features into halves
    n_samples, n_features = features.shape
    half_features = n_features // 2
    first_half = features[:, :half_features]
    second_half = features[:, half_features:2 * half_features] if n_features % 2 == 0 else features[:, half_features:]

    # Stack both halves vertically
    combined_halves = np.vstack((first_half, second_half))

    # Run clustering optimization on the stacked halves
    best_method, optimal_k, best_score, best_method_func, best_labels, best_model = find_best_clustering_method(
        combined_halves)

    # Visualize the best clustering result using UMAP
    visualize_clusters_with_umap(combined_halves, best_labels, best_method, optimal_k)

    # Test the hypothesis using original features and the optimized method
    results = test_split_clusters_hypothesis(features, best_method_func, optimal_k)

    # Print summary
    print("\n--- Summary of Results ---")
    print(f"Best clustering method: {best_method} with k={optimal_k} (score={best_score:.4f})")
    print(f"Chi-square test p-value: {format_p_value(results['chi2_test']['p_value'])}")
    print(f"Same cluster percentage: {results['same_cluster_percentage']:.2f}%")
    print(f"Average dominance percentage: {results['dominance_percent']:.2f}%")
    print(f"Binomial test p-value: {format_p_value(results['binomial_test']['p_value'])}")
