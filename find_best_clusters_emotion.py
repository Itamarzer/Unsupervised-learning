import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
import clustering_methods as cm
from tqdm import tqdm
from kneed import KneeLocator
from data_import import download_emotion_csv

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


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
                        # Determine cluster centers
                        if hasattr(model, 'cluster_centers_'):
                            centers = model.cluster_centers_
                        elif hasattr(model, 'means_'):
                            centers = model.means_
                        else:
                            centers = np.array([data[labels == i].mean(axis=0) for i in range(k)
                                                if np.sum(labels == i) > 0])

                        # Calculate pseudo-inertia (sum of squared distances to cluster centers)
                        pseudo_inertia = 0
                        for i in range(len(np.unique(labels))):
                            if i == -1:  # Skip noise points if any
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
    best_score = max(silhouette_scores) if silhouette_scores else None

    best_k_elbow = None
    if inertias and not np.isnan(inertias).all() and len(valid_ks) > 2:
        try:
            kn = KneeLocator(valid_ks, inertias, curve="convex", direction="decreasing")
            best_k_elbow = kn.knee

            plt.figure(figsize=(10, 6))
            plt.plot(valid_ks, inertias, 'bo-', markersize=5)
            plt.xlabel('Number of clusters (k)')
            plt.ylabel('Inertia / Pseudo-inertia')
            plt.title(f'Elbow Method for {clustering_func.__name__}')
            if best_k_elbow:
                plt.axvline(best_k_elbow, color='red', linestyle='--', label=f'Elbow at k={best_k_elbow}')
                plt.legend()
            plt.grid(True)
            plt.savefig(f'elbow_{clustering_func.__name__}.png')
            plt.close()
        except Exception as e:
            print(f"Error in elbow analysis: {e}")

    # Fix the f-string issue
    score_str = f"{best_score:.4f}" if best_score is not None else "0"
    print(f"Best silhouette k for {clustering_func.__name__}: {best_k_silhouette} (score={score_str})")
    print(f"Elbow k for {clustering_func.__name__}: {best_k_elbow}")

    # Return elbow method k as the best k for all algorithms
    return best_k_elbow, best_score, best_k_silhouette


def find_optimal_dbscan_params(data):
    """
    Function to find optimal parameters for DBSCAN
    """
    best_score = -1
    best_params = None
    eps_range = np.linspace(0.1, 2.0, 20)
    min_samples_range = range(2, 21)

    for eps in tqdm(eps_range, desc="Finding optimal DBSCAN params"):
        for min_samples in min_samples_range:
            try:
                _, labels = cm.dbscan(data, eps=eps, min_samples=min_samples)
                if len(np.unique(labels)) > 1 and -1 not in labels:
                    score = silhouette_score(data, labels)
                    if score > best_score:
                        best_score = score
                        best_params = (eps, min_samples)
            except Exception:
                continue

    return best_params


def find_cluster_k():
    # Read the CSV file
    print("=====starting clustering analysis=====")
    df = download_emotion_csv()

    # Remove the last column and null values
    data = df.iloc[:, :-1].dropna().values
    print(f"Data shape after removing last column and null values: {data.shape}")

    # Split into train and test (60% train)
    train_size = int(len(data) * 0.6)
    data_train = data[:train_size]
    print(f"Train data shape: {data_train.shape}")

    # Dictionary to store results
    results = {}

    # Process clustering methods with n_clusters
    for method_name, method_func in cm.CLUSTERING_METHODS_FUNCTIONS_DICT.items():
        if method_name != 'dbscan':  # Exclude dbscan and hdbscan from this loop
            print(f"\nProcessing {method_name}...")
            try:
                best_k_elbow, best_score, best_k_silhouette = find_optimal_k(method_func, data_train)

                # Use elbow k as the primary result, fall back to silhouette if elbow fails
                optimal_k = best_k_elbow if best_k_elbow else best_k_silhouette

                if optimal_k:
                    _, labels = method_func(data_train, n_clusters=optimal_k)
                    actual_score = silhouette_score(data_train, labels) if len(np.unique(labels)) > 1 else 0

                    results[method_name] = {
                        'elbow_k': best_k_elbow,
                        'silhouette_k': best_k_silhouette,
                        'optimal_k_used': optimal_k,
                        'silhouette_score': actual_score
                    }
            except Exception as e:
                print(f"Error processing {method_name}: {e}")

    # Process DBSCAN
    print("\nProcessing DBSCAN...")
    dbscan_params = find_optimal_dbscan_params(data_train)
    if dbscan_params:
        eps, min_samples = dbscan_params
        _, labels = cm.dbscan(data_train, eps=eps, min_samples=min_samples)
        score = silhouette_score(data_train, labels)
        results['dbscan'] = {
            'optimal_params': f"eps={eps:.2f}, min_samples={min_samples}",
            'silhouette_score': score,
            'n_clusters_found': len(np.unique(labels))
        }

    # Print summary of results
    print("\n--- Results Summary ---")
    for method, result in results.items():
        print(f"{method.upper()}:")
        for k, v in result.items():
            print(f"  {k}: {v}")

    # Plot silhouette comparison
    methods = [m for m in results if 'silhouette_score' in results[m]]
    scores = [results[m]['silhouette_score'] for m in methods]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(methods, scores)
    plt.xlabel('Clustering Method')
    plt.ylabel('Silhouette Score')
    plt.title('Clustering Methods Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()

    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{score:.4f}', ha='center', va='bottom', fontsize=8)

    plt.savefig('clustering_comparison.png')
    plt.show()

