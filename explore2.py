import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
import warnings
from tqdm import tqdm
from kneed import KneeLocator
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
from sklearn.manifold import LocallyLinearEmbedding
from data_import import download_emotion_csv

# Import clustering methods
from clustering_methods import kmeans, gmm, birch, miniBatchKmeans, fuzzy_kmeans

# Dictionary of clustering methods
CLUSTERING_METHODS = {
    'kmeans': kmeans,
    'gmm': gmm,
    'birch': birch,
    'miniBatchKmeans': miniBatchKmeans,
    'fuzzy_kmeans': fuzzy_kmeans
}

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def format_p_value(p_value):
    """Format p-values to avoid displaying them as zero."""
    if p_value < 1e-6:
        return "p-value < 1e-6"
    elif p_value < 0.0001:
        return f"{p_value:.2e}"
    else:
        return f"{p_value:.6f}"

def find_optimal_k_with_elbow(clustering_func, data, k_range=range(2, 21), output_dir='plots'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    method_name = clustering_func.__name__
    silhouette_scores = []
    inertias = []
    valid_ks = []

    for k in tqdm(k_range, desc=f"Finding optimal k for {method_name}"):
        try:
            model, labels = clustering_func(data, n_clusters=k)
            if len(np.unique(labels)) > 1:
                score = silhouette_score(data, labels)
                silhouette_scores.append(score)
                valid_ks.append(k)
                if hasattr(model, 'inertia_'):
                    inertias.append(model.inertia_)
                else:
                    try:
                        if hasattr(model, 'cluster_centers_'):
                            centers = model.cluster_centers_
                        elif hasattr(model, 'means_'):
                            centers = model.means_
                        else:
                            centers = np.array([data[labels == i].mean(axis=0) for i in range(k) if np.sum(labels == i) > 0])
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

    print(f"Best silhouette k for {method_name}: {best_k_silhouette} (score={best_score:.4f})")
    print(f"Elbow k for {method_name}: {best_k_elbow}")

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(valid_ks, inertias, 'o-', markersize=8, linewidth=2)
    if best_k_elbow:
        best_inertia = inertias[valid_ks.index(best_k_elbow)]
        plt.plot(best_k_elbow, best_inertia, 'D', markersize=12, markerfacecolor='red', markeredgecolor='black')
        plt.annotate(f'Elbow: k={best_k_elbow}', xy=(best_k_elbow, best_inertia), xytext=(best_k_elbow + 1, best_inertia * 1.1), arrowprops=dict(facecolor='black', shrink=0.05, width=2))
    plt.title(f'Elbow Method for {method_name}')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(valid_ks, silhouette_scores, 'o-', markersize=8, linewidth=2)
    if best_k_silhouette:
        best_silhouette = silhouette_scores[valid_ks.index(best_k_silhouette)]
        plt.plot(best_k_silhouette, best_silhouette, 'D', markersize=12, markerfacecolor='red', markeredgecolor='black')
        plt.annotate(f'Best: k={best_k_silhouette}', xy=(best_k_silhouette, best_silhouette), xytext=(best_k_silhouette + 1, best_silhouette * 0.95), arrowprops=dict(facecolor='black', shrink=0.05, width=2))
    plt.title(f'Silhouette Score for {method_name}')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{method_name}_elbow_method.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()

    return best_k_elbow, best_score, best_k_silhouette

def load_and_preprocess_data():
    try:
        df = download_emotion_csv()
        print(f"Data loaded successfully. Original shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    df_clean = df.dropna()
    print(f"Data shape after removing null values: {df_clean.shape}")
    data = df_clean.iloc[:, :-1].values
    n_samples = len(data)
    train_size = int(n_samples * 0.6)
    data_train = data[:train_size]
    print(f"Train data shape (60% of cleaned data): {data_train.shape}")

    n_samples, n_features = data_train.shape
    first_15_percent = int(n_features * 0.15)
    last_15_percent = int(n_features * 0.15)
    start_idx = first_15_percent
    end_idx = n_features - last_15_percent
    data_train_variant2 = data_train[:, start_idx:end_idx]
    print(f"Variant 2 data shape (first and last 15% features removed): {data_train_variant2.shape}")

    return data_train, data_train_variant2

def find_optimal_dimensions(data, best_method_func, best_k, dim_range=range(2, 51), output_dir='plots'):
    silhouette_scores = []
    valid_dims = []

    for n_components in tqdm(dim_range, desc="Finding optimal LLE dimensions"):
        try:
            lle = LocallyLinearEmbedding(n_components=n_components, random_state=42, eigen_solver='dense')
            data_lle = lle.fit_transform(data)
            model, labels = best_method_func(data_lle, n_clusters=best_k)
            if len(np.unique(labels)) > 1:
                score = silhouette_score(data_lle, labels)
                silhouette_scores.append(score)
                valid_dims.append(n_components)
        except Exception as e:
            print(f"Error with dimensions={n_components}: {e}")
            continue

    best_dim = valid_dims[np.argmax(silhouette_scores)] if silhouette_scores else None
    best_score = max(silhouette_scores) if silhouette_scores else None

    if best_score is not None:
        print(f"Best LLE dimensions: {best_dim} (score={best_score:.4f})")
    else:
        print(f"Best LLE dimensions: {best_dim} (score=0)")

    if valid_dims and silhouette_scores:
        plt.figure(figsize=(10, 6))
        plt.plot(valid_dims, silhouette_scores, 'o-', markersize=8, linewidth=2)
        if best_dim:
            plt.plot(best_dim, best_score, 'D', markersize=12, markerfacecolor='red', markeredgecolor='black')
            plt.annotate(f'Best: dim={best_dim}', xy=(best_dim, best_score), xytext=(best_dim + 2, best_score * 0.95), arrowprops=dict(facecolor='black', shrink=0.05, width=2))
        plt.title(f'Silhouette Score vs LLE Dimensions')
        plt.xlabel('Number of dimensions')
        plt.ylabel('Silhouette Score')
        plt.grid(True)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, f'lle_dimensions_optimization.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Dimensions plot saved to {output_path}")
        plt.close()

    return best_dim, best_score

def statistical_comparison(data_train, data_train_variant2, best_method_func, best_k, best_dim, output_dir='plots'):
    print("\n--- Running statistical tests comparing baseline and optimized approaches ---")
    n_runs = 30
    baseline_scores = []
    optimized_scores = []
    fuzzy_method = CLUSTERING_METHODS['fuzzy_kmeans']
    best_method_name = best_method_func.__name__

    print(f"Comparing baseline (LLE with 2 dimensions + fuzzy_kmeans with 7 clusters)")
    print(f"vs optimized (LLE with {best_dim} dimensions + {best_method_name} with {best_k} clusters)")

    for i in tqdm(range(n_runs), desc="Statistical test iterations"):
        random_indices = np.random.choice(len(data_train), size=int(0.8 * len(data_train)), replace=False)
        data_sample_original = data_train[random_indices]
        lle_2d = LocallyLinearEmbedding(n_components=2, random_state=42 + i, eigen_solver='dense')
        data_lle_2d = lle_2d.fit_transform(data_sample_original)
        _, labels_fuzzy = fuzzy_method(data_lle_2d, n_clusters=7)
        if len(np.unique(labels_fuzzy)) > 1:
            baseline_scores.append(silhouette_score(data_lle_2d, labels_fuzzy))
        else:
            baseline_scores.append(0)

        data_sample_variant2 = data_train_variant2[random_indices]
        lle_best = LocallyLinearEmbedding(n_components=best_dim, random_state=42 + i, eigen_solver='dense')
        data_lle_best = lle_best.fit_transform(data_sample_variant2)
        _, labels_best = best_method_func(data_lle_best, n_clusters=best_k)
        if len(np.unique(labels_best)) > 1:
            optimized_scores.append(silhouette_score(data_lle_best, labels_best))
        else:
            optimized_scores.append(0)

    baseline_mean = np.mean(baseline_scores)
    optimized_mean = np.mean(optimized_scores)
    improvement = optimized_mean - baseline_mean
    improvement_percent = (improvement / baseline_mean) * 100 if baseline_mean > 0 else 0

    print(f"\nMean baseline score: {baseline_mean:.4f}")
    print(f"Mean optimized score: {optimized_mean:.4f}")
    print(f"Improvement: {improvement:.4f} ({improvement_percent:.2f}%)")

    t_stat, p_value_t = stats.ttest_rel(optimized_scores, baseline_scores)
    print(f"Paired t-test: t={t_stat:.4f}, p={format_p_value(p_value_t)}")
    w_stat, p_value_w = stats.wilcoxon(optimized_scores, baseline_scores)
    print(f"Wilcoxon signed-rank test: W={w_stat:.4f}, p={format_p_value(p_value_w)}")
    u_stat, p_value_u = stats.mannwhitneyu(optimized_scores, baseline_scores)
    print(f"Mann-Whitney U test: U={u_stat:.4f}, p={format_p_value(p_value_u)}")

    plt.figure(figsize=(10, 6))
    box_data = [baseline_scores, optimized_scores]
    labels = ['Baseline\nLLE(2)+Fuzzy(7)', f'Optimized\nLLE({best_dim})+{best_method_name}({best_k})']
    box = plt.boxplot(box_data, patch_artist=True, labels=labels)
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.title('Comparison of Clustering Approaches')
    plt.ylabel('Silhouette Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    alpha = 0.05
    significance = "Significant" if p_value_t < alpha and p_value_w < alpha and p_value_u < alpha else "Not Significant"
    plt.annotate(f'p-value (t-test): {format_p_value(p_value_t)}\nResult: {significance}',
                 xy=(0.5, 0.02), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'statistical_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {output_path}")
    plt.close()

    print("\n--- Conclusions ---")
    if p_value_t < alpha and p_value_w < alpha and p_value_u < alpha:
        if np.mean(optimized_scores) > np.mean(baseline_scores):
            print(f"The optimized approach (LLE with {best_dim} dimensions + {best_method_name} with {best_k} clusters) "
                  f"significantly outperforms the baseline approach (LLE with 2 dimensions + fuzzy_kmeans with 7 clusters).")
        else:
            print(f"The baseline approach (LLE with 2 dimensions + fuzzy_kmeans with 7 clusters) "
                  f"significantly outperforms the optimized approach (LLE with {best_dim} dimensions + {best_method_name} with {best_k} clusters).")
    else:
        print("No significant difference was found between the optimized and baseline approaches.")

    return {
        'baseline_mean': baseline_mean,
        'optimized_mean': optimized_mean,
        'improvement': improvement,
        'improvement_percent': improvement_percent,
        'p_value_t': p_value_t,
        'p_value_w': p_value_w,
        'p_value_u': p_value_u
    }

def explore2():
    output_dir = 'cluster_analysis'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_train, data_train_variant2 = load_and_preprocess_data()
    if data_train is None:
        print("Error: Could not load or process the data.")
        return

    results_variant2 = {}
    print("\n--- Processing Variant 2 (First and Last 15% Features Removed) ---")
    for method_name, method_func in CLUSTERING_METHODS.items():
        print(f"\nProcessing {method_name}...")
        try:
            best_k_elbow, best_score, best_k_silhouette = find_optimal_k_with_elbow(
                method_func, data_train_variant2, output_dir=output_dir)
            optimal_k = best_k_elbow if best_k_elbow else best_k_silhouette
            if optimal_k:
                model, labels = method_func(data_train_variant2, n_clusters=optimal_k)
                actual_score = silhouette_score(data_train_variant2, labels) if len(np.unique(labels)) > 1 else 0
                results_variant2[method_name] = {
                    'optimal_k': optimal_k,
                    'silhouette_score': actual_score,
                    'labels': labels
                }
                print(f"Final results for {method_name}:")
                print(f"  Optimal k: {optimal_k}")
                print(f"  Silhouette score: {actual_score:.4f}")
            else:
                print(f"No optimal k found for {method_name}")
        except Exception as e:
            print(f"Error processing {method_name}: {e}")

    if results_variant2:
        best_method = max(results_variant2.items(), key=lambda x: x[1]['silhouette_score'])
        best_method_name = best_method[0]
        best_k = best_method[1]['optimal_k']
        best_silhouette = best_method[1]['silhouette_score']
        best_method_func = CLUSTERING_METHODS[best_method_name]

        print(f"\n--- Best Method: {best_method_name} ---")
        print(f"Optimal k: {best_k}")
        print(f"Silhouette score: {best_silhouette:.4f}")

        best_dim, best_dim_score = find_optimal_dimensions(
            data_train_variant2, best_method_func, best_k, output_dir=output_dir)

        print(f"\n--- Optimal LLE Dimensions for {best_method_name} with k={best_k} ---")
        print(f"Best dimension: {best_dim}")
        print(f"Silhouette score: {best_dim_score:.4f}")

        stats_results = statistical_comparison(
            data_train, data_train_variant2, best_method_func, best_k, best_dim, output_dir=output_dir)

        print("\n" + "=" * 50)
        print("FINAL SUMMARY")
        print("=" * 50)
        print(f"Best clustering method: {best_method_name}")
        print(f"Optimal number of clusters (k): {best_k}")
        print(f"Optimal LLE dimensions: {best_dim}")
        print(f"Baseline approach: LLE(2) + fuzzy_kmeans(7) - Mean score: {stats_results['baseline_mean']:.4f}")
        print(
            f"Optimized approach: LLE({best_dim}) + {best_method_name}({best_k}) - Mean score: {stats_results['optimized_mean']:.4f}")
        print(f"Improvement: {stats_results['improvement']:.4f} ({stats_results['improvement_percent']:.2f}%)")
        alpha = 0.05
        is_significant = stats_results['p_value_t'] < alpha and stats_results['p_value_w'] < alpha and stats_results[
            'p_value_u'] < alpha
        print(f"Statistical significance: {'Yes' if is_significant else 'No'}")
        print("\nAll clustering methods ranked by performance:")
        sorted_methods = sorted(results_variant2.items(), key=lambda x: x[1]['silhouette_score'], reverse=True)
        for i, (method_name, result) in enumerate(sorted_methods, 1):
            print(f"{i}. {method_name}: k={result['optimal_k']}, score={result['silhouette_score']:.4f}")
    else:
        print("No valid clustering results found.")