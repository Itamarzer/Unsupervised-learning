import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import LocallyLinearEmbedding
import umap
from sklearn.metrics import adjusted_mutual_info_score
from clustering_methods import gmm, kmeans, birch, miniBatchKmeans, fuzzy_kmeans
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import warnings
from data_import import download_emotion_csv
from scipy import stats  # Added for statistical tests

warnings.filterwarnings('ignore')

# Set random state for reproducibility
RANDOM_STATE = 42


# Define dimension reduction methods
def apply_lle(data, n_components=2):
    lle = LocallyLinearEmbedding(n_components=n_components, random_state=RANDOM_STATE)
    return lle.fit_transform(data)


def apply_umap(data, n_components=3):
    reducer = umap.UMAP(n_components=n_components, random_state=RANDOM_STATE)
    return reducer.fit_transform(data)


# Define function to evaluate clustering methods
def evaluate_clustering(X_data, y_data, dim_reduction_method, dim_components, clustering_method, cluster_params):
    # Apply dimension reduction
    if dim_reduction_method == "lle":
        X_reduced = apply_lle(X_data, n_components=dim_components)
    elif dim_reduction_method == "umap":
        X_reduced = apply_umap(X_data, n_components=dim_components)
    else:
        raise ValueError(f"Unknown dimension reduction method: {dim_reduction_method}")

    # Apply clustering
    try:
        _, cluster_labels = clustering_method(X_reduced, **cluster_params)

        # Calculate adjusted mutual information
        # Handle cases where clusters might not have the same number of points
        unique_clusters = np.unique(cluster_labels)
        if len(unique_clusters) <= 1:  # Only one cluster found
            return 0.0

        ami = adjusted_mutual_info_score(y_data, cluster_labels)
        return ami
    except Exception as e:
        print(f"Error in clustering: {str(e)}")
        return np.nan


def cluster_and_external_variables():
    # Load the CSV file
    df = download_emotion_csv()

    # Drop null values
    df = df.dropna()

    # Encode the labels from the last column
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df.iloc[:, -1])

    # Separate features and labels
    features = df.iloc[:, :-1].values
    class_labels = labels

    # Split data into train (60%) and test (40%)
    split_idx = int(0.6 * len(features))
    X_train = features[:split_idx]
    X_test = features[split_idx:]
    y_train = class_labels[:split_idx]
    y_test = class_labels[split_idx:]

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(class_labels))}")

    # List of methods to evaluate
    evaluation_methods = [
        {
            'name': 'LLE (5D) + KMeans (8)',
            'short_name': 'LLE+KMeans',
            'dim_reduction': 'lle',
            'dim_components': 2,
            'clustering': kmeans,
            'cluster_params': {'n_clusters': 8}
        },
        {
            'name': 'LLe (2D) + GMM (8)',
            'short_name': 'UMAP+GMM',
            'dim_reduction': 'lle',
            'dim_components': 2,
            'clustering': gmm,
            'cluster_params': {'n_clusters': 8}
        },
        {
            'name': 'LLE (2D) + Fuzzy KMeans (7)',
            'short_name': 'LLE+FuzzyK',
            'dim_reduction': 'lle',
            'dim_components': 2,
            'clustering': fuzzy_kmeans,
            'cluster_params': {'n_clusters': 7}
        },
        {
            'name': 'UMAP (2D) + Birch (6)',
            'short_name': 'UMAP+Birch',
            'dim_reduction': 'umap',
            'dim_components': 2,
            'clustering': birch,
            'cluster_params': {'n_clusters': 6}
        },
        {
            'name': 'LLE (5D) + MiniBatch KMeans (10)',
            'short_name': 'LLE+MiniBatch',
            'dim_reduction': 'lle',
            'dim_components': 5,
            'clustering': miniBatchKmeans,
            'cluster_params': {'n_clusters': 2}
        }
    ]

    # Split the test data into 40 groups using KFold
    print("\nSplitting test data into 40 groups...")
    n_splits = 40
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    # Initialize lists to store AMI scores for each method
    method_scores = [[] for _ in range(len(evaluation_methods))]

    # Counter for tracking progress
    group_counter = 1

    # Process each group
    for train_idx, test_idx in kf.split(X_test):
        print(f"\rProcessing group {group_counter}/{n_splits}", end="")
        group_counter += 1

        X_group = X_test[test_idx]
        y_group = y_test[test_idx]

        # Apply each method to the group
        for i, method in enumerate(evaluation_methods):
            ami = evaluate_clustering(
                X_group,
                y_group,
                method['dim_reduction'],
                method['dim_components'],
                method['clustering'],
                method['cluster_params']
            )
            method_scores[i].append(ami)

    print("\n")  # New line after progress indicator

    # Convert lists to numpy arrays for statistical analysis
    method_scores_np = [np.array(scores) for scores in method_scores]

    # Clean up NaN values if any
    for i in range(len(method_scores_np)):
        method_scores_np[i] = method_scores_np[i][~np.isnan(method_scores_np[i])]

    # Display summary statistics for each method
    print("\n=== SUMMARY STATISTICS ===")
    for i, method in enumerate(evaluation_methods):
        scores = method_scores_np[i]
        if len(scores) > 0:
            print(f"{method['name']}:")
            print(f"  Mean AMI: {np.mean(scores):.6f}")
            print(f"  Std Dev: {np.std(scores):.6f}")
            print(f"  Min: {np.min(scores):.6f}")
            print(f"  Max: {np.max(scores):.6f}")
            print(f"  Number of valid results: {len(scores)}")

    # Find the two best methods based on mean AMI
    mean_scores = [np.mean(scores) if len(scores) > 0 else -float('inf') for scores in method_scores_np]
    best_indices = np.argsort(mean_scores)[-2:][::-1]  # Get indices of two highest means
    best_methods = [evaluation_methods[i]['name'] for i in best_indices]
    best_scores = [method_scores_np[i] for i in best_indices]

    print(f"\n=== COMPARING BEST TWO METHODS ===")
    print(f"Best method: {best_methods[0]} (Mean AMI: {np.mean(best_scores[0]):.6f})")
    print(f"Second best method: {best_methods[1]} (Mean AMI: {np.mean(best_scores[1]):.6f})")

    # Perform one-way ANOVA to compare all methods
    print("\n=== ONE-WAY ANOVA TEST ===")
    # Filter out methods with too few valid results
    valid_methods = []
    valid_scores = []
    valid_names = []

    for i, scores in enumerate(method_scores_np):
        if len(scores) >= 10:  # Require at least 10 valid results for inclusion
            valid_methods.append(i)
            valid_scores.append(scores)
            valid_names.append(evaluation_methods[i]['name'])

    if len(valid_methods) >= 2:
        try:
            f_stat, p_value = stats.f_oneway(*valid_scores)
            print(f"F-statistic: {f_stat:.6f}")
            # Convert p-value to string for very small values
            p_value_str = f"{p_value:.6f}" if p_value >= 1e-6 else f"< 1e-6"
            print(f"P-value: {p_value_str}")

            if p_value < 0.05:
                print("There is a statistically significant difference among the methods.")

                # If significant, you might want to do post-hoc tests
                print("\nPost-hoc pairwise comparisons (Bonferroni-corrected):")
                for i in range(len(valid_methods)):
                    for j in range(i + 1, len(valid_methods)):
                        t_stat, p_val_ttest = stats.ttest_ind(valid_scores[i], valid_scores[j], equal_var=False)
                        # Bonferroni correction
                        corrected_p = p_val_ttest * (len(valid_methods) * (len(valid_methods) - 1) / 2)
                        corrected_p = min(corrected_p, 1.0)  # Cap at 1.0
                        # Convert corrected p-value to string for very small values
                        corrected_p_str = f"{corrected_p:.6f}" if corrected_p >= 1e-6 else f"< 1e-6"
                        print(
                            f"{valid_names[i]} vs {valid_names[j]}: p-value = {corrected_p_str} {'(significant)' if corrected_p < 0.05 else ''}")
            else:
                print("There is no statistically significant difference among the methods.")
        except Exception as e:
            print(f"Could not perform ANOVA test: {str(e)}")
    else:
        print("Not enough valid methods with sufficient data for ANOVA.")

    # Perform a paired t-test between the best two methods
    if len(best_scores[0]) > 0 and len(best_scores[1]) > 0:
        # Find common valid results
        valid_indices = []
        for i in range(min(len(method_scores[best_indices[0]]), len(method_scores[best_indices[1]]))):
            if (not np.isnan(method_scores[best_indices[0]][i]) and
                    not np.isnan(method_scores[best_indices[1]][i])):
                valid_indices.append(i)

        if len(valid_indices) > 0:
            method1_scores = np.array([method_scores[best_indices[0]][i] for i in valid_indices])
            method2_scores = np.array([method_scores[best_indices[1]][i] for i in valid_indices])

            print("\n=== PAIRED T-TEST ===")
            try:
                t_stat, p_value = stats.ttest_rel(method1_scores, method2_scores)
                print(f"T-statistic: {t_stat:.6f}")
                # Convert p-value to string for very small values
                p_value_str = f"{p_value:.6f}" if p_value >= 1e-6 else f"< 1e-6"
                print(f"P-value: {p_value_str}")

                if p_value < 0.05:
                    better_method = best_methods[0] if np.mean(method1_scores) > np.mean(method2_scores) else \
                    best_methods[1]
                    print(f"There is a statistically significant difference. {better_method} performs better.")
                else:
                    print("There is no statistically significant difference between the two methods.")
            except Exception as e:
                print(f"Could not perform paired t-test: {str(e)}")

            # Bootstrap test
            try:
                n_bootstrap = 10000
                bootstrap_diffs = []

                for _ in range(n_bootstrap):
                    # Sample with replacement
                    bootstrap_indices = np.random.choice(len(method1_scores), size=len(method1_scores), replace=True)
                    bootstrap_method1 = method1_scores[bootstrap_indices]
                    bootstrap_method2 = method2_scores[bootstrap_indices]
                    bootstrap_diffs.append(np.mean(bootstrap_method1) - np.mean(bootstrap_method2))

                # Calculate p-value as proportion of bootstrap samples where difference is opposite of observed
                observed_diff = np.mean(method1_scores) - np.mean(method2_scores)
                bootstrap_diffs = np.array(bootstrap_diffs)

                if observed_diff > 0:
                    p_value_boot = np.mean(bootstrap_diffs <= 0)
                else:
                    p_value_boot = np.mean(bootstrap_diffs >= 0)

                print("\n=== BOOTSTRAP TEST ===")
                print(f"Observed difference: {observed_diff:.6f}")
                # Convert bootstrap p-value to string for very small values
                if p_value_boot < 1e-10:
                    p_value_boot_str = "< 1e-10 (extremely significant)"
                elif p_value_boot < 1e-6:
                    p_value_boot_str = f"< 1e-6"
                else:
                    p_value_boot_str = f"{p_value_boot:.10f}"
                print(f"P-value: {p_value_boot_str}")

                if p_value_boot < 0.05:
                    better_method = best_methods[0] if observed_diff > 0 else best_methods[1]
                    print(f"There is a statistically significant difference. {better_method} performs better.")
                else:
                    print("There is no statistically significant difference between the two methods.")
            except Exception as e:
                print(f"Could not perform Bootstrap test: {str(e)}")
        else:
            print("No common valid results between the best two methods.")
    else:
        print("Not enough valid results to compare the best two methods.")

    # Calculate mean scores for each method
    method_means = []
    method_names = []
    method_short_names = []

    for i, method in enumerate(evaluation_methods):
        scores = method_scores_np[i]
        if len(scores) > 0:
            method_means.append(np.mean(scores))
            method_names.append(method['name'])
            method_short_names.append(method['short_name'])

    # Save mean scores to CSV
    means_df = pd.DataFrame({
        'Method': method_names,
        'Short_Name': method_short_names,
        'Mean_AMI': method_means
    })
    means_df.to_csv('method_mean_scores.csv', index=False)
    print("\nMean scores saved to 'method_mean_scores.csv'")

    # Create a bar plot similar to the provided image
    plt.figure(figsize=(10, 8))
    plt.bar(method_short_names, method_means, color=['blue', 'gray', 'green', 'magenta'])
    plt.xlabel('Method')
    plt.ylabel('AMI Score')
    plt.title('B', fontsize=16, fontweight='bold')
    plt.ylim(0, 1.0)  # Set y-axis from 0 to 1.0 as in the example image

    # Add grid lines for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot
    plt.tight_layout()
    plt.savefig('ami_scores_comparison.png', dpi=300)
    print("AMI scores comparison plot saved as 'ami_scores_comparison.png'")

