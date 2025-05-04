import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import warnings
from data_import import download_emotion_csv

warnings.filterwarnings('ignore')

# Import clustering methods
from clustering_methods import (kmeans, gmm, fuzzy_kmeans,
                                birch, miniBatchKmeans)

# Also import the dimension reduction methods
from sklearn.manifold import LocallyLinearEmbedding
import umap

# Set random state for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def statistic_test_for_clusters():
    print("Starting clustering analysis...")

    # Step 1: Read the CSV file
    print(f"Reading data from emotions")
    try:
        data = download_emotion_csv()
        print(f"Data loaded successfully. Shape: {data.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Step 2: Remove the last column and null values
    print("Preprocessing data...")
    data = data.iloc[:, :-1]  # Remove last column
    data = data.dropna()  # Remove null values
    print(f"After preprocessing. Shape: {data.shape}")

    # Step 3: Take the last 40% of the data
    total_samples = len(data)
    last_40_percent = int(total_samples * 0.4)
    test_data = data.iloc[-last_40_percent:].reset_index(drop=True)
    print(f"Last 40% of data shape: {test_data.shape}")

    # Step 4: Create 40 groups from the test data
    num_groups = 40
    group_indices = []

    # Determine group size
    group_size = len(test_data) // num_groups

    # Create groups
    for i in range(num_groups):
        if i == num_groups - 1:
            group_idx = np.arange(i * group_size, len(test_data))
        else:
            group_idx = np.arange(i * group_size, (i + 1) * group_size)
        group_indices.append(group_idx)

    # Step 5: Read the adjusted mutual information scores from CSV
    ami_file_path = 'method_mean_scores.csv'
    print(f"Reading AMI scores from {ami_file_path}")
    try:
        ami_data = pd.read_csv(ami_file_path)
        print(f"AMI data loaded successfully. Shape: {ami_data.shape}")
        ami_dict = dict(zip(ami_data['Short_Name'], ami_data['Mean_AMI']))
    except Exception as e:
        print(f"Error loading AMI data: {e}")
        ami_dict = {
            'LLE+KMeans': 0.06074830470380317,
            'UMAP+GMM': 0.07691006488443831,
            'LLE+FuzzyK': 0.05711739612374591,
            'UMAP+Birch': 0.07057303390534601,
            'LLE+MiniBatch': 0.06136106714803098
        }
        print("Using fallback AMI data from problem description")

    # Step 6: Define evaluation methods
    evaluation_methods = [
        {
            'name': 'LLE (2D) + KMeans (8)',
            'short_name': 'LLE+KMeans',
            'clustering': kmeans,
            'cluster_params': {'n_clusters': 8},
            'reduction': 'LLE',
            'reduction_params': {'n_components': 5}
        },
        {
            'name': 'UMAP (35D) + GMM (8)',
            'short_name': 'UMAP+GMM',
            'clustering': gmm,
            'cluster_params': {'n_clusters': 8},
            'reduction': 'UMAP',
            'reduction_params': {'n_components': 2}
        },
        {
            'name': 'LLE (2D) + Fuzzy KMeans (6)',
            'short_name': 'LLE+FuzzyK',
            'clustering': fuzzy_kmeans,
            'cluster_params': {'n_clusters': 7},
            'reduction': 'LLE',
            'reduction_params': {'n_components': 2}
        },
        {
            'name': 'UMAP (2D) + Birch (6)',
            'short_name': 'UMAP+Birch',
            'clustering': birch,
            'cluster_params': {'n_clusters': 6},
            'reduction': 'UMAP',
            'reduction_params': {'n_components': 2}
        },
        {
            'name': 'LLE (5D) + MiniBatch KMeans (10)',
            'short_name': 'LLE+MiniBatch',
            'clustering': miniBatchKmeans,
            'cluster_params': {'n_clusters': 10},
            'reduction': 'LLE',
            'reduction_params': {'n_components': 5}
        }
    ]

    # Step 7: Apply clustering methods to each group and calculate silhouette scores
    print("Starting clustering analysis on groups...")
    silhouette_scores_dict = {method['name']: [] for method in evaluation_methods}
    all_groups_labels = {method['name']: [] for method in evaluation_methods}

    for i, group_idx in enumerate(group_indices):
        group_data = test_data.iloc[group_idx].values

        for method in evaluation_methods:
            method_name = method['name']
            clustering_func = method['clustering']
            n_clusters = method['cluster_params']['n_clusters']
            reduction_method = method['reduction']
            reduction_params = method['reduction_params']

            try:
                # Apply dimension reduction
                if reduction_method == 'LLE':
                    reducer = LocallyLinearEmbedding(n_components=reduction_params['n_components'],
                                                     random_state=RANDOM_STATE)
                elif reduction_method == 'UMAP':
                    reducer = umap.UMAP(n_components=reduction_params['n_components'], random_state=RANDOM_STATE)
                else:
                    raise ValueError(f"Unknown reduction method: {reduction_method}")

                reduced_data = reducer.fit_transform(group_data)

                # Apply clustering
                cluster_model, labels = clustering_func(reduced_data, n_clusters=n_clusters)

                all_groups_labels[method_name].append(labels)

                # Calculate silhouette score
                if len(np.unique(labels)) > 1:
                    sil_score = silhouette_score(reduced_data, labels)
                    silhouette_scores_dict[method_name].append(sil_score)
                else:
                    print(f"Warning: {method_name} resulted in only one cluster for group {i}")
                    silhouette_scores_dict[method_name].append(float('nan'))
            except Exception as e:
                print(f"Error for {method_name} on group {i}: {e}")
                silhouette_scores_dict[method_name].append(float('nan'))
                all_groups_labels[method_name].append(None)

    weighted_scores_dict = {method['name']: [] for method in evaluation_methods}

    for method in evaluation_methods:
        method_name = method['name']
        ami_key = method['short_name']
        ami_score = ami_dict.get(ami_key, 0)

        for i in range(len(silhouette_scores_dict[method_name])):
            sil = silhouette_scores_dict[method_name][i]
            if np.isnan(sil):
                weighted_scores_dict[method_name].append(float('nan'))
            else:
                weighted_score = 0.7 * sil + 0.3 * ami_score
                weighted_scores_dict[method_name].append(weighted_score)

    silhouette_df = pd.DataFrame(silhouette_scores_dict)
    weighted_df = pd.DataFrame(weighted_scores_dict)

    print("\nSilhouette scores summary:")
    print(silhouette_df.describe())

    print("\nWeighted scores summary (85% Silhouette, 15% AMI):")
    print(weighted_df.describe())

    print("\nPerforming one-way ANOVA on silhouette scores...")
    anova_data = []
    method_names = []

    for method_name, scores in silhouette_scores_dict.items():
        valid_scores = [score for score in scores if not np.isnan(score)]
        if valid_scores:
            anova_data.append(valid_scores)
            method_names.append(method_name)

    if len(anova_data) >= 2:
        f_stat, p_value = stats.f_oneway(*anova_data)
        print(f"ANOVA results (silhouette): F-statistic = {f_stat:.4f}, p-value = {p_value:.10e}")
    else:
        print("Warning: Not enough valid data for ANOVA test.")

    methods_mean_silhouette = {method: np.nanmean(scores) for method, scores in silhouette_scores_dict.items()}
    best_methods_silhouette = sorted(methods_mean_silhouette.items(), key=lambda x: x[1], reverse=True)[:2]
    best_sil_method_1, best_sil_score_1 = best_methods_silhouette[0]
    best_sil_method_2, best_sil_score_2 = best_methods_silhouette[1]

    print(f"\nBest methods based on mean silhouette score:")
    print(f"1. {best_sil_method_1}: {best_sil_score_1:.4f}")
    print(f"2. {best_sil_method_2}: {best_sil_score_2:.4f}")

    valid_pairs = [(silhouette_scores_dict[best_sil_method_1][i], silhouette_scores_dict[best_sil_method_2][i])
                   for i in range(len(silhouette_scores_dict[best_sil_method_1]))
                   if not np.isnan(silhouette_scores_dict[best_sil_method_1][i]) and
                   not np.isnan(silhouette_scores_dict[best_sil_method_2][i])]

    if valid_pairs:
        scores_1_paired, scores_2_paired = zip(*valid_pairs)

        t_stat, t_p_value = stats.ttest_rel(scores_1_paired, scores_2_paired)
        print(f"Paired t-test results: t-statistic = {t_stat:.4f}, p-value = {t_p_value:.10e}")

        w_stat, w_p_value = stats.wilcoxon(scores_1_paired, scores_2_paired)
        print(f"Wilcoxon signed-rank test results: W-statistic = {w_stat:.4f}, p-value = {w_p_value:.10e}")

        # Adding Mann-Whitney U test
        u_stat, u_p_value = stats.mannwhitneyu(scores_1_paired, scores_2_paired)
        print(f"Mann-Whitney U test results: U-statistic = {u_stat:.4f}, p-value = {u_p_value:.10e}")
    else:
        print("Warning: Not enough valid paired data for paired tests.")

    print("\nPerforming one-way ANOVA on weighted scores...")
    anova_weighted_data = []
    method_names_weighted = []

    for method_name, scores in weighted_scores_dict.items():
        valid_scores = [score for score in scores if not np.isnan(score)]
        if valid_scores:
            anova_weighted_data.append(valid_scores)
            method_names_weighted.append(method_name)

    if len(anova_weighted_data) >= 2:
        f_stat_w, p_value_w = stats.f_oneway(*anova_weighted_data)
        print(f"ANOVA results (weighted): F-statistic = {f_stat_w:.4f}, p-value = {p_value_w:.10e}")

        methods_mean_weighted = {method: np.nanmean(scores) for method, scores in weighted_scores_dict.items()}
        best_methods_weighted = sorted(methods_mean_weighted.items(), key=lambda x: x[1], reverse=True)[:2]
        best_weighted_method_1, best_weighted_score_1 = best_methods_weighted[0]
        best_weighted_method_2, best_weighted_score_2 = best_methods_weighted[1]

        print(f"\nBest methods based on mean weighted score:")
        print(f"1. {best_weighted_method_1}: {best_weighted_score_1:.4f}")
        print(f"2. {best_weighted_method_2}: {best_weighted_score_2:.4f}")

        valid_weighted_pairs = [
            (weighted_scores_dict[best_weighted_method_1][i], weighted_scores_dict[best_weighted_method_2][i])
            for i in range(len(weighted_scores_dict[best_weighted_method_1]))
            if not np.isnan(weighted_scores_dict[best_weighted_method_1][i]) and
               not np.isnan(weighted_scores_dict[best_weighted_method_2][i])]

        if valid_weighted_pairs:
            scores_1_weighted_paired, scores_2_weighted_paired = zip(*valid_weighted_pairs)

            t_stat_w_paired, t_p_value_w_paired = stats.ttest_rel(scores_1_weighted_paired, scores_2_weighted_paired)
            print(
                f"Paired t-test (weighted) results: t-statistic = {t_stat_w_paired:.4f}, p-value = {t_p_value_w_paired:.10e}")

            w_stat_w_paired, w_p_value_w_paired = stats.wilcoxon(scores_1_weighted_paired, scores_2_weighted_paired)
            print(
                f"Wilcoxon signed-rank test (weighted) results: W-statistic = {w_stat_w_paired:.4f}, p-value = {w_p_value_w_paired:.10e}")

            # Adding Mann-Whitney U test for weighted scores
            u_stat_w_paired, u_p_value_w_paired = stats.mannwhitneyu(scores_1_weighted_paired, scores_2_weighted_paired)
            print(
                f"Mann-Whitney U test (weighted) results: U-statistic = {u_stat_w_paired:.4f}, p-value = {u_p_value_w_paired:.10e}")
        else:
            print("Warning: Not enough valid paired data for weighted paired tests.")
    else:
        print("Warning: Not enough valid data for ANOVA test on weighted scores.")

    # NEW CODE: Create a single figure with multiple subplots
    print("\nGenerating combined analysis plot...")

    # Get the data for all plots
    ami_values = [ami_dict.get(method['short_name'], 0) for method in evaluation_methods]
    method_short_names = [method['short_name'] for method in evaluation_methods]
    method_names = [method['name'] for method in evaluation_methods]
    mean_silhouette = [np.nanmean(silhouette_scores_dict[method['name']]) for method in evaluation_methods]
    mean_weighted = [np.nanmean(weighted_scores_dict[method['name']]) for method in evaluation_methods]

    # Find the best method for silhouette plot
    best_method_name = best_methods_silhouette[0][0]
    best_method_idx = [i for i, method in enumerate(evaluation_methods) if method['name'] == best_method_name][0]
    best_method = evaluation_methods[best_method_idx]

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # Subplot 1: AMI scores
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.bar(method_short_names, ami_values, color='skyblue')
    ax1.set_title('A')
    ax1.set_xlabel('Method')
    ax1.set_ylabel('AMI Score')
    ax1.set_xticklabels(method_short_names, rotation=45)

    # Subplot 2: Mean silhouette scores
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.bar(method_short_names, mean_silhouette, color='lightgreen')
    ax2.set_title('B')
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_xticklabels(method_short_names, rotation=45)

    # Subplot 3: Mean weighted scores
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.bar(method_short_names, mean_weighted, color='salmon')
    ax3.set_title('C')
    ax3.set_xlabel('Method')
    ax3.set_ylabel('Weighted Score')
    ax3.set_xticklabels(method_short_names, rotation=45)

    # Subplot 4: Silhouette plot for the best method
    ax4 = fig.add_subplot(2, 2, 4)

    # Generate silhouette plot data for the best method
    silhouette_plot_generated = False

    for group_idx in group_indices:
        group_data = test_data.iloc[group_idx].values

        # Apply dimension reduction
        if best_method['reduction'] == 'LLE':
            reducer = LocallyLinearEmbedding(n_components=best_method['reduction_params']['n_components'],
                                             random_state=RANDOM_STATE)
        elif best_method['reduction'] == 'UMAP':
            reducer = umap.UMAP(n_components=best_method['reduction_params']['n_components'], random_state=RANDOM_STATE)
        reduced_data = reducer.fit_transform(group_data)

        # Apply clustering
        cluster_model, labels = best_method['clustering'](reduced_data,
                                                          n_clusters=best_method['cluster_params']['n_clusters'])

        # If we have valid labels (more than one cluster), create the silhouette plot
        if len(np.unique(labels)) > 1:
            from sklearn.metrics import silhouette_samples

            # Compute the silhouette scores for each sample
            silhouette_values = silhouette_samples(reduced_data, labels)

            # Plot the silhouette plot
            y_lower = 10
            cluster_labels = np.unique(labels)
            n_clusters = len(cluster_labels)

            for i, cluster in enumerate(cluster_labels):
                # Aggregate the silhouette scores for samples belonging to cluster i
                ith_cluster_silhouette_values = silhouette_values[labels == cluster]
                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = plt.cm.nipy_spectral(float(i) / n_clusters)
                ax4.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax4.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            # The vertical line for average silhouette score of all the values
            ax4.axvline(x=np.mean(silhouette_values), color="red", linestyle="--")
            ax4.set_title("D")
            ax4.set_xlabel("Silhouette Coefficient")
            ax4.set_ylabel("Cluster")
            ax4.set_yticks([])  # Clear the yaxis labels / ticks
            ax4.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            silhouette_plot_generated = True
            break

    if not silhouette_plot_generated:
        ax4.text(0.5, 0.5, "No valid silhouette data available for best method",
                 horizontalalignment='center', verticalalignment='center')

    plt.tight_layout()
    plt.savefig('clustering_analysis_summary.png', dpi=300)
    print("Saved combined analysis plot as 'clustering_analysis_summary.png'")

    # Also generate individual plots for detailed examination if needed
    # Plot 1: Combined plot with all three metrics
    plt.figure(figsize=(12, 8))
    x = np.arange(len(method_names))
    width = 0.25

    plt.bar(x - width, mean_silhouette, width, label='Silhouette Score', color='lightgreen')
    plt.bar(x, ami_values, width, label='AMI Score', color='skyblue')
    plt.bar(x + width, mean_weighted, width, label='Weighted Score', color='salmon')

    plt.title('Comparison of Evaluation Metrics Across Methods')
    plt.xlabel('Method')
    plt.ylabel('Score')
    plt.xticks(x, method_short_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('metrics_comparison.png')
    print("Saved metrics comparison plot as 'metrics_comparison.png'")

    print("\nAll plots have been generated and saved!")
