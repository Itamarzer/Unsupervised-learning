# === Import Packages ===
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score
import matplotlib.pyplot as plt
from data_import import download_emotion_csv

# Import fuzzy_kmeans instead of kmeans
from clustering_methods import fuzzy_kmeans


# Helper function to format AMI scores as strings to prevent displaying zeros for small values
def format_ami(ami_value):
    if ami_value < 1e-6:
        return f"< 1e-6"
    else:
        return f"{ami_value:.4f}"


def anomalies_detection():
    # === Load Dataset ===
    data = download_emotion_csv()

    print("Original data shape:", data.shape)

    # === Data Cleaning ===
    data_clean = data.dropna()
    print("Data shape after removing null values:", data_clean.shape)

    # Extract the labels (last column)
    labels_column = data_clean.columns[-1]
    labels = data_clean[labels_column].copy()

    # Map string labels to integers
    unique_labels = labels.unique()
    label_mapping = {label: i for i, label in enumerate(unique_labels)}
    labels_int = labels.map(label_mapping)

    # Save integer labels back
    data_clean['labels_int'] = labels_int

    # === Train/Test Split (60%/40%) ===
    split_idx = int(len(data_clean) * 0.6)
    train_data = data_clean.iloc[:split_idx]
    test_data = data_clean.iloc[split_idx:]
    print("Test data shape (last 40%):", test_data.shape)

    # Prepare test features (exclude labels)
    X_test = test_data.drop([labels_column, 'labels_int'], axis=1)
    test_labels_int = test_data['labels_int'].values

    # === Silhouette Score without removing anomalies ===
    print("\nSilhouette Score without removing any anomalies:")
    lle_full = LocallyLinearEmbedding(n_components=2, random_state=42)
    X_test_lle = lle_full.fit_transform(X_test)

    fuzzy_kmeans_model_full, fuzzy_kmeans_clusters_full = fuzzy_kmeans(
        X_test_lle,
        n_clusters=7,
        m=1.1,
        error=0.0001,
        maxiter=100
    )

    fuzzy_sil_score_full = silhouette_score(X_test_lle, fuzzy_kmeans_clusters_full)
    print(f"Silhouette Score (Full data without anomaly removal): {fuzzy_sil_score_full:.4f}")

    # === 1. Anomaly Detection - Isolation Forest ===
    print("\n1. Isolation Forest Anomaly Detection:")
    iso_forest = IsolationForest(random_state=42)
    iso_forest.fit(X_test)
    iso_forest_pred = iso_forest.predict(X_test)

    # Anomalies are -1
    iso_forest_anomalies = np.where(iso_forest_pred == -1)[0]
    print(f"Number of anomalies detected: {len(iso_forest_anomalies)} out of {len(X_test)}")
    print(f"Percentage of anomalies: {(len(iso_forest_anomalies) / len(X_test)) * 100:.2f}%")

    # Clustering after removing anomalies (Isolation Forest)
    normal_mask_iso = iso_forest_pred == 1
    anomaly_mask_iso = iso_forest_pred == -1

    if np.sum(normal_mask_iso) > 0:
        X_normal_iso = X_test.loc[normal_mask_iso].values

        # Perform LLE with 2 components
        lle_iso = LocallyLinearEmbedding(n_components=2, random_state=42)
        X_normal_lle_iso = lle_iso.fit_transform(X_normal_iso)

        # Fuzzy KMeans clustering
        fuzzy_kmeans_model_iso, fuzzy_kmeans_clusters_iso = fuzzy_kmeans(
            X_normal_lle_iso,
            n_clusters=7,
            m=1.1,
            error=0.0001,
            maxiter=100
        )

        fuzzy_sil_score_iso = silhouette_score(X_normal_lle_iso, fuzzy_kmeans_clusters_iso)
        print(f"Silhouette Score (Fuzzy KMeans after Isolation Forest): {fuzzy_sil_score_iso:.4f}")

    else:
        print("No normal data points found by Isolation Forest. Skipping clustering.")

    # === 2. Anomaly Detection - One-Class SVM ===
    print("\n2. One-Class SVM Anomaly Detection:")
    one_class_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
    one_class_svm.fit(X_test)
    one_class_svm_pred = one_class_svm.predict(X_test)

    one_class_svm_anomalies = np.where(one_class_svm_pred == -1)[0]
    print(f"Number of anomalies detected: {len(one_class_svm_anomalies)} out of {len(X_test)}")
    print(f"Percentage of anomalies: {(len(one_class_svm_anomalies) / len(X_test)) * 100:.2f}%")

    # Clustering after removing anomalies (One-Class SVM)
    normal_mask_svm = one_class_svm_pred == 1
    anomaly_mask_svm = one_class_svm_pred == -1

    if np.sum(normal_mask_svm) > 0:
        X_normal_svm = X_test.loc[normal_mask_svm].values

        # Perform LLE with 2 components
        lle_svm = LocallyLinearEmbedding(n_components=2, random_state=42)
        X_normal_lle_svm = lle_svm.fit_transform(X_normal_svm)

        # Fuzzy KMeans clustering
        fuzzy_kmeans_model_svm, fuzzy_kmeans_clusters_svm = fuzzy_kmeans(
            X_normal_lle_svm,
            n_clusters=7,
            m=1.1,
            error=0.0001,
            maxiter=100
        )

        fuzzy_sil_score_svm = silhouette_score(X_normal_lle_svm, fuzzy_kmeans_clusters_svm)
        print(f"Silhouette Score (Fuzzy KMeans after One-Class SVM): {fuzzy_sil_score_svm:.4f}")

    else:
        print("No normal data points found by One-Class SVM. Skipping clustering.")

    # === 3. Adjusted Mutual Information (AMI) for Normal vs Anomaly Classification ===
    print("\nAdjusted Mutual Information (AMI) Scores for Normal vs Anomaly Classification:")
    iso_forest_binary = np.where(iso_forest_pred == -1, 1, 0)
    ami_score_iso = adjusted_mutual_info_score(test_labels_int, iso_forest_binary)
    print(f"AMI (Isolation Forest): {format_ami(ami_score_iso)}")

    one_class_svm_binary = np.where(one_class_svm_pred == -1, 1, 0)
    ami_score_svm = adjusted_mutual_info_score(test_labels_int, one_class_svm_binary)
    print(f"AMI (One-Class SVM): {format_ami(ami_score_svm)}")

    # === 4. AMI for Normal Data Clusters vs Original Labels ===
    print("\nAMI for Normal Data Clusters vs Original Labels:")
    # If we have clusters from Isolation Forest, calculate AMI
    if np.sum(normal_mask_iso) > 0:
        normal_indices_iso = np.where(iso_forest_pred == 1)[0]
        normal_true_labels_iso = test_labels_int[normal_indices_iso]
        ami_score_fuzzy_iso = adjusted_mutual_info_score(normal_true_labels_iso, fuzzy_kmeans_clusters_iso)
        print(f"AMI (Fuzzy KMeans after Isolation Forest vs true labels): {format_ami(ami_score_fuzzy_iso)}")

    # If we have clusters from One-Class SVM, calculate AMI
    if np.sum(normal_mask_svm) > 0:
        normal_indices_svm = np.where(one_class_svm_pred == 1)[0]
        normal_true_labels_svm = test_labels_int[normal_indices_svm]
        ami_score_fuzzy_svm = adjusted_mutual_info_score(normal_true_labels_svm, fuzzy_kmeans_clusters_svm)
        print(f"AMI (Fuzzy KMeans after One-Class SVM vs true labels): {format_ami(ami_score_fuzzy_svm)}")

    # === 5. NEW: AMI for Anomaly Data vs Original Labels ===
    print("\nAMI for Anomaly Data vs Original Labels:")

    # For Isolation Forest anomalies
    if np.sum(anomaly_mask_iso) > 0:
        anomaly_indices_iso = np.where(iso_forest_pred == -1)[0]
        anomaly_true_labels_iso = test_labels_int[anomaly_indices_iso]
        # Calculate AMI between anomaly points and their original labels
        # Since anomalies are all in one group (-1), we need to calculate AMI between original labels
        # and a single-class assignment
        if len(np.unique(anomaly_true_labels_iso)) > 1:  # Only calculate if there are multiple classes in anomalies
            anomaly_assignment_iso = np.ones_like(anomaly_true_labels_iso)  # All points assigned to class 1
            ami_anomalies_iso = adjusted_mutual_info_score(anomaly_true_labels_iso, anomaly_assignment_iso)
            print(f"AMI (Isolation Forest anomalies vs true labels): {format_ami(ami_anomalies_iso)}")

            # Create a frequency table of anomalies by original label
            anomaly_label_counts = pd.Series(anomaly_true_labels_iso).value_counts().sort_index()
            print("Distribution of anomalies by original label (Isolation Forest):")
            for label_idx, count in anomaly_label_counts.items():
                original_label = [k for k, v in label_mapping.items() if v == label_idx][0]
                print(f"  Label '{original_label}' (class {label_idx}): {count} anomalies")
        else:
            print("All Isolation Forest anomalies belong to the same class, AMI calculation not applicable")
    else:
        print("No anomalies detected by Isolation Forest")

    # For One-Class SVM anomalies
    if np.sum(anomaly_mask_svm) > 0:
        anomaly_indices_svm = np.where(one_class_svm_pred == -1)[0]
        anomaly_true_labels_svm = test_labels_int[anomaly_indices_svm]
        # Calculate AMI between anomaly points and their original labels
        if len(np.unique(anomaly_true_labels_svm)) > 1:  # Only calculate if there are multiple classes in anomalies
            anomaly_assignment_svm = np.ones_like(anomaly_true_labels_svm)  # All points assigned to class 1
            ami_anomalies_svm = adjusted_mutual_info_score(anomaly_true_labels_svm, anomaly_assignment_svm)
            print(f"AMI (One-Class SVM anomalies vs true labels): {format_ami(ami_anomalies_svm)}")

            # Create a frequency table of anomalies by original label
            anomaly_label_counts = pd.Series(anomaly_true_labels_svm).value_counts().sort_index()
            print("Distribution of anomalies by original label (One-Class SVM):")
            for label_idx, count in anomaly_label_counts.items():
                original_label = [k for k, v in label_mapping.items() if v == label_idx][0]
                print(f"  Label '{original_label}' (class {label_idx}): {count} anomalies")
        else:
            print("All One-Class SVM anomalies belong to the same class, AMI calculation not applicable")
    else:
        print("No anomalies detected by One-Class SVM")

    print("\nAnalysis Complete!")

