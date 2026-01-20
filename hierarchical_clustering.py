import numpy as np
import logging
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from tslearn.metrics import cdist_dtw

def extract_seasonal_features(data):
    """
    Extracts dominant seasonal frequencies using Fourier Transform with preprocessing.

    Args:
        data (pd.DataFrame): Time-series data (rows = features, columns = time points).

    Returns:
        np.ndarray: Fourier-transformed features.
    """
    logging.info("Extracting seasonal features using Fourier Transform.")

    # Normalize the data using StandardScaler
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    seasonal_features = []
    for row in data_scaled:
        freq = fft(row)
        seasonal_features.append(np.abs(freq[:len(freq) // 2]))  # Keep only dominant frequencies

    return np.array(seasonal_features)


from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from tslearn.metrics import cdist_dtw
from sklearn.preprocessing import MinMaxScaler

def calculate_cluster_metrics(data, labels):
    ch_score = calinski_harabasz_score(data, labels)
    db_score = davies_bouldin_score(data, labels)
    return ch_score, db_score

from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

def optimize_and_cluster(data, max_clusters, method):
    """
    Finds the optimal number of clusters for time-series data using DTW or Euclidean clustering.
    Uses multiple evaluation metrics to select the best number of clusters.

    Args:
        data (np.ndarray or pd.DataFrame): Preprocessed time-series data.
        max_clusters (int): Maximum number of clusters to test.
        method (str): Clustering method, either "dtw" or "euclidean".

    Returns:
        tuple: (clusters, best_labels, best_n_clusters)
    """
    best_score = -1
    best_labels = None
    best_n_clusters = None  # Store the best number of clusters
    clusters = None  # Store the actual cluster assignments
    best_ch_score = -1
    best_db_score = float('inf')

    # Ensure data is a NumPy array
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()  # Convert to NumPy array

    for n_clusters in range(2, max_clusters + 1):
        model = TimeSeriesKMeans(
            n_clusters=n_clusters,
            metric="dtw" if method == "dtw" else "euclidean",
            random_state=42
        )
        labels = model.fit_predict(data)

        try:
            if method == "dtw":
                # Optimize DTW computation for large datasets
                if data.shape[0] > 500:  # Limit to 500 samples for performance
                    sampled_data = data[np.random.choice(data.shape[0], 500, replace=False)]
                    distance_matrix = cdist_dtw(sampled_data)
                else:
                    distance_matrix = cdist_dtw(data)

                score = silhouette_score(distance_matrix, labels[: len(distance_matrix)], metric="precomputed")
            else:
                # Ensure reshaped_data is a NumPy array before using reshape()
                reshaped_data = np.array(data).reshape(data.shape[0], -1)
                score = silhouette_score(reshaped_data, labels, metric="euclidean")

            # Compute other metrics
            ch_score = calinski_harabasz_score(data, labels)  # Calinski-Harabasz index
            db_score = davies_bouldin_score(data, labels)  # Davies-Bouldin index

            # Update the best score and labels based on a combination of metrics
            combined_score = (score + ch_score - db_score) / 3  # Normalize the combined score

            if combined_score > best_score:
                best_score = combined_score
                best_labels = labels  # Best cluster assignments (labels)
                best_n_clusters = n_clusters  # Store the best number of clusters
                clusters = labels  # Assign the current cluster labels to clusters

            # For debugging purposes, log the scores
            logging.debug(f"Method: {method}, Clusters: {n_clusters}, Silhouette: {score:.4f}, "
                          f"CH Score: {ch_score:.4f}, DB Score: {db_score:.4f}")

        except Exception as e:
            logging.warning(f"Score computation failed for {n_clusters} clusters using {method}: {e}")

    # Ensure at least one valid clustering result exists
    if best_labels is None or best_n_clusters is None:
        logging.error(f"Failed to determine optimal clustering for method {method}. Using default n_clusters=2.")
        best_n_clusters = 2
        model = TimeSeriesKMeans(n_clusters=best_n_clusters, metric="dtw" if method == "dtw" else "euclidean", random_state=42)
        best_labels = model.fit_predict(data)
        clusters = best_labels

    logging.info(f"Optimal number of clusters: {best_n_clusters} using method: {method}")
    logging.info(f"Silhouette score: {score:.4f}, Calinski-Harabasz score: {ch_score:.4f}, "
                 f"Davies-Bouldin score: {db_score:.4f}")

    return clusters, best_labels, best_n_clusters  # Return clusters, labels, and number of clusters
