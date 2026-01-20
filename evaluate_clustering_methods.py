import logging
from sklearn.metrics import silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_dtw
from hierarchical_clustering import extract_seasonal_features
import warnings
from tqdm import tqdm

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Using categorical units")
warnings.filterwarnings("ignore", category=DeprecationWarning)

def evaluate_clustering_methods(data, max_clusters=5, return_scores=False):
    logging.info("Evaluating clustering methods: DTW and Fourier.")
    methods = ["dtw", "fourier"]
    scores = {}

    for method in tqdm(methods, desc="Testing clustering methods"):
        if method == "fourier":
            # Fourier transformation
            transformed_data = extract_seasonal_features(data)
            metric = "euclidean"
        else:
            # Scale time-series data for DTW
            scaler = TimeSeriesScalerMeanVariance()
            transformed_data = scaler.fit_transform(data.values.reshape(data.shape[0], -1, 1))
            metric = "dtw"

        try:
            best_score = -1
            for n_clusters in range(2, max_clusters + 1):
                model = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric, random_state=42)
                labels = model.fit_predict(transformed_data)

                if method == "dtw":
                    # Compute DTW distance matrix
                    distance_matrix = cdist_dtw(transformed_data)
                    score = silhouette_score(distance_matrix, labels, metric="precomputed")
                else:
                    # Flatten data for Fourier method
                    reshaped_data = transformed_data.reshape(transformed_data.shape[0], -1)
                    score = silhouette_score(reshaped_data, labels, metric="euclidean")

                logging.debug(f"Method: {method}, Clusters: {n_clusters}, Silhouette Score: {score:.4f}")

                if score > best_score:
                    best_score = score

            scores[method] = best_score
        except Exception as e:
            logging.warning(f"Method {method} failed: {e}", exc_info=True)

    if not scores:
        raise ValueError("All clustering methods failed.")

    best_method = max(scores, key=scores.get)
    logging.info(f"Best clustering method: {best_method} (Score: {scores[best_method]:.4f})")

    # Now it conditionally returns scores if requested
    return (best_method, scores) if return_scores else best_method

