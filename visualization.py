import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import os
import numpy as np
import pandas as pd
import logging
import warnings
from tqdm import tqdm
from collections import defaultdict

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Using categorical units")
warnings.filterwarnings("ignore", category=DeprecationWarning)

def generate_cluster_graphs(data, clusters, cluster_labels, months, colors, output_dir):
    """
    Generate line graphs for each cluster showing normalized read counts over months.

    Args:
        data (pd.DataFrame): Normalized data (rows = features, columns = samples).
        clusters (array or dict): Cluster assignments (as a NumPy array) or a dictionary mapping cluster labels to row indices.
        cluster_labels (list): List of cluster assignments for each row in the data.
        months (pd.Series): Extracted month numbers for column names.
        colors (dict): A dictionary mapping cluster labels to color codes.
        output_dir (str): Directory to save the generated graphs.
    """

    # Ensure clusters is a dictionary (Fix for AttributeError)
    if isinstance(clusters, np.ndarray):
        cluster_dict = defaultdict(list)
        for idx, label in enumerate(clusters):
            cluster_dict[label].append(idx)
        clusters = dict(cluster_dict)  # Convert defaultdict to a normal dictionary

    # Map months to integers and aggregate data by month
    month_nums = pd.Series(range(1, 13), index=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    month_indices = months.map(month_nums)

    # Log diagnostic information
    logging.debug(f"Original columns in data: {data.columns}")
    logging.debug(f"Extracted months: {months}")
    logging.debug(f"Month indices mapping: {month_indices}")

    if month_indices.isnull().any():
        raise ValueError("Invalid month mapping. Ensure all months are properly extracted.")

    # Ensure month_indices align with data columns
    if len(data.columns) != len(month_indices):
        raise ValueError(f"Mismatch between data columns ({len(data.columns)}) and month indices ({len(month_indices)}).")

    # Aggregate data by month
    try:
        aggregated_data = data.groupby(month_indices, axis=1).mean()
    except KeyError as e:
        logging.error(f"Aggregation failed due to missing keys: {e}")
        raise ValueError("Aggregation failed. Check if data columns match the month mapping.")

    if aggregated_data.empty:
        raise ValueError("Aggregated data is empty. Check the data and month mapping.")

    for cluster_id, indices in tqdm(clusters.items(), desc="Generating cluster graphs"):
        cluster_data = aggregated_data.iloc[indices]

        if cluster_data.empty:
            logging.warning(f"Cluster {cluster_id} has no data after aggregation. Skipping plot.")
            continue

        medoid_index = compute_medoid(cluster_data)
        medoid = cluster_data.iloc[medoid_index]

        plt.figure(figsize=(10, 6))

        for row in cluster_data.values:
            plt.plot(range(1, 13), row, color=colors.get(cluster_id, 'gray'), alpha=0.1, linewidth=0.5)

        plt.plot(range(1, 13), medoid, color='black', linewidth=2, linestyle='--', label="Medoid")
        plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.title(f"Cluster {cluster_id + 1}", fontsize=14, color='red')
        plt.xlabel("Month", fontsize=12)
        plt.ylabel("Normalized Read Counts", fontsize=12)
        plt.grid(True)
        plt.legend(loc="best")
        plt.tight_layout()

        graph_path = os.path.join(output_dir, f"cluster_{cluster_id + 1}.png")
        plt.savefig(graph_path)
        plt.close()

def compute_medoid(data):
    """
    Compute the medoid of a cluster using pairwise distances.

    Args:
        data (pd.DataFrame): Data points in the cluster (rows = features, columns = samples).

    Returns:
        int: Index of the medoid in the cluster data.
    """
    if len(data) == 1:
        return 0  # If only one element, return its index

    distances = cdist(data, data, metric='euclidean')
    medoid_index = np.argmin(distances.sum(axis=1))  # Find the row with the smallest total distance
    return medoid_index
