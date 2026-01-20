import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import argparse
import logging
from data_import_normalization import load_data, extract_months, normalize_data
from evaluate_clustering_methods import evaluate_clustering_methods
from hierarchical_clustering import optimize_and_cluster
from visualization import generate_cluster_graphs
from seasonality_analysis import assess_seasonality

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Command-line argument parsing
        parser = argparse.ArgumentParser(description="Clustering analysis for read count data.")
        parser.add_argument("input_file", help="Path to the read count matrix (TSV format)")
        parser.add_argument("output_dir", help="Directory to save the results")
        parser.add_argument("max_clusters", type=int, help="Maximum number of clusters")
        parser.add_argument("--normalization", choices=["standard", "minmax", "log", "rowise","robust","zscore","power"], 
                            default="standard", help="Normalization method (default: standard)")

        args = parser.parse_args()

        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)

        # Step 1: Data Import and Normalization
        logging.info("Loading and normalizing data.")
        data = load_data(args.input_file)
        months = extract_months(data.columns)

        # Apply the selected normalization method
        normalized_data = normalize_data(data, method=args.normalization)

        # Step 2: Evaluate Clustering Methods
        logging.info("Evaluating clustering methods.")
        best_method, method_scores = evaluate_clustering_methods(normalized_data, args.max_clusters, return_scores=True)
        logging.info(f"Clustering method scores: {method_scores}")
        logging.info(f"Selected best clustering method: {best_method}")

        # Step 3: Optimize Number of Clusters and Perform Clustering
        logging.info("Optimizing the number of clusters and performing clustering.")
        clusters, cluster_labels, optimal_clusters = optimize_and_cluster(normalized_data, args.max_clusters, best_method)

        if optimal_clusters is None or optimal_clusters < 2:
            raise ValueError(f"Invalid or failed clustering: optimal_clusters={optimal_clusters}")

        logging.info(f"Optimal number of clusters: {optimal_clusters}")

        # Step 4: Generate Visualizations
        logging.info("Generating visualizations.")
        colors = {i: f"C{i}" for i in range(optimal_clusters)}

        generate_cluster_graphs(normalized_data, clusters, cluster_labels, months, colors, args.output_dir)

        # Step 5: Seasonality Analysis
        logging.info("Analyzing seasonality...")
        seasonality_output_dir = os.path.join(args.output_dir, "Seasonality_Results")
        assess_seasonality(data, cluster_labels, seasonality_output_dir)

        logging.info("Workflow completed successfully.")

    except Exception as e:
        logging.error("An error occurred during the pipeline execution.", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
