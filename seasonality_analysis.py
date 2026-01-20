import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal, friedmanchisquare
from statsmodels.stats.diagnostic import acorr_ljungbox

def assess_seasonality(data, cluster_labels, output_dir):
    """
    Assess seasonality of clusters using Kruskal-Wallis, Friedman, and Ljung-Box tests.

    Parameters:
    - data (pd.DataFrame): Input data (rows = genomes, columns = time points).
    - cluster_labels (pd.Series): Cluster labels for each genome.
    - output_dir (str): Directory to save results.

    Returns:
    - None
    """
    os.makedirs(output_dir, exist_ok=True)
    unique_clusters = np.unique(cluster_labels)
    seasonality_results = []
    significant_genomes = []

    for cluster in unique_clusters:
        cluster_data = data.loc[cluster_labels == cluster]  # ✅ Fix Indexing
        
        # Ensure column headers are datetime
        try:
            timepoints = pd.to_datetime(cluster_data.columns, errors="coerce")
            valid_columns = timepoints.notna()
            cluster_data = cluster_data.loc[:, valid_columns]  # Keep only valid time columns
            timepoints = timepoints[valid_columns]  # Filter timepoints accordingly
        except Exception as e:
            print(f"Error in datetime conversion: {e}")
            continue  # Skip if datetime conversion fails

        # Compute monthly means
        monthly_means = cluster_data.groupby(timepoints.month, axis=1).mean()

        # ✅ Ensure at least two months exist before running tests
        if len(monthly_means.columns) > 1:
            try:
                kruskal_p_value = kruskal(*[monthly_means[month].dropna() for month in monthly_means.columns])[1]
                friedman_p_value = friedmanchisquare(*[monthly_means[month].dropna() for month in monthly_means.columns])[1]
            except ValueError:
                kruskal_p_value, friedman_p_value = np.nan, np.nan  # Skip if insufficient data
        else:
            kruskal_p_value, friedman_p_value = np.nan, np.nan

        # ✅ Ljung-Box Test for Autocorrelation
        ljung_p_value = np.nan
        if cluster_data.shape[1] > 1:  # At least two time points needed
            try:
                ljung_p_value = acorr_ljungbox(cluster_data.mean(axis=0).dropna(), lags=[min(10, cluster_data.shape[1] - 1)], return_df=True)["lb_pvalue"].values[0]
            except Exception as e:
                print(f"Error in Ljung-Box test: {e}")
                ljung_p_value = np.nan

        # Compute seasonality score
        total_variance = cluster_data.var().mean()
        seasonal_variance = monthly_means.var().mean()
        seasonality_score = seasonal_variance / total_variance if total_variance > 0 else 0

        # Check significance
        significant_tests = sum([kruskal_p_value < 0.05, friedman_p_value < 0.05, ljung_p_value < 0.05])
        if significant_tests >= 2:
            for genome_index in cluster_data.index:
                significant_genomes.append({
                    "Genome": genome_index,
                    "Cluster": cluster,
                    "Kruskal-Wallis": kruskal_p_value,
                    "Friedman": friedman_p_value,
                    "Ljung-Box": ljung_p_value,
                    "Seasonality Score": seasonality_score
                })

        seasonality_results.append({
            "Cluster": cluster,
            "Kruskal-Wallis": kruskal_p_value,
            "Friedman": friedman_p_value,
            "Ljung-Box": ljung_p_value,
            "Seasonality Score": seasonality_score
        })

    # Save results
    pd.DataFrame(seasonality_results).to_csv(os.path.join(output_dir, "seasonality_results.csv"), index=False)
    pd.DataFrame(significant_genomes).to_csv(os.path.join(output_dir, "significant_genomes.csv"), index=False)
