import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import logging
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Using categorical units")

# Suppress deprecation warnings (if needed)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def load_data(file_path):
    logging.debug(f"Attempting to load data from {file_path}")
    try:
        data = pd.read_csv(file_path, sep='\t', index_col='Contig')
        logging.debug(f"Data successfully loaded with shape {data.shape}")
        return data
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise

def extract_months(date_columns):
    """
    Extracts months from the provided date columns.
    Converts them to month names for better readability in graphs.

    Args:
        date_columns (pd.Index): Column names from the input data.

    Returns:
        pd.Series: A series mapping each column to a corresponding month.
    """
    logging.debug(f"Original date columns: {date_columns}")

    try:
        # Clean unwanted characters from column names
        cleaned_columns = date_columns.str.replace(r"\.\d+$", "", regex=True)
        logging.debug(f"Cleaned date columns: {cleaned_columns}")

        # Attempt to parse dates
        parsed_dates = pd.to_datetime(cleaned_columns, format="%m/%d/%y", errors="coerce")
        logging.debug(f"Parsed dates: {parsed_dates}")

        # Convert parsed dates to months
        months = parsed_dates.month  # Directly access .month for DatetimeIndex
        logging.debug(f"Extracted months (numeric): {months}")

        # Check for invalid columns
        if months.isnull().any():
            invalid_columns = cleaned_columns[months.isnull()]
            logging.warning(f"Skipping invalid date columns: {invalid_columns.tolist()}")

        # Map months to their names for readability
        month_names = months.map({
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
        })
        logging.debug(f"Extracted months (names): {month_names}")

        return month_names

    except Exception as e:
        logging.error(f"Error extracting months: {e}")
        raise


def zscore_normalization(data):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(data.T).T, index=data.index, columns=data.columns)

def minmax_normalization(data, feature_range=(0, 1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    return pd.DataFrame(scaler.fit_transform(data.T).T, index=data.index, columns=data.columns)

def robust_normalization(data):
    scaler = RobustScaler()
    return pd.DataFrame(scaler.fit_transform(data.T).T, index=data.index, columns=data.columns)

def rowwise_normalization(data):
    return (data.T - data.mean(axis=1)).T / data.std(axis=1).replace(0, 1)

def energy_normalization(data):
    return data / np.sqrt((data ** 2).sum(axis=1)).replace(0, 1)

def log_normalization(data):
    return np.log1p(data)  # log(1 + x) to avoid log(0)

def normalize_data(data, method="zscore"):
    logging.debug(f"Applying {method} normalization")
    try:
        if method == "zscore":
            return zscore_normalization(data)
        elif method == "minmax":
            return minmax_normalization(data)
        elif method == "robust":
            return robust_normalization(data)
        elif method == "rowwise":
            return rowwise_normalization(data)
        elif method == "energy":
            return energy_normalization(data)
        elif method == "log":
            return log_normalization(data)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    except Exception as e:
        logging.error(f"Error during normalization: {e}")
        raise

def average_read_counts_by_month(data, months):
    """
    Averages read counts for each genome across columns belonging to the same month.

    Args:
        data (pd.DataFrame): Normalized data (rows = features, columns = samples).
        months (pd.Series): Series mapping each column to a corresponding month.

    Returns:
        pd.DataFrame: DataFrame with averaged read counts for each month.
    """
    # Verify that the number of months matches the number of columns
    if len(data.columns) != len(months):
        raise ValueError("Number of columns in data must match the length of the months series.")

    # Map columns to their corresponding months
    month_mapping = {col: month for col, month in zip(data.columns, months)}

    # Replace column names with corresponding months
    data_by_month = data.rename(columns=month_mapping)

    # Group by month and calculate the mean for each genome (row)
    averaged_data = data_by_month.groupby(axis=1, level=0).mean()

    return averaged_data
