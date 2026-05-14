"""
Data loader for the Wisconsin Breast Cancer dataset.

Reads data/data.csv, drops artifact columns ('id' and 'Unnamed: 32'),
and returns the feature matrix X and target Series y.
"""

import pandas as pd


def load_and_clean_data(csv_path: str):
    """
    Load the Breast Cancer Wisconsin dataset and remove non-feature columns.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file (e.g. 'data/data.csv').

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with 30 numeric columns (569 rows).
    y : pd.Series
        Target column ('diagnosis') with values 'M' (Malignant) and 'B' (Benign).
    """
    df = pd.read_csv(csv_path)

    # Drop the ID column and the empty artifact column present in this dataset
    cols_to_drop = ['id', 'Unnamed: 32']
    existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(existing_cols_to_drop, axis=1)

    y = df['diagnosis']
    X = df.drop(['diagnosis'], axis=1)

    return X, y
