#!/usr/bin/env python3
"""
Dropping vs Replacing vs Imputation
"""


def clean_total_charges(df, method='drop'):
    """
    Cleans missing values in the TotalCharges column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        method (str): Cleaning strategy.
            - 'drop': Remove rows with missing TotalCharges.
            - 'median': Replace missing values with the median.
            - 'impute': Replace missing values with
                        MonthlyCharges * tenure.

    Returns:
        pd.DataFrame: Modified DataFrame.
    """

    # Work on a copy to avoid modifying the original DataFrame
    df = df.copy()

    if method == "drop":
        df = df.dropna(subset=["TotalCharges"])

    elif method == "median":
        median = df["TotalCharges"].median()
        df["TotalCharges"] = df["TotalCharges"].fillna(median)

    elif method == "impute":
        df["TotalCharges"] = df["TotalCharges"].fillna(
            df["MonthlyCharges"] * df["tenure"]
        )

    return df
