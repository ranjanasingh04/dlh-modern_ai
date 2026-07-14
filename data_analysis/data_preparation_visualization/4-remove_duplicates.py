#!/usr/bin/env python3
"""
Removing Duplicates
"""


def remove_duplicates(df):
    """
    Remove duplicate rows from a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with duplicate rows removed.
    """
    return df.drop_duplicates()
