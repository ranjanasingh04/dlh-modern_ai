#!/usr/bin/env python3
"""
Drop the customerID column from the DataFrame.
"""


def drop_customerID(df):
    """
    Remove the customerID column.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame without the customerID column.
    """
    return df.drop(columns=["customerID"])
