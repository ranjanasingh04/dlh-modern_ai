#!/usr/bin/env python3
"""
Changing Column Types
"""
import matplotlib.pyplot as plt
import numpy as np


def convert_columns(df):
    """
    Convert TotalCharges to numeric and SeniorCitizen to categories.

    Args:
        df (pandas.DataFrame): DataFrame containing TotalCharges
            and SeniorCitizen columns.

    Returns:
        pandas.DataFrame: Modified DataFrame.
    """
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["SeniorCitizen"] = df["SeniorCitizen"].map({
        0: "No",
        1: "Yes"
    })

    return df
