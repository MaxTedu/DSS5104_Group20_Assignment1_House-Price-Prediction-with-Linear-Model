"""
Data Loader Module for House Price Prediction

This module handles loading and initial preprocessing of the house price dataset.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the house price dataset from a CSV file.

    Args:
        file_path: Path to the CSV file containing the dataset

    Returns:
        Loaded dataframe with initial preprocessing
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    print(f"Data loaded successfully. Shape: {df.shape}")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform initial preprocessing on the dataset.

    Args:
        df: Raw dataframe

    Returns:
        Preprocessed dataframe
    """
    print("Starting preprocessing...")

    # Remove duplicate rows
    print(f"Duplicate rows before removal: {df.duplicated().sum()}")
    df = df.drop_duplicates()
    print(f"Duplicate rows after removal: {df.duplicated().sum()}")
    print(f"Data shape after removing duplicates: {df.shape}")

    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Extract year and month from date
    df['sale_year'] = df['date'].dt.year
    df['sale_month'] = df['date'].dt.month

    # Handle missing values (if any)
    print(f"Missing values before handling: {df.isnull().sum().sum()}")
    df = df.dropna()
    print(f"Missing values after handling: {df.isnull().sum().sum()}")

    # Remove country column (all USA)
    if 'country' in df.columns:
        df = df.drop('country', axis=1)

    # Convert numeric columns to appropriate types
    numeric_cols = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                   'floors', 'waterfront', 'view', 'condition', 'sqft_above',
                   'sqft_basement', 'yr_built', 'yr_renovated', 'sale_year', 'sale_month']

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove price outliers
    print("Rows before price outlier removal:", len(df))
    print("Price = 0:", (df['price'] == 0).sum())
    print("Price > $5M:", (df['price'] > 5000000).sum())
    
    # Remove rows with price = 0 or extreme high prices
    df = df[(df['price'] > 0) & (df['price'] <= 5000000)]
    print("Rows after price outlier removal:", len(df))

    print("Preprocessing complete!")
    return df


def split_features_target(df: pd.DataFrame, target_col: str = 'price') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features and target.

    Args:
        df: Preprocessed dataframe
        target_col: Name of target column

    Returns:
        Tuple of features dataframe and target series
    """
    # Features to exclude from training
    exclude_cols = [target_col, 'date', 'street']

    features = df.drop(exclude_cols, axis=1)
    target = df[target_col]

    print(f"Features shape: {features.shape}")
    print(f"Target shape: {target.shape}")

    return features, target


def separate_numeric_categorical(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate numeric and categorical features.

    Args:
        df: Features dataframe

    Returns:
        Tuple of numeric features and categorical features dataframes
    """
    numeric_features = df.select_dtypes(include=[np.number])
    categorical_features = df.select_dtypes(exclude=[np.number])

    print(f"Numeric features: {list(numeric_features.columns)}")
    print(f"Categorical features: {list(categorical_features.columns)}")

    return numeric_features, categorical_features


if __name__ == "__main__":
    # Test data loading
    file_path = "dataset/house_dataset.csv"
    df = load_data(file_path)
    df = preprocess_data(df)

    print("\nData Preview:")
    print(df.head())
    print("\nData Info:")
    print(df.info())
