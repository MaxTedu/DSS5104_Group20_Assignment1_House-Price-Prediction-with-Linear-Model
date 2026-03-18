"""
Feature Engineering Module for House Price Prediction

This module implements various feature engineering techniques to improve
predictive performance of linear models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    PolynomialFeatures,
    OneHotEncoder,
    StandardScaler
)
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from typing import Tuple, Dict, Any


def create_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create composite features from existing columns.

    Args:
        df: Features dataframe

    Returns:
        Dataframe with composite features added
    """
    df = df.copy()

    # House age at sale
    if 'yr_built' in df.columns and 'sale_year' in df.columns:
        df['house_age'] = df['sale_year'] - df['yr_built']

    # Renovation indicator
    if 'yr_renovated' in df.columns:
        df['is_renovated'] = (df['yr_renovated'] > 0).astype(int)

    # Total square footage
    if 'sqft_above' in df.columns and 'sqft_basement' in df.columns:
        df['sqft_total'] = df['sqft_above'] + df['sqft_basement']

    # Square footage per bedroom
    if 'sqft_living' in df.columns and 'bedrooms' in df.columns:
        df['sqft_per_bedroom'] = df['sqft_living'] / (df['bedrooms'] + 1)  # +1 to avoid division by zero

    # Bathrooms per bedroom
    if 'bathrooms' in df.columns and 'bedrooms' in df.columns:
        df['bathrooms_per_bedroom'] = df['bathrooms'] / (df['bedrooms'] + 1)

    # Lot size per bedroom
    if 'sqft_lot' in df.columns and 'bedrooms' in df.columns:
        df['sqft_lot_per_bedroom'] = df['sqft_lot'] / (df['bedrooms'] + 1)

    return df


def transform_numeric_features(df: pd.DataFrame, target: pd.Series = None) -> pd.DataFrame:
    """
    Transform numeric features to improve linear model performance.

    Args:
        df: Numeric features dataframe
        target: Target series (for log transform reference)

    Returns:
        Dataframe with transformed numeric features
    """
    df = df.copy()

    # Log transformation for skewed features
    skewed_features = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_total']

    for col in skewed_features:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col])

    return df


def generate_polynomial_features(df: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
    """
    Generate polynomial features.

    Args:
        df: Numeric features dataframe
        degree: Degree of polynomials

    Returns:
        Dataframe with polynomial features added
    """
    # Select key features for polynomial generation
    key_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'house_age']
    key_features = [f for f in key_features if f in df.columns]

    if len(key_features) == 0:
        return df

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(df[key_features])
    poly_columns = poly.get_feature_names_out(key_features)

    poly_df = pd.DataFrame(poly_features, columns=poly_columns, index=df.index)

    # Remove duplicate columns (e.g., bedrooms^2 is same as bedrooms*bedrooms)
    poly_df = poly_df.loc[:, ~poly_df.columns.duplicated()]

    return pd.concat([df, poly_df], axis=1)


def encode_categorical_features(df: pd.DataFrame, target: pd.Series = None,
                               cat_cols: list = None) -> pd.DataFrame:
    """
    Encode categorical features using various techniques.

    Args:
        df: Features dataframe
        target: Target series (for target encoding)
        cat_cols: List of categorical columns to encode

    Returns:
        Dataframe with encoded categorical features
    """
    df = df.copy()

    if cat_cols is None:
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Separate location features from other categorical features
    location_cols = ['city', 'statezip']
    other_cat_cols = [col for col in cat_cols if col not in location_cols]

    # One-hot encoding for non-location categorical features with few unique values
    onehot_cols = []
    for col in other_cat_cols:
        unique_vals = df[col].nunique()
        if unique_vals < 10:  # Only one-hot encode for very few unique values
            onehot_cols.append(col)

    if onehot_cols:
        encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        onehot_features = encoder.fit_transform(df[onehot_cols])
        onehot_df = pd.DataFrame(
            onehot_features,
            columns=encoder.get_feature_names_out(onehot_cols),
            index=df.index
        )
        df = pd.concat([df, onehot_df], axis=1)
        df = df.drop(onehot_cols, axis=1)

    # Target encoding for location-based features (with cross-validation) - MOST IMPORTANT!
    if target is not None:
        target_encode_cols = ['city', 'statezip']
        target_encode_cols = [col for col in target_encode_cols if col in df.columns]

        for col in target_encode_cols:
            df[f'{col}_target_encoded'] = target_encode(df[col], target)

    # Frequency encoding for remaining categorical features
    remaining_cat_cols = [col for col in cat_cols if col not in onehot_cols]
    for col in remaining_cat_cols:
        if col in df.columns:
            freq_encoding = df[col].value_counts(normalize=True)
            df[f'{col}_freq'] = df[col].map(freq_encoding)

    # Drop original categorical columns
    df = df.drop([col for col in cat_cols if col in df.columns], axis=1)

    return df


def target_encode(series: pd.Series, target: pd.Series, n_splits: int = 5) -> pd.Series:
    """
    Target encoding with cross-validation to prevent data leakage.

    Args:
        series: Categorical series to encode
        target: Target series
        n_splits: Number of cross-validation splits

    Returns:
        Target-encoded series
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    encoded = pd.Series(index=series.index)

    for train_idx, val_idx in kf.split(series):
        train_series, train_target = series.iloc[train_idx], target.iloc[train_idx]
        val_series = series.iloc[val_idx]

        # Calculate mean target per category in training data
        mean_target = train_target.groupby(train_series).mean()

        # Apply to validation data
        encoded.iloc[val_idx] = val_series.map(mean_target)

    # Fill missing values with overall mean
    encoded.fillna(target.mean(), inplace=True)

    return encoded


def create_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create location-based features.

    Args:
        df: Features dataframe

    Returns:
        Dataframe with location features added
    """
    df = df.copy()

    # K-means clustering for location groups (using zipcode as proxy)
    if 'statezip' in df.columns:
        # Extract zipcode as numeric
        df['zipcode'] = df['statezip'].str.extract(r'(\d{5})').astype(float)

        # Create location clusters
        if 'zipcode' in df.columns:
            kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
            df['location_cluster'] = kmeans.fit_predict(df[['zipcode']].fillna(0))

    return df


def engineer_features(df: pd.DataFrame, target: pd.Series = None) -> pd.DataFrame:
    """
    Orchestrate all feature engineering steps.

    Args:
        df: Input features dataframe
        target: Target series (for encoding)

    Returns:
        Dataframe with all engineered features
    """
    print("Starting feature engineering...")

    # Step 1: Create composite features
    df = create_composite_features(df)

    # Step 2: Transform numeric features
    df = transform_numeric_features(df, target)

    # Step 3: Generate polynomial features
    df = generate_polynomial_features(df)

    # Step 4: Encode categorical features
    df = encode_categorical_features(df, target)

    # Step 5: Create location features
    df = create_location_features(df)

    # Clean up any remaining categorical columns
    df = df.select_dtypes(exclude=['object'])

    print(f"Feature engineering complete! Total features: {df.shape[1]}")

    return df
