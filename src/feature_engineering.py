"""
Feature Engineering Module for House Price Prediction

This module implements various feature engineering techniques to improve
predictive performance of linear models.

IMPORTANT: Target encoding must be computed on training set only to avoid data leakage.
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
from typing import Tuple, Dict, Any, Optional


def create_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create composite features from existing columns.

    Args:
        df: Features dataframe

    Returns:
        Dataframe with composite features added
    """
    df = df.copy()

    if 'yr_built' in df.columns and 'sale_year' in df.columns:
        df['house_age'] = df['sale_year'] - df['yr_built']

    if 'yr_renovated' in df.columns:
        df['is_renovated'] = (df['yr_renovated'] > 0).astype(int)

    if 'sqft_above' in df.columns and 'sqft_basement' in df.columns:
        df['sqft_total'] = df['sqft_above'] + df['sqft_basement']

    if 'sqft_living' in df.columns and 'bedrooms' in df.columns:
        df['sqft_per_bedroom'] = df['sqft_living'] / (df['bedrooms'] + 1)

    if 'bathrooms' in df.columns and 'bedrooms' in df.columns:
        df['bathrooms_per_bedroom'] = df['bathrooms'] / (df['bedrooms'] + 1)

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
    key_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'house_age']
    key_features = [f for f in key_features if f in df.columns]

    if len(key_features) == 0:
        return df

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(df[key_features])
    poly_columns = poly.get_feature_names_out(key_features)

    poly_df = pd.DataFrame(poly_features, columns=poly_columns, index=df.index)

    poly_df = poly_df.loc[:, ~poly_df.columns.duplicated()]

    return pd.concat([df, poly_df], axis=1)


def compute_target_encoding_map(series: pd.Series, target: pd.Series) -> Dict[str, float]:
    """
    Compute target encoding mapping from training data only.
    
    This is the FIT step - should only be called on training data.

    Args:
        series: Categorical series to encode (from training set)
        target: Target series (from training set)

    Returns:
        Dictionary mapping category values to mean target
    """
    mean_target = target.groupby(series).mean()
    global_mean = target.mean()
    
    return {
        'mapping': mean_target.to_dict(),
        'global_mean': global_mean
    }


def apply_target_encoding(series: pd.Series, encoding_map: Dict[str, float]) -> pd.Series:
    """
    Apply pre-computed target encoding to a series.
    
    This is the TRANSFORM step - can be applied to train or test data.

    Args:
        series: Categorical series to encode
        encoding_map: Dictionary with 'mapping' and 'global_mean' keys

    Returns:
        Target-encoded series
    """
    mapping = encoding_map['mapping']
    global_mean = encoding_map['global_mean']
    
    encoded = series.map(mapping)
    encoded.fillna(global_mean, inplace=True)
    
    return encoded


class FeatureEngineeringPipeline:
    """
    A feature engineering pipeline that properly handles train/test split.
    
    IMPORTANT: This class follows the fit/transform pattern to prevent data leakage.
    - fit() computes all statistics (target encodings, etc.) from training data only
    - transform() applies these statistics to any data (train or test)
    """
    
    def __init__(self):
        self.target_encoding_maps = {}
        self.onehot_encoder = None
        self.onehot_cols = []
        self.kmeans = None
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureEngineeringPipeline':
        """
        Fit the feature engineering pipeline on training data.
        
        This computes all statistics (target encodings, etc.) from training data only.
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            self
        """
        X = X.copy()
        
        cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        location_cols = ['city', 'statezip']
        for col in location_cols:
            if col in X.columns:
                self.target_encoding_maps[col] = compute_target_encoding_map(X[col], y)
        
        other_cat_cols = [col for col in cat_cols if col not in location_cols]
        
        self.onehot_cols = []
        for col in other_cat_cols:
            unique_vals = X[col].nunique()
            if unique_vals < 10:
                self.onehot_cols.append(col)
        
        if self.onehot_cols:
            self.onehot_encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
            self.onehot_encoder.fit(X[self.onehot_cols])
        
        if 'statezip' in X.columns:
            zipcodes = X['statezip'].str.extract(r'(\d{5})').astype(float)
            self.kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
            self.kmeans.fit(zipcodes.fillna(0).values.reshape(-1, 1))
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering transformations.
        
        Args:
            X: Features to transform
            
        Returns:
            Transformed features dataframe
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform!")
        
        X = X.copy()
        
        X = create_composite_features(X)
        
        X = transform_numeric_features(X)
        
        X = generate_polynomial_features(X)
        
        for col, encoding_map in self.target_encoding_maps.items():
            if col in X.columns:
                X[f'{col}_target_encoded'] = apply_target_encoding(X[col], encoding_map)
        
        if self.onehot_encoder is not None and len(self.onehot_cols) > 0:
            onehot_features = self.onehot_encoder.transform(X[self.onehot_cols])
            onehot_df = pd.DataFrame(
                onehot_features,
                columns=self.onehot_encoder.get_feature_names_out(self.onehot_cols),
                index=X.index
            )
            X = pd.concat([X, onehot_df], axis=1)
            X = X.drop(self.onehot_cols, axis=1)
        
        cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        remaining_cat_cols = [col for col in cat_cols if col not in self.onehot_cols]
        
        for col in remaining_cat_cols:
            if col in X.columns:
                freq_encoding = X[col].value_counts(normalize=True)
                X[f'{col}_freq'] = X[col].map(freq_encoding)
        
        if self.kmeans is not None and 'statezip' in X.columns:
            zipcodes = X['statezip'].str.extract(r'(\d{5})').astype(float)
            X['zipcode'] = zipcodes
            X['location_cluster'] = self.kmeans.predict(zipcodes.fillna(0).values.reshape(-1, 1))
        
        X = X.select_dtypes(exclude=['object'])
        
        X = X.loc[:, ~X.columns.duplicated()]
        
        return X
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit and transform in one step (for training data).
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            Transformed features
        """
        self.fit(X, y)
        return self.transform(X)


def create_spline_features(df: pd.DataFrame, n_knots: int = 5) -> pd.DataFrame:
    """
    Create spline basis expansion for continuous features.
    
    This is a structurally different approach from polynomial features.
    
    Args:
        df: Features dataframe
        n_knots: Number of knots for spline
        
    Returns:
        Dataframe with spline features added
    """
    from scipy.interpolate import BSpline
    from scipy import interpolate
    
    df = df.copy()
    
    spline_cols = ['sqft_living', 'house_age', 'sqft_lot']
    spline_cols = [col for col in spline_cols if col in df.columns]
    
    for col in spline_cols:
        x = df[col].values
        
        x_min, x_max = x.min(), x.max()
        knots = np.linspace(x_min, x_max, n_knots + 2)[1:-1]
        
        for i, knot in enumerate(knots):
            df[f'{col}_spline_{i}'] = np.maximum(0, x - knot) ** 3
    
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-specific interaction features.
    
    Args:
        df: Features dataframe
        
    Returns:
        Dataframe with interaction features
    """
    df = df.copy()
    
    df = df.loc[:, ~df.columns.duplicated()]
    
    if 'sqft_living' in df.columns and 'waterfront' in df.columns:
        df['sqft_living_x_waterfront'] = df['sqft_living'].values * df['waterfront'].values
    
    if 'sqft_living' in df.columns and 'view' in df.columns:
        df['sqft_living_x_view'] = df['sqft_living'].values * df['view'].values
    
    if 'house_age' in df.columns and 'condition' in df.columns:
        df['age_x_condition'] = df['house_age'].values * df['condition'].values
    
    if 'bedrooms' in df.columns and 'bathrooms' in df.columns:
        df['bedrooms_x_bathrooms'] = df['bedrooms'].values * df['bathrooms'].values
    
    if 'sqft_living' in df.columns and 'floors' in df.columns:
        df['sqft_per_floor'] = df['sqft_living'].values / (df['floors'].values + 1)
    
    return df


def create_price_per_sqft_by_city(X_train: pd.DataFrame, y_train: pd.Series,
                                   X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create price per sqft feature aggregated by city.
    
    IMPORTANT: Computed on training set only, then mapped to test set.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        
    Returns:
        Tuple of (X_train with new feature, X_test with new feature)
    """
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    if 'city' in X_train.columns and 'sqft_living' in X_train.columns:
        price_per_sqft = y_train / X_train['sqft_living']
        city_price_per_sqft = price_per_sqft.groupby(X_train['city']).mean()
        global_mean = price_per_sqft.mean()
        
        X_train['city_price_per_sqft'] = X_train['city'].map(city_price_per_sqft).fillna(global_mean)
        X_test['city_price_per_sqft'] = X_test['city'].map(city_price_per_sqft).fillna(global_mean)
    
    return X_train, X_test


def create_knn_features(X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, n_neighbors: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create KNN-based features using only training data.
    
    IMPORTANT: KNN is fit on training data only, then used to predict for test data.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        n_neighbors: Number of neighbors
        
    Returns:
        Tuple of (X_train with KNN feature, X_test with KNN feature)
    """
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    numeric_cols = ['sqft_living', 'sqft_lot', 'bedrooms', 'bathrooms', 'house_age', 'lat', 'long']
    numeric_cols = [col for col in numeric_cols if col in X_train.columns]
    
    if len(numeric_cols) >= 2:
        train_numeric = X_train[numeric_cols].fillna(X_train[numeric_cols].median())
        test_numeric = X_test[numeric_cols].fillna(X_train[numeric_cols].median())
        
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_numeric)
        test_scaled = scaler.transform(test_numeric)
        
        knn = KNeighborsRegressor(n_neighbors=n_neighbors)
        knn.fit(train_scaled, y_train)
        
        X_train['knn_price_pred'] = knn.predict(train_scaled)
        X_test['knn_price_pred'] = knn.predict(test_scaled)
    
    return X_train, X_test


def engineer_features(df: pd.DataFrame, target: pd.Series = None) -> pd.DataFrame:
    """
    Orchestrate all feature engineering steps.
    
    WARNING: This function is kept for backward compatibility but has DATA LEAKAGE!
    Use FeatureEngineeringPipeline class instead for proper train/test handling.

    Args:
        df: Input features dataframe
        target: Target series (for encoding)

    Returns:
        Dataframe with all engineered features
    """
    print("Starting feature engineering...")

    df = create_composite_features(df)

    df = transform_numeric_features(df, target)

    df = generate_polynomial_features(df)

    df = encode_categorical_features(df, target)

    df = create_location_features(df)

    df = df.select_dtypes(exclude=['object'])

    print(f"Feature engineering complete! Total features: {df.shape[1]}")

    return df


def encode_categorical_features(df: pd.DataFrame, target: pd.Series = None,
                               cat_cols: list = None) -> pd.DataFrame:
    """
    Encode categorical features using various techniques.
    
    WARNING: This function has DATA LEAKAGE when target encoding is computed
    on the full dataset. Use FeatureEngineeringPipeline instead.

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

    location_cols = ['city', 'statezip']
    other_cat_cols = [col for col in cat_cols if col not in location_cols]

    onehot_cols = []
    for col in other_cat_cols:
        unique_vals = df[col].nunique()
        if unique_vals < 10:
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

    if target is not None:
        target_encode_cols = ['city', 'statezip']
        target_encode_cols = [col for col in target_encode_cols if col in df.columns]

        for col in target_encode_cols:
            df[f'{col}_target_encoded'] = target_encode(df[col], target)

    remaining_cat_cols = [col for col in cat_cols if col not in onehot_cols]
    for col in remaining_cat_cols:
        if col in df.columns:
            freq_encoding = df[col].value_counts(normalize=True)
            df[f'{col}_freq'] = df[col].map(freq_encoding)

    df = df.drop([col for col in cat_cols if col in df.columns], axis=1)

    return df


def target_encode(series: pd.Series, target: pd.Series, n_splits: int = 5) -> pd.Series:
    """
    Target encoding with cross-validation to prevent data leakage.
    
    WARNING: This still has leakage when used on full dataset before train/test split.
    Use compute_target_encoding_map and apply_target_encoding instead.

    Args:
        series: Categorical series to encode
        target: Target series
        n_splits: Number of cross-validation splits

    Returns:
        Target-encoded series
    """
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    encoded = pd.Series(index=series.index)

    for train_idx, val_idx in kf.split(series):
        train_series, train_target = series.iloc[train_idx], target.iloc[train_idx]
        val_series = series.iloc[val_idx]

        mean_target = train_target.groupby(train_series).mean()

        encoded.iloc[val_idx] = val_series.map(mean_target)

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

    if 'statezip' in df.columns:
        df['zipcode'] = df['statezip'].str.extract(r'(\d{5})').astype(float)

        if 'zipcode' in df.columns:
            kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
            df['location_cluster'] = kmeans.fit_predict(df[['zipcode']].fillna(0))

    return df
