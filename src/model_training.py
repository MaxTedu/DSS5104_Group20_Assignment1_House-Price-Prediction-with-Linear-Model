"""
Model Training Module for House Price Prediction

This module handles training and evaluation of linear regression models.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)
from xgboost import XGBRegressor
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple, Dict, Any


def calculate_mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        MAPE score
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Handle cases where y_true is 0 to avoid division by zero
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.inf

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def create_pipeline(model: Any) -> Pipeline:
    """
    Create a pipeline with standard scaler and model.

    Args:
        model: Regression model

    Returns:
        Scikit-learn pipeline
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])


def train_linear_models(X_train: pd.DataFrame, y_train: pd.Series,
                       X_test: pd.DataFrame, y_test: pd.Series,
                       log_transform_target: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Train multiple linear regression models.

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        log_transform_target: Whether target was log-transformed

    Returns:
        Dictionary with model results
    """
    results = {}

    # Define models and parameter grids
    models = {
        'ols': {
            'model': LinearRegression(),
            'params': {}
        },
        'ridge': {
            'model': Ridge(),
            'params': {
                'model__alpha': [0.1, 1.0, 10.0, 100.0]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'model__alpha': [0.1, 1.0, 10.0, 100.0]
            }
        },
        'elasticnet': {
            'model': ElasticNet(),
            'params': {
                'model__alpha': [0.1, 1.0, 10.0],
                'model__l1_ratio': [0.1, 0.5, 0.9]
            }
        }
    }

    for name, config in models.items():
        print(f"\nTraining {name.upper()} model...")

        pipeline = create_pipeline(config['model'])

        if config['params']:
            # Grid search for hyperparameter tuning
            grid_search = GridSearchCV(
                pipeline,
                config['params'],
                cv=5,
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            cv_score = -grid_search.best_score_
        else:
            pipeline.fit(X_train, y_train)
            best_model = pipeline
            best_params = {}
            cv_score = -cross_val_score(pipeline, X_train, y_train, cv=5,
                                      scoring='neg_mean_absolute_error').mean()

        # Predictions
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

        # Inverse log transform if needed
        if log_transform_target:
            y_train_inv = np.expm1(y_train)
            y_test_inv = np.expm1(y_test)
            y_pred_train_inv = np.expm1(y_pred_train)
            y_pred_test_inv = np.expm1(y_pred_test)
        else:
            y_train_inv = y_train
            y_test_inv = y_test
            y_pred_train_inv = y_pred_train
            y_pred_test_inv = y_pred_test

        # Calculate metrics
        train_mape = calculate_mape(y_train_inv, y_pred_train_inv)
        test_mape = calculate_mape(y_test_inv, y_pred_test_inv)
        test_mae = mean_absolute_error(y_test_inv, y_pred_test_inv)
        test_mse = mean_squared_error(y_test_inv, y_pred_test_inv)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test_inv, y_pred_test_inv)

        results[name] = {
            'model': best_model,
            'best_params': best_params,
            'cv_score': cv_score,
            'train_mape': train_mape,
            'test_mape': test_mape,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'y_pred_train': y_pred_train_inv,
            'y_pred_test': y_pred_test_inv
        }

        print(f"Train MAPE: {train_mape:.2f}%")
        print(f"Test MAPE: {test_mape:.2f}%")
        print(f"Test R2: {test_r2:.4f}")

    return results


def train_xgboost_baseline(X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series,
                           log_transform_target: bool = False) -> Dict[str, Any]:
    """
    Train XGBoost model as a performance baseline/upper bound.

    XGBoost typically achieves strong performance and serves as a benchmark
    to compare how well linear models perform relative to this upper bound.

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        log_transform_target: Whether target was log-transformed

    Returns:
        Dictionary with XGBoost model results
    """
    print("\n" + "=" * 60)
    print("Training XGBOOST Baseline Model...")
    print("=" * 60)

    # Create XGBoost regressor with hyperparameter tuning
    xgb = XGBRegressor(random_state=42, n_jobs=-1)

    # Parameter grid for hyperparameter tuning (simplified for speed)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'min_child_weight': [1, 3]
    }

    # Grid search with 5-fold CV
    grid_search = GridSearchCV(
        xgb,
        param_grid,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    cv_score = -grid_search.best_score_

    print(f"Best parameters: {best_params}")
    print(f"Best CV MAE: {cv_score:.0f}")

    # Predictions
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    # Inverse log transform if needed
    if log_transform_target:
        y_train_inv = np.expm1(y_train)
        y_test_inv = np.expm1(y_test)
        y_pred_train_inv = np.expm1(y_pred_train)
        y_pred_test_inv = np.expm1(y_pred_test)
    else:
        y_train_inv = y_train
        y_test_inv = y_test
        y_pred_train_inv = y_pred_train
        y_pred_test_inv = y_pred_test

    # Calculate metrics
    train_mape = calculate_mape(y_train_inv, y_pred_train_inv)
    test_mape = calculate_mape(y_test_inv, y_pred_test_inv)
    test_mae = mean_absolute_error(y_test_inv, y_pred_test_inv)
    test_mse = mean_squared_error(y_test_inv, y_pred_test_inv)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test_inv, y_pred_test_inv)

    results = {
        'model': best_model,
        'best_params': best_params,
        'cv_score': cv_score,
        'train_mape': train_mape,
        'test_mape': test_mape,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'y_pred_train': y_pred_train_inv,
        'y_pred_test': y_pred_test_inv
    }

    print(f"Train MAPE: {train_mape:.2f}%")
    print(f"Test MAPE: {test_mape:.2f}%")
    print(f"Test R2: {test_r2:.4f}")
    print(f"Test MAE: ${test_mae:,.0f}")
    print(f"Test RMSE: ${test_rmse:,.0f}")

    return results


def select_best_model(results: Dict[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """
    Select the best performing model based on test MAPE.

    Args:
        results: Dictionary of model results

    Returns:
        Tuple of best model name and its results
    """
    best_model = None
    best_mape = float('inf')
    best_name = ''

    for name, result in results.items():
        # Skip models with infinite MAPE
        if np.isfinite(result['test_mape']) and result['test_mape'] < best_mape:
            best_mape = result['test_mape']
            best_model = result
            best_name = name

    if best_model is None:
        print("\nWarning: All models have infinite MAPE. Selecting first model.")
        best_name = list(results.keys())[0]
        best_model = results[best_name]
        best_mape = best_model['test_mape']

    if np.isfinite(best_mape):
        print(f"\nBest model: {best_name.upper()} with Test MAPE: {best_mape:.2f}%")
    else:
        print(f"\nBest model: {best_name.upper()} with Test MAPE: inf%")

    return best_name, best_model


def split_train_test(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2,
                     random_state: int = 42, log_transform_target: bool = False) -> Tuple[
                         pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets.

    Args:
        X: Features dataframe
        y: Target series
        test_size: Test set size
        random_state: Random seed
        log_transform_target: Whether to log transform target

    Returns:
        Train-test split
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    if log_transform_target:
        y_train = np.log1p(y_train)
        y_test = np.log1p(y_test)

    return X_train, X_test, y_train, y_test
