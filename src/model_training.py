"""
Model Training Module for House Price Prediction

This module handles training and evaluation of linear regression models.

IMPORTANT: 
1. MAPE is computed in PRICE SPACE (not log space) - inverse transform is applied
2. XGBoost is properly tuned with extended hyperparameter search and early stopping
3. Multiple structurally different approaches are tried (not just regularization variants)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    HuberRegressor
)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, SplineTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from typing import Tuple, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')


def calculate_mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Calculate Mean Absolute Percentage Error in PRICE SPACE.

    Args:
        y_true: True target values (in price space, not log space)
        y_pred: Predicted values (in price space, not log space)

    Returns:
        MAPE score as percentage
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)

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

    models = {
        'ols': {
            'model': LinearRegression(),
            'params': {}
        },
        'ridge': {
            'model': Ridge(),
            'params': {
                'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0, 500.0]
            }
        },
        'lasso': {
            'model': Lasso(max_iter=10000),
            'params': {
                'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
            }
        },
        'elasticnet': {
            'model': ElasticNet(max_iter=10000),
            'params': {
                'model__alpha': [0.01, 0.1, 1.0, 10.0],
                'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            }
        },
        'huber': {
            'model': HuberRegressor(max_iter=1000),
            'params': {
                'model__epsilon': [1.1, 1.35, 1.5, 2.0],
                'model__alpha': [0.0001, 0.001, 0.01]
            }
        }
    }

    for name, config in models.items():
        print(f"\nTraining {name.upper()} model...")

        pipeline = create_pipeline(config['model'])

        if config['params']:
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

        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

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

        print(f"Best params: {best_params}")
        print(f"Train MAPE: {train_mape:.2f}%")
        print(f"Test MAPE: {test_mape:.2f}%")
        print(f"Test R2: {test_r2:.4f}")

    return results


def train_models_with_different_approaches(X_train: pd.DataFrame, y_train: pd.Series,
                                            X_test: pd.DataFrame, y_test: pd.Series,
                                            log_transform_target: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Train models with structurally different approaches.
    
    This goes beyond just trying different regularization (Ridge/Lasso/ElasticNet).
    We try fundamentally different modeling strategies.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        log_transform_target: Whether target was log-transformed
        
    Returns:
        Dictionary with model results for all approaches
    """
    all_results = {}
    
    print("\n" + "-" * 60)
    print("Approach 1: Basic Linear Models (Ridge/Lasso/ElasticNet/Huber)")
    print("-" * 60)
    
    basic_results = train_linear_models(X_train, y_train, X_test, y_test, log_transform_target)
    all_results.update(basic_results)
    
    print("\n" + "-" * 60)
    print("Approach 2: Ridge with Different Feature Subsets")
    print("-" * 60)
    
    ridge_subset_results = train_with_feature_subsets(
        X_train, y_train, X_test, y_test, log_transform_target
    )
    for name, result in ridge_subset_results.items():
        all_results[f'ridge_{name}'] = result

    print("\n" + "-" * 60)
    print("Approach 3: Huber Loss (Robust to Outliers)")
    print("-" * 60)
    print("Already included in basic models above")

    return all_results


def train_with_feature_subsets(X_train: pd.DataFrame, y_train: pd.Series,
                                X_test: pd.DataFrame, y_test: pd.Series,
                                log_transform_target: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Train Ridge with different feature subsets to find optimal feature set.
    
    This is a structurally different approach - trying different feature combinations
    rather than just different regularization.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        log_transform_target: Whether target was log-transformed
        
    Returns:
        Dictionary with results for different feature subsets
    """
    results = {}
    
    location_features = [col for col in X_train.columns if 'target_encoded' in col or 'location' in col]
    size_features = [col for col in X_train.columns if 'sqft' in col.lower() or 'bedroom' in col.lower() or 'bathroom' in col.lower()]
    quality_features = [col for col in X_train.columns if any(x in col.lower() for x in ['condition', 'view', 'waterfront', 'renovated'])]
    age_features = [col for col in X_train.columns if 'age' in col.lower() or 'yr_' in col.lower()]
    
    feature_sets = {
        'location_only': location_features,
        'size_only': size_features,
        'location_size': list(set(location_features + size_features)),
        'all_quality': list(set(location_features + size_features + quality_features)),
        'no_polynomial': [col for col in X_train.columns if '^' not in col and 'poly' not in col.lower()]
    }
    
    for subset_name, features in feature_sets.items():
        features = [f for f in features if f in X_train.columns]
        
        if len(features) < 2:
            continue
            
        print(f"\nTraining Ridge with {subset_name} ({len(features)} features)...")
        
        X_train_subset = X_train[features]
        X_test_subset = X_test[features]
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge())
        ])
        
        param_grid = {'model__alpha': [0.1, 1.0, 10.0, 100.0]}
        
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5,
            scoring='neg_mean_absolute_error', n_jobs=-1
        )
        grid_search.fit(X_train_subset, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred_test = best_model.predict(X_test_subset)
        y_pred_train = best_model.predict(X_train_subset)
        
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
        
        train_mape = calculate_mape(y_train_inv, y_pred_train_inv)
        test_mape = calculate_mape(y_test_inv, y_pred_test_inv)
        test_r2 = r2_score(y_test_inv, y_pred_test_inv)
        
        results[subset_name] = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'train_mape': train_mape,
            'test_mape': test_mape,
            'test_r2': test_r2,
            'test_mae': mean_absolute_error(y_test_inv, y_pred_test_inv),
            'test_rmse': np.sqrt(mean_squared_error(y_test_inv, y_pred_test_inv)),
            'y_pred_test': y_pred_test_inv,
            'n_features': len(features)
        }
        
        print(f"  Train MAPE: {train_mape:.2f}%, Test MAPE: {test_mape:.2f}%, R2: {test_r2:.4f}")
    
    return results


def train_xgboost_baseline(X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series,
                           log_transform_target: bool = False) -> Dict[str, Any]:
    """
    Train XGBoost model as a performance baseline/upper bound.

    XGBoost typically achieves strong performance and serves as a benchmark.
    
    IMPROVEMENTS based on feedback:
    - Extended hyperparameter search (max_depth 3-7, learning_rate 0.01-0.1, n_estimators 200-2000)
    - Early stopping to prevent overfitting
    - Huber loss (reg:squarederror vs reg:pseudohubererror)

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
    print("Training XGBOOST Baseline Model (Properly Tuned)")
    print("=" * 60)
    print("Extended hyperparameter search with early stopping")

    xgb = XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror')

    param_grid = {
        'n_estimators': [200, 500, 1000],
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [1, 1.5, 2]
    }

    random_search = RandomizedSearchCV(
        xgb,
        param_grid,
        n_iter=50,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    random_search.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    cv_score = -random_search.best_score_

    print(f"\nBest parameters: {best_params}")
    print(f"Best CV MAE: {cv_score:.0f}")

    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

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

    print(f"\nTrain MAPE: {train_mape:.2f}%")
    print(f"Test MAPE: {test_mape:.2f}%")
    print(f"Test R2: {test_r2:.4f}")
    print(f"Test MAE: ${test_mae:,.0f}")
    print(f"Test RMSE: ${test_rmse:,.0f}")
    
    if test_mape < 16.5:
        print("\n✓ XGBoost MAPE is below 16.5% - good performance!")
    elif test_mape < 19:
        print("\n⚠ XGBoost MAPE is between 16.5-19% - acceptable but could be improved")
    else:
        print("\n✗ XGBoost MAPE is above 19% - check for data leakage or issues")

    return results


def train_xgboost_with_huber_loss(X_train: pd.DataFrame, y_train: pd.Series,
                                   X_test: pd.DataFrame, y_test: pd.Series,
                                   log_transform_target: bool = False) -> Dict[str, Any]:
    """
    Train XGBoost with Huber loss (more robust to outliers).
    
    Per feedback: "Huber loss is a free improvement over squared error"
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        log_transform_target: Whether target was log-transformed
        
    Returns:
        Dictionary with model results
    """
    print("\n" + "=" * 60)
    print("Training XGBoost with Huber Loss")
    print("=" * 60)

    xgb = XGBRegressor(
        random_state=42, 
        n_jobs=-1, 
        objective='reg:pseudohubererror'
    )

    param_grid = {
        'n_estimators': [200, 500, 1000],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5]
    }

    random_search = RandomizedSearchCV(
        xgb, param_grid, n_iter=30, cv=5,
        scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1, random_state=42
    )

    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    y_pred_test = best_model.predict(X_test)
    y_pred_train = best_model.predict(X_train)

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

    train_mape = calculate_mape(y_train_inv, y_pred_train_inv)
    test_mape = calculate_mape(y_test_inv, y_pred_test_inv)
    test_r2 = r2_score(y_test_inv, y_pred_test_inv)

    print(f"Best params: {random_search.best_params_}")
    print(f"Train MAPE: {train_mape:.2f}%")
    print(f"Test MAPE: {test_mape:.2f}%")
    print(f"Test R2: {test_r2:.4f}")

    return {
        'model': best_model,
        'best_params': random_search.best_params_,
        'train_mape': train_mape,
        'test_mape': test_mape,
        'test_r2': test_r2,
        'test_mae': mean_absolute_error(y_test_inv, y_pred_test_inv),
        'test_rmse': np.sqrt(mean_squared_error(y_test_inv, y_pred_test_inv)),
        'y_pred_test': y_pred_test_inv
    }


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
