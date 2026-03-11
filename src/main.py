"""
Main Pipeline Module

This module orchestrates the complete house price prediction workflow.
"""

import pandas as pd
import numpy as np

# Local modules
from .data_loader import load_data, preprocess_data, split_features_target
from .feature_engineering import engineer_features
from .model_training import (
    train_linear_models,
    train_xgboost_baseline,
    select_best_model,
    split_train_test,
    calculate_mape
)
from .utils import (
    print_metric_comparison,
    plot_feature_importance,
    plot_prediction_comparison,
    save_results,
    clean_infinite_values,
    remove_collinear_features
)


def main():
    """
    Main pipeline for house price prediction.
    """
    # Configuration
    DATA_PATH = "dataset/house_dataset.csv"
    LOG_TRANSFORM_TARGET = True
    COLLINEARITY_THRESHOLD = 0.95

    print("=" * 70)
    print("HOUSE PRICE PREDICTION WITH LINEAR MODELS")
    print("=" * 70)

    # Step 1: Load and preprocess data
    df = load_data(DATA_PATH)
    df = preprocess_data(df)

    # Step 2: Split features and target
    X, y = split_features_target(df)

    # Step 3: Feature engineering
    X = engineer_features(X, y)

    # Step 4: Clean data
    X = clean_infinite_values(X)
    y = y.loc[X.index]

    # Step 5: Remove highly collinear features
    print(f"Features before removing collinear: {X.shape[1]}")
    X = remove_collinear_features(X, COLLINEARITY_THRESHOLD)
    print(f"Features after removing collinear: {X.shape[1]}")

    # Step 6: Train-test split
    X_train, X_test, y_train, y_test = split_train_test(
        X, y,
        test_size=0.2,
        random_state=42,
        log_transform_target=LOG_TRANSFORM_TARGET
    )

    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    # Step 7: Train linear models
    print("\n" + "=" * 70)
    print("STEP 7: TRAINING LINEAR MODELS")
    print("=" * 70)

    results = train_linear_models(
        X_train, y_train,
        X_test, y_test,
        log_transform_target=LOG_TRANSFORM_TARGET
    )

    # Step 8: Train XGBoost baseline
    print("\n" + "=" * 70)
    print("STEP 8: TRAINING XGBOOST BASELINE")
    print("=" * 70)

    xgb_results = train_xgboost_baseline(
        X_train, y_train,
        X_test, y_test,
        log_transform_target=LOG_TRANSFORM_TARGET
    )
    results['xgboost'] = xgb_results

    # Step 9: Select best model (excluding XGBoost from selection, as it's a benchmark)
    linear_results = {k: v for k, v in results.items() if k != 'xgboost'}
    best_model_name, best_model_result = select_best_model(linear_results)

    # Step 9: Analysis and visualization
    print_metric_comparison(results)
    save_results(results)

    # Plot feature importance (if model supports it)
    if hasattr(best_model_result['model'].named_steps['model'], 'coef_'):
        plot_feature_importance(best_model_result['model'], X_train.columns)

    # Plot prediction comparison
    if LOG_TRANSFORM_TARGET:
        y_test_inv = np.expm1(y_test)
    else:
        y_test_inv = y_test

    plot_prediction_comparison(y_test_inv, best_model_result['y_pred_test'], best_model_name)

    print("\n" + "=" * 70)
    print("PREDICTION WORKFLOW COMPLETED")
    print("=" * 70)
    print(f"Best Model: {best_model_name.upper()}")
    print(f"Test MAPE: {best_model_result['test_mape']:.2f}%")
    print(f"Test R2: {best_model_result['test_r2']:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        print(traceback.format_exc())
