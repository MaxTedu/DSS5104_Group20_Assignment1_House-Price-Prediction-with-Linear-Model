"""
Main Pipeline Module

This module orchestrates the complete house price prediction workflow.

IMPORTANT: To avoid data leakage, feature engineering (especially target encoding)
must be done AFTER train-test split, using only training data to compute statistics.
"""

import pandas as pd
import numpy as np

from .data_loader import load_data, preprocess_data, split_features_target
from .feature_engineering import (
    FeatureEngineeringPipeline,
    create_spline_features,
    create_interaction_features,
    create_price_per_sqft_by_city,
    create_knn_features
)
from .model_training import (
    train_linear_models,
    train_xgboost_baseline,
    select_best_model,
    split_train_test,
    calculate_mape,
    train_models_with_different_approaches
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
    
    CRITICAL: Feature engineering is done AFTER train-test split to prevent data leakage.
    Target encoding is computed on training set only, then mapped to test set.
    """
    DATA_PATH = "dataset/house_dataset.csv"
    LOG_TRANSFORM_TARGET = True
    COLLINEARITY_THRESHOLD = 0.95

    print("=" * 70)
    print("HOUSE PRICE PREDICTION WITH LINEAR MODELS")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("STEP 1: LOADING AND PREPROCESSING DATA")
    print("=" * 70)
    
    df = load_data(DATA_PATH)
    df = preprocess_data(df)

    print("\n" + "=" * 70)
    print("STEP 2: SPLITTING FEATURES AND TARGET")
    print("=" * 70)
    
    X, y = split_features_target(df)

    print("\n" + "=" * 70)
    print("STEP 3: TRAIN-TEST SPLIT (BEFORE FEATURE ENGINEERING)")
    print("=" * 70)
    print("IMPORTANT: Splitting BEFORE feature engineering to prevent data leakage!")
    
    X_train, X_test, y_train, y_test = split_train_test(
        X, y,
        test_size=0.2,
        random_state=42,
        log_transform_target=False
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    print("\n" + "=" * 70)
    print("STEP 4: FEATURE ENGINEERING (FIT ON TRAIN ONLY)")
    print("=" * 70)
    print("Target encoding computed on training set, then mapped to test set!")
    
    fe_pipeline = FeatureEngineeringPipeline()
    X_train = fe_pipeline.fit_transform(X_train, y_train)
    X_test = fe_pipeline.transform(X_test)

    print("\n" + "=" * 70)
    print("STEP 5: ADDITIONAL FEATURE ENGINEERING")
    print("=" * 70)
    
    X_train, X_test = create_price_per_sqft_by_city(X_train, y_train, X_test)
    print("Added city_price_per_sqft feature")
    
    X_train = create_interaction_features(X_train)
    X_test = create_interaction_features(X_test)
    print("Added interaction features")

    print("\n" + "=" * 70)
    print("STEP 6: CLEANING DATA")
    print("=" * 70)
    
    X_train = clean_infinite_values(X_train)
    y_train = y_train.loc[X_train.index]
    
    X_test = clean_infinite_values(X_test)
    y_test = y_test.loc[X_test.index]
    
    print(f"Training data shape after cleaning: {X_train.shape}")
    print(f"Test data shape after cleaning: {X_test.shape}")

    print("\n" + "=" * 70)
    print("STEP 7: REMOVING HIGHLY CORRELATED FEATURES")
    print("=" * 70)
    
    print(f"Features before removing collinear: {X_train.shape[1]}")
    X_train = remove_collinear_features(X_train, COLLINEARITY_THRESHOLD)
    
    common_cols = [col for col in X_train.columns if col in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    print(f"Features after removing collinear: {X_train.shape[1]}")

    print("\n" + "=" * 70)
    print("STEP 8: LOG TRANSFORM TARGET")
    print("=" * 70)
    
    if LOG_TRANSFORM_TARGET:
        y_train = np.log1p(y_train)
        y_test = np.log1p(y_test)
        print("Target log-transformed for training")

    print(f"\nFinal training data shape: {X_train.shape}")
    print(f"Final test data shape: {X_test.shape}")

    print("\n" + "=" * 70)
    print("STEP 9: TRAINING MODELS WITH DIFFERENT APPROACHES")
    print("=" * 70)
    print("Trying structurally different modeling approaches (not just regularization variants)")
    
    all_results = train_models_with_different_approaches(
        X_train, y_train,
        X_test, y_test,
        log_transform_target=LOG_TRANSFORM_TARGET
    )

    print("\n" + "=" * 70)
    print("STEP 10: TRAINING XGBOOST BASELINE (PROPERLY TUNED)")
    print("=" * 70)
    
    xgb_results = train_xgboost_baseline(
        X_train, y_train,
        X_test, y_test,
        log_transform_target=LOG_TRANSFORM_TARGET
    )
    all_results['xgboost'] = xgb_results

    print("\n" + "=" * 70)
    print("STEP 11: SELECTING BEST MODEL")
    print("=" * 70)
    
    linear_results = {k: v for k, v in all_results.items() if k != 'xgboost'}
    best_model_name, best_model_result = select_best_model(linear_results)

    print_metric_comparison(all_results)
    save_results(all_results)

    if hasattr(best_model_result['model'].named_steps['model'], 'coef_'):
        plot_feature_importance(best_model_result['model'], X_train.columns)

    if LOG_TRANSFORM_TARGET:
        y_test_inv = np.expm1(y_test)
    else:
        y_test_inv = y_test

    plot_prediction_comparison(y_test_inv, best_model_result['y_pred_test'], best_model_name)

    print("\n" + "=" * 70)
    print("PREDICTION WORKFLOW COMPLETED")
    print("=" * 70)
    print(f"Best Linear Model: {best_model_name.upper()}")
    print(f"Test MAPE: {best_model_result['test_mape']:.2f}%")
    print(f"Test R2: {best_model_result['test_r2']:.4f}")
    print(f"\nXGBoost Benchmark MAPE: {xgb_results['test_mape']:.2f}%")
    print(f"Performance Gap: {best_model_result['test_mape'] - xgb_results['test_mape']:.2f} percentage points")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        print(traceback.format_exc())
