"""
Utility Functions Module

This module provides helper functions for various tasks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List


def print_metric_comparison(results: Dict[str, Dict[str, Any]]):
    """
    Print metric comparison for all models.

    Args:
        results: Dictionary with model results
    """
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"{'Model':<15} | {'Train MAPE (%)':<15} | {'Test MAPE (%)':<15} | {'Test R2':<12} | {'Test MAE ($)':<15}")
    print("-" * 80)

    # Separate linear models and XGBoost
    linear_models = {k: v for k, v in results.items() if k != 'xgboost'}
    xgboost_result = results.get('xgboost', None)

    # Print linear models
    for name, result in linear_models.items():
        print(f"{name.upper():<15} | {result['train_mape']:<15.2f} | {result['test_mape']:<15.2f} | {result['test_r2']:<12.4f} | {result['test_mae']:<15,.0f}")

    # Print XGBoost baseline
    if xgboost_result:
        print("-" * 80)
        print(f"{'XGBOOST (benchmark)':<15} | {xgboost_result['train_mape']:<15.2f} | {xgboost_result['test_mape']:<15.2f} | {xgboost_result['test_r2']:<12.4f} | {xgboost_result['test_mae']:<15,.0f}")
        print("=" * 80)

        # Calculate gap between best linear model and XGBoost
        best_linear_mape = min(v['test_mape'] for v in linear_models.values())
        best_linear_name = min(linear_models.keys(), key=lambda k: linear_models[k]['test_mape'])
        gap = best_linear_mape - xgboost_result['test_mape']

        print(f"\nBest Linear Model: {best_linear_name.upper()} (Test MAPE: {best_linear_mape:.2f}%)")
        print(f"XGBoost Benchmark:  (Test MAPE: {xgboost_result['test_mape']:.2f}%)")
        if gap > 0:
            print(f"Performance Gap: Linear model is {gap:.2f} percentage points worse than XGBoost (MAPE)")
        else:
            print(f"Performance Gap: Linear model outperforms XGBoost by {-gap:.2f} percentage points (MAPE)")


def plot_feature_importance(model: Any, feature_names: List[str], top_n: int = 20):
    """
    Plot feature importance from model.

    Args:
        model: Trained model
        feature_names: List of feature names
        top_n: Number of top features to plot
    """
    if hasattr(model.named_steps['model'], 'coef_'):
        coefficients = model.named_steps['model'].coef_

        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(coefficients)
        }).sort_values('importance', ascending=False).head(top_n)

        plt.figure(figsize=(14, 10))
        sns.barplot(x='importance', y='feature', data=feature_importance, palette='viridis')
        plt.title(f'Top {top_n} Feature Importances', fontsize=14, pad=20)
        plt.xlabel('Absolute Coefficient Value', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.yticks(fontsize=10)

        # Add value labels to bars
        for i, v in enumerate(feature_importance['importance']):
            plt.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig('results/feature_importance.png', dpi=300)
        plt.close()

        print(f"\nFeature importance plot saved as 'results/feature_importance.png'")


def plot_prediction_comparison(y_true: pd.Series, y_pred: pd.Series, model_name: str):
    """
    Plot prediction vs actual values.

    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Model name for plot title
    """
    # Remove extreme outliers (top 1% and bottom 1%) to improve visualization
    percentile_1 = y_true.quantile(0.01)
    percentile_99 = y_true.quantile(0.99)
    mask = (y_true >= percentile_1) & (y_true <= percentile_99)

    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=y_true_filtered, y=y_pred_filtered, alpha=0.6, s=60)
    plt.plot([y_true_filtered.min(), y_true_filtered.max()],
             [y_true_filtered.min(), y_true_filtered.max()], 'r--', lw=2)

    # Set reasonable axis limits with some padding
    padding = 0.1 * (y_true_filtered.max() - y_true_filtered.min())
    plt.xlim(y_true_filtered.min() - padding, y_true_filtered.max() + padding)
    plt.ylim(y_true_filtered.min() - padding, y_true_filtered.max() + padding)

    # Format axes for readability
    plt.xlabel('Actual Price ($)', fontsize=12)
    plt.ylabel('Predicted Price ($)', fontsize=12)
    plt.title(f'Actual vs Predicted Prices - {model_name.upper()}', fontsize=14, pad=20)

    # Add grid for better readability
    plt.grid(True, alpha=0.3)

    # Format tick labels to show thousands separators
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    plt.tight_layout()
    plt.savefig(f'results/prediction_comparison_{model_name}.png', dpi=300)
    plt.close()

    print(f"Prediction comparison plot saved as 'results/prediction_comparison_{model_name}.png'")


def save_results(results: Dict[str, Dict[str, Any]], file_path: str = 'results/model_results.csv'):
    """
    Save model results to CSV.

    Args:
        results: Model results dictionary
        file_path: Path to save CSV
    """
    data = []
    for model_name, result in results.items():
        data.append({
            'model': model_name,
            'train_mape': result.get('train_mape', None),
            'test_mape': result.get('test_mape', None),
            'test_mae': result.get('test_mae', None),
            'test_rmse': result.get('test_rmse', None),
            'test_r2': result.get('test_r2', None),
            'cv_score': result.get('cv_score', None)
        })

    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"Results saved to {file_path}")


def clean_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove infinite values from dataframe.

    Args:
        df: Input dataframe

    Returns:
        Dataframe with infinite values removed
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df


def check_multicollinearity(df: pd.DataFrame, threshold: float = 0.9) -> List[str]:
    """
    Check for multicollinearity between features.

    Args:
        df: Features dataframe
        threshold: Correlation threshold to consider

    Returns:
        List of highly correlated features to remove
    """
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    if to_drop:
        print(f"Features to drop due to high multicollinearity: {to_drop}")

    return to_drop


def remove_collinear_features(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """
    Remove collinear features.

    Args:
        df: Features dataframe
        threshold: Correlation threshold

    Returns:
        Dataframe with collinear features removed
    """
    to_drop = check_multicollinearity(df, threshold)
    return df.drop(to_drop, axis=1)
