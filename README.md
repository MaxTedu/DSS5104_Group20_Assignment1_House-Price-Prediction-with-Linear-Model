# House Price Prediction with Linear Models

## Project Description

This repository implements a house price prediction system using linear regression models, adhering to the constraints and objectives of the DSS5104 assignment. The goal is to demonstrate that linear models can achieve strong predictive performance through careful feature engineering while maintaining interpretability.

## Dataset

| Attribute | Description |
|:----------|:------------|
| **Source** | `house_dataset.csv` (King County, WA housing data) |
| **Original Size** | 9,200 records |
| **After Deduplication** | 4,602 records |
| **Duplicates Removed** | 4,598 records (49.98% of data) |
| **After Outlier Removal** | 4,550 records |
| **Outliers Removed** | 52 records (49 with price=0, 3 with price>$5M) |
| **Features** | 18 raw features (location, size, condition, sale details) |

## Project Structure

```
├── src/                   # Source code package
│   ├── __init__.py
│   ├── main.py            # Main pipeline orchestration
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── feature_engineering.py  # Feature engineering (NO DATA LEAKAGE)
│   ├── model_training.py  # Model training and evaluation
│   └── utils.py           # Helper functions and visualization
├── results/               # Output results and reports
│   ├── model_results.csv  # Performance metrics for all models
│   ├── feature_importance.png
│   ├── prediction_comparison_elasticnet.png
│   └── project_report.md  # Comprehensive project report
├── dataset/
│   └── house_dataset.csv  # Housing price dataset
├── .venv/                 # Virtual environment
├── requirements.txt       # Project dependencies
└── README.md
```

## Installation

### Prerequisites

- Python 3.8 or later
- pip (Python package manager)

### Create Virtual Environment

```bash
python -m venv .venv
# For Windows
.venv\Scripts\activate
# For macOS/Linux
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Run Complete Pipeline

```bash
python -m src.main
```

### Output Files

After running the pipeline, the following files will be generated in the `results/` folder:

1. `results/model_results.csv` - Performance metrics for all models
2. `results/feature_importance.png` - Feature importance plot (top 20 features)
3. `results/prediction_comparison_elasticnet.png` - Actual vs predicted prices scatter plot

## Project Components

### Data Loading and Preprocessing

- Load CSV data using pandas
- **Remove duplicate rows** (critical data quality step - 4,598 duplicates removed)
- **Remove price outliers** (price = 0 or price > $5M)
- Handle missing values
- Convert date column to datetime and extract features
- Separate numerical and categorical features

### Feature Engineering

The project implements several effective feature engineering techniques:

| Type | Techniques | Purpose |
|:-----|:-----------|:--------|
| **Transformations** | Log, Polynomial (degree=2), Ratios | Reduce skewness, capture non-linearity |
| **Composite Features** | House age, Renovation indicator, sqft_per_bedroom | Domain-informed features |
| **Categorical Encoding** | Target encoding (city, statezip) | Handle categorical variables |
| **Location Features** | Target encoding for city and zipcode | Geographic grouping |
| **Interaction Features** | sqft_living × waterfront, sqft_living × view | Domain-specific interactions |

### Model Training

Trains and evaluates multiple linear regression models:
- Ordinary Least Squares (OLS)
- Ridge Regression
- Lasso Regression
- ElasticNet (L1 + L2 regularization)
- Huber Regressor (robust to outliers)

XGBoost is used as a benchmark with extended hyperparameter tuning via RandomizedSearchCV.

### Evaluation Metrics

- **MAPE**: Mean Absolute Percentage Error (primary metric, computed in PRICE SPACE)
- **R2 Score**: Coefficient of determination
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error

## Results

### Complete Model Comparison

| Model | Train MAPE (%) | Test MAPE (%) | Test R² | Test MAE ($) |
|:------|:--------------:|:-------------:|:-------:|:------------:|
| OLS | 18.19 | 23.02 | 0.5911 | 116,295 |
| Ridge | 18.22 | 22.97 | 0.6087 | 115,120 |
| Lasso | 18.34 | 22.91 | 0.6277 | 113,475 |
| **ElasticNet** | 18.43 | **22.85** | **0.6442** | **111,280** |
| Huber | 18.54 | 24.15 | 0.4510 | 118,858 |
| **XGBoost** | 10.98 | **17.50** | **0.6883** | **95,900** |

### Best Linear Model: ElasticNet

| Metric | Value |
|:-------|:------|
| **Test MAPE** | 22.85% |
| **Test R²** | 0.6442 |
| **Mean Absolute Error** | $111,280 |
| **Root Mean Squared Error** | $249,662 |

### XGBoost Benchmark

| Metric | Value |
|:-------|:------|
| **Test MAPE** | 17.50% |
| **Test R²** | 0.6883 |
| **Mean Absolute Error** | $95,900 |

### Performance Gap

| Comparison | Value |
|:-----------|:------|
| Best Linear Model (ElasticNet) | MAPE: 22.85% |
| XGBoost Benchmark | MAPE: 17.50% |
| **Performance Gap** | +5.34 percentage points |

## Key Improvements

### 1. Data Leakage Prevention (Critical Fix)

| Issue | Problem | Solution |
|:------|:--------|:---------|
| **Target Encoding Leakage** | Target encoding was computed on full dataset before train-test split | **Fixed**: Target encoding now computed ONLY on training set, then mapped to test set |

**Implementation**: `FeatureEngineeringPipeline` class follows fit/transform pattern:
- `fit()`: Computes target encoding maps from training data only
- `transform()`: Applies pre-computed maps to any data (train or test)

### 2. Extended XGBoost Tuning

| Parameter | Before | After |
|:----------|:-------|:------|
| n_estimators | 100, 200 | 200, 500, 1000 |
| max_depth | 4, 6, 8 | 3, 4, 5, 6, 7 |
| learning_rate | 0.05, 0.1 | 0.01, 0.03, 0.05, 0.1 |
| Search method | GridSearchCV | RandomizedSearchCV (50 iterations) |

### 3. Multiple Modeling Approaches

Beyond basic regularization variants, we tried:
- **Huber Regressor**: Robust to outliers
- **Feature subset experiments**: location_only, size_only, location_size, all_quality
- **Total**: 10+ different model configurations

### 4. Negative Results Documented

| Approach | Result | Why It Failed |
|:---------|:-------|:--------------|
| Location features only | 36.23% MAPE, R² = -0.11 | Insufficient information alone |
| Size features only | 30.79% MAPE | Missing critical location information |
| Huber Regressor | 24.15% MAPE | Outliers already removed, no benefit |

## Report

A comprehensive project report is available in `results/project_report.md` which includes:
- Detailed feature engineering explanation
- Data leakage prevention implementation
- Model performance comparison table
- Visualizations of feature importance and predictions
- Technical implementation details
- What worked and what didn't

## Requirements and Constraints

| Requirement | Status |
|:------------|:------:|
| Linear Models Only | ✅ OLS, Ridge, Lasso, ElasticNet, Huber |
| Feature Transformations | ✅ Log, polynomial, ratios |
| Categorical Encoding | ✅ Target encoding (NO DATA LEAKAGE) |
| MAPE in Price Space | ✅ Inverse transform applied |
| XGBoost Benchmark | ✅ Properly tuned |
| No Forbidden Methods | ✅ XGBoost only as benchmark |
| Duplicate Removal | ✅ 4,598 rows removed |
| Outlier Removal | ✅ 52 rows removed |
| Multiple Approaches | ✅ 10+ configurations tried |
| Negative Results Reported | ✅ Documented in report |

## License

This project is for educational purposes only.
