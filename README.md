# House Price Prediction with Linear Models

## Project Description

This repository implements a house price prediction system using linear regression models, adhering to the constraints and objectives of the DSS5104 assignment. The goal is to demonstrate that linear models can achieve strong predictive performance through careful feature engineering while maintaining interpretability.

## Dataset

- **Source**: `house_dataset.csv` (King County, WA housing data)
- **Original Size**: 9,200 records
- **After Deduplication**: 4,602 records
- **Duplicates Removed**: 4,598 records (49.98% of data)
- **After Outlier Removal**: 4,550 records
- **Features**: 18 raw features including location, size, condition, and sale details

## Project Structure

```
├── src/                   # Source code package
│   ├── __init__.py
│   ├── main.py            # Main pipeline orchestration
│   ├── data_loader.py     # Data loading and preprocessing (duplicate + outlier removal)
│   ├── feature_engineering.py  # Feature engineering (target encoding for location)
│   ├── model_training.py  # Model training and evaluation
│   └── utils.py           # Helper functions and visualization
├── results/               # Output results and reports
│   ├── model_results.csv  # Performance metrics for all models
│   ├── feature_importance.png
│   ├── prediction_comparison_ridge.png
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
3. `results/prediction_comparison_ridge.png` - Actual vs predicted prices scatter plot

## Project Components

### Data Loading and Preprocessing

- Load CSV data using pandas
- **Remove duplicate rows** (critical data quality step)
- **Remove price outliers** (price = 0 or price > $5M)
- Handle missing values
- Convert date column to datetime and extract features
- Separate numerical and categorical features

### Feature Engineering

The project implements several effective feature engineering techniques:

1. **Feature Transformations**: Log transformation, polynomial features
2. **Composite Features**: House age, renovation indicator, total square footage
3. **Categorical Encoding**: Target encoding (city, statezip), one-hot encoding
4. **Location Features**: Target encoding for city and zipcode (CRITICAL for performance!)

### Model Training

Trains and evaluates multiple linear regression models:
- Ordinary Least Squares (OLS)
- Ridge Regression
- Lasso Regression
- ElasticNet (L1 + L2 regularization)

XGBoost is used as a benchmark with hyperparameter tuning via grid search.

### Evaluation Metrics

- **MAPE**: Mean Absolute Percentage Error (primary metric)
- **R2 Score**: Coefficient of determination
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error

## Results

Best Linear Model: **Ridge**
- **Test MAPE**: 22.06%
- **Test R2**: 0.6077
- **Mean Absolute Error**: $110,818
- **Root Mean Squared Error**: $249,662

XGBoost Benchmark:
- **Test MAPE**: 18.13%
- **Test R2**: 0.6479

The performance gap between Ridge and XGBoost is 3.93 percentage points. Using target encoding for `city` and `statezip` was critical to achieving these strong results, confirming the instructor's guidance.

### Key Updates

1. **Data Deduplication**: Removed 4,598 duplicate rows (nearly 50% of the original data).
2. **Outlier Removal**: Removed 52 rows with price = 0 or price > $5M.
3. **Location Target Encoding**: Target encoding for `city` and `statezip` drove massive performance improvements.

## Report

A comprehensive project report is available in `results/project_report.md` which includes:
- Detailed feature engineering explanation
- Model performance comparison table
- Visualizations of feature importance and predictions
- Technical implementation details
- Data deduplication and outlier removal impact analysis

## Requirements and Constraints

- **Linear Models Only**: No neural networks, decision trees, random forests, or gradient boosting (XGBoost only as benchmark)
- **Creative Feature Engineering**: Encouraged (log transformation, target encoding, etc.)
- **Interpretability**: Models must remain interpretable
- **Data Quality**: Duplicate rows and outliers must be removed

## License

This project is for educational purposes only.
