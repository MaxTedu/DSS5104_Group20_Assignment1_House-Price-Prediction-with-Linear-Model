# House Price Prediction with Linear Models

## Project Description

This repository implements a house price prediction system using linear regression models, adhering to the constraints and objectives of the DSS5104 assignment. The goal is to demonstrate that linear models can achieve strong predictive performance through careful feature engineering while maintaining interpretability.

## Dataset

- **Source**: `house_dataset.csv` (King County, WA housing data)
- **Size**: 9,200 records
- **Features**: 18 raw features including location, size, condition, and sale details

## Project Structure

```
├── src/                   # Source code package
│   ├── __init__.py
│   ├── main.py            # Main pipeline orchestration
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── feature_engineering.py  # Feature engineering and transformation
│   ├── model_training.py  # Model training and evaluation
│   └── utils.py           # Helper functions and visualization
├── results/               # Output results and reports
│   ├── model_results.csv  # Performance metrics for all models
│   ├── feature_importance.png
│   ├── prediction_comparison_elasticnet.png
│   └── project_report.md  # Comprehensive project report
├── dataset/
│   └── house_dataset.csv  # Housing price dataset
├── dss/                   # Virtual environment
├── requirements.txt       # Project dependencies
└── README.md
```

## Installation

### Prerequisites

- Python 3.8 or later
- pip (Python package manager)

### Create Virtual Environment

```bash
python -m venv dss
# For Windows
dss\Scripts\activate
# For macOS/Linux
source dss/bin/activate
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
- Handle missing values
- Convert date column to datetime and extract features
- Separate numerical and categorical features

### Feature Engineering

The project implements several effective feature engineering techniques:

1. **Feature Transformations**: Log transformation, polynomial features
2. **Composite Features**: House age, renovation indicator, total square footage
3. **Categorical Encoding**: One-hot encoding, frequency encoding, target encoding
4. **Location Features**: K-means clustering, target encoding

### Model Training

Trains and evaluates multiple linear regression models:
- Ordinary Least Squares (OLS)
- Ridge Regression
- Lasso Regression
- ElasticNet (L1 + L2 regularization)

### Evaluation Metrics

- **MAPE**: Mean Absolute Percentage Error (primary metric)
- **R2 Score**: Coefficient of determination
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error

## Results

Best Model: **ElasticNet**
- **Test MAPE**: 23.16%
- **Test R2**: 0.0961
- **Mean Absolute Error**: $160,762
- **Root Mean Squared Error**: $717,666

## Report

A comprehensive project report is available in `results/project_report.md` which includes:
- Detailed feature engineering explanation
- Model performance comparison table
- Visualizations of feature importance and predictions
- Technical implementation details

## Requirements and Constraints

- **Linear Models Only**: No neural networks, decision trees, random forests, or gradient boosting
- **Creative Feature Engineering**: Encouraged (log transformation, target encoding, etc.)
- **Interpretability**: Models must remain interpretable

## License

This project is for educational purposes only.
