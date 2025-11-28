"""
Configuration file for the house price prediction project.

Contains all hyperparameters, file paths, and constants.
"""

import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent

# Data paths
DATA_DIR = ROOT_DIR / 'data'
TRAIN_PATH = DATA_DIR / 'train.csv'
TEST_PATH = DATA_DIR / 'test.csv'

# Output paths
OUTPUT_DIR = ROOT_DIR / 'outputs'
SUBMISSION_DIR = OUTPUT_DIR / 'submissions'
VISUALIZATION_DIR = OUTPUT_DIR / 'visualizations'
MODEL_DIR = ROOT_DIR / 'models'

# Create directories if they don't exist
for directory in [SUBMISSION_DIR, VISUALIZATION_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Random seed for reproducibility
RANDOM_STATE = 42

# Cross-validation settings
N_FOLDS = 5
CV_SHUFFLE = True

# Feature engineering settings
SKEWNESS_THRESHOLD = 0.75
BOXCOX_LAMBDA = 0.15

# Model hyperparameters
MODEL_PARAMS = {
    'lasso': {
        'alpha': 0.0005,
        'random_state': RANDOM_STATE,
        'max_iter': 10000
    },
    'ridge': {
        'alpha': 10,
        'random_state': RANDOM_STATE
    },
    'elasticnet': {
        'alpha': 0.0005,
        'l1_ratio': 0.9,
        'random_state': RANDOM_STATE,
        'max_iter': 10000
    },
    'gradient_boosting': {
        'n_estimators': 3000,
        'learning_rate': 0.05,
        'max_depth': 4,
        'max_features': 'sqrt',
        'min_samples_leaf': 15,
        'min_samples_split': 10,
        'loss': 'huber',
        'random_state': RANDOM_STATE
    },
    'xgboost': {
        'colsample_bytree': 0.4603,
        'gamma': 0.0468,
        'learning_rate': 0.05,
        'max_depth': 3,
        'min_child_weight': 1.7817,
        'n_estimators': 2200,
        'reg_alpha': 0.4640,
        'reg_lambda': 0.8571,
        'subsample': 0.5213,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbosity': 0
    },
    'lightgbm': {
        'objective': 'regression',
        'num_leaves': 5,
        'learning_rate': 0.05,
        'n_estimators': 720,
        'max_bin': 55,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'feature_fraction': 0.2319,
        'feature_fraction_seed': 9,
        'bagging_seed': 9,
        'min_data_in_leaf': 6,
        'min_sum_hessian_in_leaf': 11,
        'verbose': -1,
        'random_state': RANDOM_STATE
    }
}

# GridSearchCV parameters
PARAM_GRIDS = {
    'lasso': {
        'model__alpha': [0.0001, 0.0003, 0.0005, 0.0007, 0.001]
    },
    'ridge': {
        'model__alpha': [5, 8, 10, 12, 15]
    },
    'elasticnet': {
        'model__alpha': [0.0001, 0.0003, 0.0005, 0.0007],
        'model__l1_ratio': [0.8, 0.85, 0.9, 0.95, 0.99]
    }
}

# Ensemble weights (will be optimized during training)
ENSEMBLE_WEIGHTS = {
    'stacking': 0.30,
    'xgboost': 0.25,
    'lightgbm': 0.25,
    'gradient_boosting': 0.20
}

# Features to exclude (if any)
EXCLUDE_FEATURES = []

# Expensive neighborhoods (for feature engineering)
EXPENSIVE_NEIGHBORHOODS = ['NoRidge', 'NridgHt', 'StoneBr']

# Ordinal feature mappings
QUALITY_MAP = {
    'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5
}

BASEMENT_EXPOSURE_MAP = {
    'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4
}

BASEMENT_FINISH_MAP = {
    'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6
}

GARAGE_FINISH_MAP = {
    'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3
}

FENCE_MAP = {
    'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4
}

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'