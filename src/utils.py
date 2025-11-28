"""
Utility functions for the house price prediction project.

Contains helper functions for:
- Data loading
- Submission creation
- Visualization
- Logging
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .config import (
    TRAIN_PATH, TEST_PATH, SUBMISSION_DIR,
    VISUALIZATION_DIR, DATA_DIR
)


def load_data(train_path=TRAIN_PATH, test_path=TEST_PATH):
    """
    Load training and test data.

    Parameters
    ----------
    train_path : str or Path, optional
        Path to training data
    test_path : str or Path, optional
        Path to test data

    Returns
    -------
    train : pd.DataFrame
        Training data
    test : pd.DataFrame
        Test data
    test_ids : pd.Series
        Test IDs for submission

    Examples
    --------
    >>> train, test, test_ids = load_data()
    >>> print(f"Train: {train.shape}, Test: {test.shape}")
    """
    # Check if files exist
    if not Path(train_path).exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}\n"
            f"Please download from Kaggle and place in {DATA_DIR}"
        )

    if not Path(test_path).exists():
        raise FileNotFoundError(
            f"Test data not found at {test_path}\n"
            f"Please download from Kaggle and place in {DATA_DIR}"
        )

    # Load data
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    test_ids = test['Id'].copy()

    print(f"✅ Data loaded successfully")
    print(f"   Train: {train.shape}")
    print(f"   Test: {test.shape}")

    return train, test, test_ids


def remove_outliers(train):
    """
    Remove outliers from training data.

    Removes specific outliers identified from EDA:
    - Houses with GrLivArea > 4000 and SalePrice < 300000

    Parameters
    ----------
    train : pd.DataFrame
        Training data

    Returns
    -------
    train : pd.DataFrame
        Training data with outliers removed
    """
    original_size = len(train)

    # Remove specific outliers
    train = train.drop(
        train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index
    )

    train = train.reset_index(drop=True)

    print(f"✅ Removed {original_size - len(train)} outliers")

    return train


def prepare_target(train):
    """
    Prepare target variable (log transformation).

    Parameters
    ----------
    train : pd.DataFrame
        Training data

    Returns
    -------
    X : pd.DataFrame
        Features
    y : np.ndarray
        Log-transformed target
    """
    y = np.log1p(train['SalePrice']).values
    X = train.drop(['Id', 'SalePrice'], axis=1)

    print(f"✅ Target prepared (log-transformed)")
    print(f"   Original skewness: {train['SalePrice'].skew():.3f}")
    print(f"   Log-transformed skewness: {pd.Series(y).skew():.3f}")

    return X, y


def create_submission(test_ids, predictions, filename='submission.csv',
                      inverse_log=True):
    """
    Create submission file for Kaggle.

    Parameters
    ----------
    test_ids : array-like
        Test IDs
    predictions : array-like
        Predictions (in log space if inverse_log=True)
    filename : str, default='submission.csv'
        Output filename
    inverse_log : bool, default=True
        Whether to inverse log transform predictions

    Returns
    -------
    submission : pd.DataFrame
        Submission DataFrame

    Examples
    --------
    >>> submission = create_submission(test_ids, predictions)
    >>> print(submission.head())
    """
    # Inverse log transform if needed
    if inverse_log:
        predictions = np.expm1(predictions)

    # Create submission
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': predictions
    })

    # Save
    output_path = SUBMISSION_DIR / filename
    submission.to_csv(output_path, index=False)

    print(f"✅ Submission created: {output_path}")
    print(f"   Shape: {submission.shape}")
    print(f"   Price range: ${submission['SalePrice'].min():,.0f} - "
          f"${submission['SalePrice'].max():,.0f}")
    print(f"   Mean price: ${submission['SalePrice'].mean():,.0f}")

    return submission


def plot_predictions(y_true, y_pred, title='Predictions vs Actual',
                     save_name=None):
    """
    Plot predictions vs actual values.

    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    title : str, optional
        Plot title
    save_name : str, optional
        Filename to save plot
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             'r--', lw=2, label='Perfect prediction')
    plt.xlabel('Actual (log scale)')
    plt.ylabel('Predicted (log scale)')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)

    # Calculate metrics
    rmse_val = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)

    plt.text(0.05, 0.95, f'RMSE: {rmse_val:.4f}\nR²: {r2:.4f}',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if save_name:
        output_path = VISUALIZATION_DIR / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved: {output_path}")

    plt.show()


def plot_feature_importance(model, feature_names, top_n=20, save_name=None):
    """
    Plot feature importance.

    Parameters
    ----------
    model : estimator
        Fitted model with feature_importances_ attribute
    feature_names : list
        List of feature names
    top_n : int, default=20
        Number of top features to plot
    save_name : str, optional
        Filename to save plot
    """
    if not hasattr(model, 'feature_importances_'):
        print("⚠️ Model doesn't have feature_importances_ attribute")
        return

    # Get importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importances[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_name:
        output_path = VISUALIZATION_DIR / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved: {output_path}")

    plt.show()


def plot_residuals(y_true, y_pred, save_name=None):
    """
    Plot residual distribution.

    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    save_name : str, optional
        Filename to save plot
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residual plot
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residual Plot')
    axes[0].grid(alpha=0.3)

    # Residual distribution
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_name:
        output_path = VISUALIZATION_DIR / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved: {output_path}")

    plt.show()


def print_cv_scores(cv_scores):
    """
    Print cross-validation scores in a nice format.

    Parameters
    ----------
    cv_scores : dict
        Dictionary with model names as keys and (mean, std) tuples as values
    """
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 80)

    sorted_scores = sorted(cv_scores.items(), key=lambda x: x[1][0])

    for rank, (name, (mean_rmse, std_rmse)) in enumerate(sorted_scores, 1):
        print(f"{rank}. {name:20s}: {mean_rmse:.5f} (+/- {std_rmse:.5f})")

    print("=" * 80)