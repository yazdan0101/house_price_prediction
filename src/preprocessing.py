"""
Data preprocessing transformers for the house price prediction pipeline.

Contains custom sklearn transformers for:
- Missing value imputation
- Categorical encoding
- Skewness correction
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew
from scipy.special import boxcox1p

from .config import (
    QUALITY_MAP, BASEMENT_EXPOSURE_MAP, BASEMENT_FINISH_MAP,
    GARAGE_FINISH_MAP, FENCE_MAP, SKEWNESS_THRESHOLD, BOXCOX_LAMBDA
)


class MissingValueHandler(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.none_features = [
            'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
            'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
            'MasVnrType', 'MSSubClass'
        ]

        self.zero_features = [
            'GarageYrBlt', 'GarageArea', 'GarageCars',
            'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
            'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
        ]

        self.categorical_modes = {}
        self.numerical_medians = {}
        self.neighborhood_lotfrontage = {}

    def fit(self, X, y=None):
        """
        Fit the transformer by learning mode/median values from training data.

        Parameters
        ----------
        X : pd.DataFrame
            Input features (training data)
        y : None
            Ignored

        Returns
        -------
        self
        """
        X = X.copy()

        # Apply Category 1 and 2 first
        for col in self.none_features:
            if col in X.columns:
                X[col] = X[col].fillna('None')

        for col in self.zero_features:
            if col in X.columns:
                X[col] = X[col].fillna(0)

        # Learn neighborhood-based LotFrontage medians
        if 'LotFrontage' in X.columns and 'Neighborhood' in X.columns:
            self.neighborhood_lotfrontage = X.groupby('Neighborhood')['LotFrontage'].median().to_dict()
            # Store overall median as fallback
            self.neighborhood_lotfrontage['__overall__'] = X['LotFrontage'].median()

        # Learn modes for ALL categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if X[col].notna().sum() > 0:
                mode_values = X[col].mode()
                self.categorical_modes[col] = mode_values[0] if len(mode_values) > 0 else 'Missing'
            else:
                self.categorical_modes[col] = 'Missing'

        # Learn medians for ALL numerical features
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if X[col].notna().sum() > 0:
                self.numerical_medians[col] = X[col].median()
            else:
                self.numerical_medians[col] = 0

        return self

    def transform(self, X):
        """
        Transform the data by imputing missing values using learned statistics.

        Parameters
        ----------
        X : pd.DataFrame
            Input features

        Returns
        -------
        X : pd.DataFrame
            Transformed features with imputed values
        """
        X = X.copy()

        # Category 1: NA means "None"
        for col in self.none_features:
            if col in X.columns:
                X[col] = X[col].fillna('None')

        # Category 2: NA means 0
        for col in self.zero_features:
            if col in X.columns:
                X[col] = X[col].fillna(0)

        # Category 3: LotFrontage - fill with median by Neighborhood
        if 'LotFrontage' in X.columns and 'Neighborhood' in X.columns:
            if self.neighborhood_lotfrontage:
                # Use learned medians from training
                for idx in X.index:
                    if pd.isna(X.loc[idx, 'LotFrontage']):
                        neighborhood = X.loc[idx, 'Neighborhood']
                        # Use neighborhood median if available, else overall median
                        if neighborhood in self.neighborhood_lotfrontage:
                            X.loc[idx, 'LotFrontage'] = self.neighborhood_lotfrontage[neighborhood]
                        else:
                            X.loc[idx, 'LotFrontage'] = self.neighborhood_lotfrontage.get('__overall__', 0)

            # Fill any remaining NaN with overall median
            if X['LotFrontage'].isna().any():
                overall_median = self.neighborhood_lotfrontage.get('__overall__', 0)
                X['LotFrontage'] = X['LotFrontage'].fillna(overall_median)

        # Category 4: Fill ALL remaining missing values with learned statistics
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                if X[col].dtype == 'object':
                    # Use learned mode
                    fill_value = self.categorical_modes.get(col, 'Missing')
                    X[col] = X[col].fillna(fill_value)
                else:
                    # Use learned median
                    fill_value = self.numerical_medians.get(col, 0)
                    X[col] = X[col].fillna(fill_value)

        # Final safety check: fill any remaining NaN with 0
        if X.isnull().sum().sum() > 0:
            print(f"⚠️  Warning: {X.isnull().sum().sum()} remaining NaN values. Filling with 0...")
            X = X.fillna(0)

        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoders = {}
        self.quality_cols = [
            'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
            'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC'
        ]

    def fit(self, X, y=None):
        """
        Fit label encoders for nominal categorical features.

        Parameters
        ----------
        X : pd.DataFrame
            Input features
        y : None
            Ignored

        Returns
        -------
        self
        """
        X = X.copy()

        # Apply ordinal encoding first
        X = self._apply_ordinal_encoding(X)

        # Fit label encoders for remaining categorical features
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = X[col].astype(str)
            le.fit(X[col])
            self.label_encoders[col] = le

        return self

    def transform(self, X):
        """
        Transform categorical features to numeric.

        Parameters
        ----------
        X : pd.DataFrame
            Input features

        Returns
        -------
        X : pd.DataFrame
            Encoded features
        """
        X = X.copy()

        # Apply ordinal encoding
        X = self._apply_ordinal_encoding(X)

        # Apply label encoding
        for col in X.select_dtypes(include=['object']).columns:
            if col in self.label_encoders:
                X[col] = X[col].astype(str)
                le = self.label_encoders[col]
                # Handle unseen categories
                X[col] = X[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

        return X

    def _apply_ordinal_encoding(self, X):
        """Apply ordinal encoding to quality features."""
        # Quality features
        for col in self.quality_cols:
            if col in X.columns:
                X[col] = X[col].map(QUALITY_MAP).fillna(0)

        # Basement exposure
        if 'BsmtExposure' in X.columns:
            X['BsmtExposure'] = X['BsmtExposure'].map(BASEMENT_EXPOSURE_MAP).fillna(0)

        # Basement finish
        for col in ['BsmtFinType1', 'BsmtFinType2']:
            if col in X.columns:
                X[col] = X[col].map(BASEMENT_FINISH_MAP).fillna(0)

        # Garage finish
        if 'GarageFinish' in X.columns:
            X['GarageFinish'] = X['GarageFinish'].map(GARAGE_FINISH_MAP).fillna(0)

        # Fence
        if 'Fence' in X.columns:
            X['Fence'] = X['Fence'].map(FENCE_MAP).fillna(0)

        return X


class SkewnessFixer(BaseEstimator, TransformerMixin):
    """
    Fix skewed features using Box-Cox transformation.

    Applies Box-Cox transformation to features with |skewness| > threshold
    to make distributions more normal, improving model performance.

    Parameters
    ----------
    threshold : float, default=0.75
        Skewness threshold for transformation
    lam : float, default=0.15
        Box-Cox lambda parameter

    Attributes
    ----------
    skewed_features : list
        Features identified as skewed during fit

    Examples
    --------
    >>> fixer = SkewnessFixer(threshold=0.75, lam=0.15)
    >>> fixer.fit(X_train)
    >>> X_fixed = fixer.transform(X_test)
    """

    def __init__(self, threshold=SKEWNESS_THRESHOLD, lam=BOXCOX_LAMBDA):
        self.threshold = threshold
        self.lam = lam
        self.skewed_features = None

    def fit(self, X, y=None):
        """
        Identify skewed features.

        Parameters
        ----------
        X : pd.DataFrame
            Input features
        y : None
            Ignored

        Returns
        -------
        self
        """
        X = X.copy()

        # Find skewed features
        numeric_feats = X.select_dtypes(include=[np.number]).columns
        skewness = X[numeric_feats].apply(lambda x: skew(x.dropna()))
        self.skewed_features = skewness[abs(skewness) > self.threshold].index.tolist()

        return self

    def transform(self, X):
        """
        Apply Box-Cox transformation to skewed features.

        Parameters
        ----------
        X : pd.DataFrame
            Input features

        Returns
        -------
        X : pd.DataFrame
            Transformed features
        """
        X = X.copy()

        # Apply Box-Cox transformation
        for feat in self.skewed_features:
            if feat in X.columns:
                X[feat] = boxcox1p(X[feat], self.lam)

        return X