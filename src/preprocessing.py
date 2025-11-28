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

from config import (
    QUALITY_MAP, BASEMENT_EXPOSURE_MAP, BASEMENT_FINISH_MAP,
    GARAGE_FINISH_MAP, FENCE_MAP, SKEWNESS_THRESHOLD, BOXCOX_LAMBDA
)


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Handle missing values with domain knowledge.

    Strategy:
    - NA means "None" for features where absence is meaningful
    - NA means 0 for associated numeric features
    - Neighborhood-based imputation for LotFrontage
    - Mode/median for remaining features

    Parameters
    ----------
    None

    Attributes
    ----------
    none_features : list
        Features where NA means "None"
    zero_features : list
        Features where NA means 0

    """

    def __init__(self):
        self.none_features = [
            'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
            'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
            'MasVnrType'
        ]

        self.zero_features = [
            'GarageYrBlt', 'GarageArea', 'GarageCars',
            'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
            'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
        ]

    def fit(self, X, y=None):
        """Fit the transformer (no-op for this transformer)."""
        return self

    def transform(self, X:pd.DataFrame):
        """
        Transform the data by imputing missing values.

        Parameters
        ----------
        X : pd.DataFrame
            Input features

        Returns
        -------
        X : pd.DataFrame
            Transformed features with imputed values
        """
        X  = X.copy()

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
            X['LotFrontage'] = X.groupby('Neighborhood')['LotFrontage'].transform(
                lambda x: x.fillna(x.median())
            )

        # Remaining: fill with mode (categorical) or median (numerical)
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                if X[col].dtype == 'object':
                    X[col] = X[col].fillna(X[col].mode()[0])
                else:
                    X[col] = X[col].fillna(X[col].median())

        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode categorical features with proper ordering.

    Uses ordinal encoding for quality features (Ex > Gd > TA > Fa > Po)
    and label encoding for nominal features.

    Parameters
    ----------
    None

    Attributes
    ----------
    label_encoders : dict
        Dictionary of fitted LabelEncoder objects
    quality_cols : list
        Columns to encode with quality mapping
    """

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