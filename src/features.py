
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import EXPENSIVE_NEIGHBORHOODS


class FeatureEngineer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.created_features = []
        self.neighborhood_stats = None

    def fit(self, X, y=None):
        X = X.copy()

        # Initialize neighborhood stats
        self.neighborhood_stats = {}

        # Learn neighborhood statistics from training data
        if 'Neighborhood' not in X.columns:
            return self

        # If we have target variable, calculate neighborhood price stats
        if y is not None:
            # Handle both Series and array
            if isinstance(y, pd.Series):
                y_values = y.values
            else:
                y_values = y

            # Create temporary dataframe for groupby
            temp_df = X.copy()
            temp_df['_target'] = y_values

            # Calculate neighborhood statistics
            neighborhood_groups = temp_df.groupby('Neighborhood')

            self.neighborhood_stats['median_price'] = neighborhood_groups['_target'].median().to_dict()
            self.neighborhood_stats['mean_price'] = neighborhood_groups['_target'].mean().to_dict()
            self.neighborhood_stats['std_price'] = neighborhood_groups['_target'].std().fillna(0).to_dict()

        # Calculate neighborhood feature statistics (always available)
        if 'GrLivArea' in X.columns:
            self.neighborhood_stats['mean_area'] = X.groupby('Neighborhood')['GrLivArea'].mean().to_dict()
            self.neighborhood_stats['median_area'] = X.groupby('Neighborhood')['GrLivArea'].median().to_dict()

        if 'OverallQual' in X.columns:
            self.neighborhood_stats['mean_qual'] = X.groupby('Neighborhood')['OverallQual'].mean().to_dict()
            self.neighborhood_stats['median_qual'] = X.groupby('Neighborhood')['OverallQual'].median().to_dict()

        if 'YearBuilt' in X.columns:
            self.neighborhood_stats['mean_year'] = X.groupby('Neighborhood')['YearBuilt'].mean().to_dict()

        # Make a list of stat names first to avoid RuntimeError
        stat_names = list(self.neighborhood_stats.keys())

        for stat_name in stat_names:
            stat_dict = self.neighborhood_stats[stat_name]
            if stat_dict:  # Only if dictionary is not empty
                overall_key = f'{stat_name}_overall'
                overall_value = np.mean(list(stat_dict.values()))
                self.neighborhood_stats[overall_key] = overall_value

        return self

    def transform(self, X):
        X = X.copy()
        original_cols = set(X.columns)

        # Apply all feature engineering methods
        X = self._create_age_features(X)
        X = self._create_area_features(X)
        X = self._create_ratio_features(X)
        X = self._create_quality_interactions(X)
        X = self._create_quality_area_interactions(X)
        X = self._create_boolean_features(X)
        X = self._create_polynomial_features(X)
        X = self._create_neighborhood_features(X)
        X = self._create_temporal_features(X)

        # ⭐ PHASE 1: ADVANCED FEATURES
        X = self._create_neighborhood_aggregated_features(X)
        X = self._create_advanced_interactions(X)
        X = self._create_polynomial_log_features(X)
        X = self._create_advanced_temporal_features(X)

        # Track created features
        self.created_features = list(set(X.columns) - original_cols)

        # ✅ SAFETY CHECK: Replace any inf/-inf with 0
        X = X.replace([np.inf, -np.inf], 0)

        # ✅ SAFETY CHECK: Fill any remaining NaN with 0
        if X.isnull().sum().sum() > 0:
            print(f"⚠️  Warning: Feature engineering created {X.isnull().sum().sum()} NaN values. Filling with 0...")
            X = X.fillna(0)

        return X

    # ========================================
    # ORIGINAL FEATURES
    # ========================================

    def _create_age_features(self, X):
        """Create age-related features."""
        X['HouseAge'] = X['YrSold'] - X['YearBuilt']
        X['RemodAge'] = X['YrSold'] - X['YearRemodAdd']
        X['GarageAge'] = X['YrSold'] - X['GarageYrBlt']
        X['GarageAge'] = X['GarageAge'].fillna(0)
        X['IsNew'] = (X['YearBuilt'] == X['YrSold']).astype(int)
        X['IsRemodeled'] = (X['YearBuilt'] != X['YearRemodAdd']).astype(int)

        return X

    def _create_area_features(self, X):
        """Create total area features."""
        # Total square footage (CRITICAL FEATURE!)
        X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']

        # Total outdoor space
        X['TotalPorchSF'] = (
                X['OpenPorchSF'] + X['EnclosedPorch'] +
                X['3SsnPorch'] + X['ScreenPorch'] + X['WoodDeckSF']
        )

        # Total bathrooms (full + 0.5 * half)
        X['TotalBath'] = (
                X['FullBath'] + 0.5 * X['HalfBath'] +
                X['BsmtFullBath'] + 0.5 * X['BsmtHalfBath']
        )

        return X

    def _create_ratio_features(self, X):
        """Create ratio features."""
        # Spaciousness metrics (add 1 to avoid division by zero)
        X['AreaPerRoom'] = X['GrLivArea'] / (X['TotRmsAbvGrd'] + 1)
        X['GarageAreaPerCar'] = X['GarageArea'] / (X['GarageCars'] + 1)

        # Completion/finish percentages
        X['BsmtFinPercent'] = (X['BsmtFinSF1'] + X['BsmtFinSF2']) / (X['TotalBsmtSF'] + 1)
        X['SecondFlrPercent'] = X['2ndFlrSF'] / (X['TotalSF'] + 1)

        # Lot utilization
        X['LotCoverage'] = X['1stFlrSF'] / (X['LotArea'] + 1)
        X['LivingToLot'] = X['GrLivArea'] / (X['LotArea'] + 1)

        # Component ratios
        X['BsmtToGround'] = X['TotalBsmtSF'] / (X['GrLivArea'] + 1)
        X['GarageToLiving'] = X['GarageArea'] / (X['GrLivArea'] + 1)

        return X

    def _create_quality_interactions(self, X):
        """Create quality interaction features."""
        # Overall quality metrics
        X['OverallQualCond'] = X['OverallQual'] * X['OverallCond']
        X['OverallQualGrade'] = X['OverallQual'] + X['OverallCond']
        X['ExterQualCond'] = X['ExterQual'] + X['ExterCond']

        # Component quality scores
        X['KitchenScore'] = X['KitchenAbvGr'] * X['KitchenQual']
        X['GarageScore'] = X['GarageQual'] + X['GarageCond']
        X['BsmtScore'] = X['BsmtQual'] + X['BsmtCond']
        X['FireplaceScore'] = X['Fireplaces'] * X['FireplaceQu']
        X['PoolScore'] = X['PoolArea'] * X['PoolQC']

        return X

    def _create_quality_area_interactions(self, X):
        """Create quality × area interaction features."""
        X['QualityArea'] = X['OverallQual'] * X['GrLivArea']
        X['QualityBath'] = X['OverallQual'] * X['TotalBath']
        X['QualityGarage'] = X['OverallQual'] * X['GarageArea']
        X['QualityBsmt'] = X['OverallQual'] * X['TotalBsmtSF']

        return X

    def _create_boolean_features(self, X):
        """Create boolean "has X" features."""
        X['HasPool'] = (X['PoolArea'] > 0).astype(int)
        X['HasGarage'] = (X['GarageArea'] > 0).astype(int)
        X['HasBsmt'] = (X['TotalBsmtSF'] > 0).astype(int)
        X['HasFireplace'] = (X['Fireplaces'] > 0).astype(int)
        X['HasPorch'] = (X['TotalPorchSF'] > 0).astype(int)
        X['Has2ndFloor'] = (X['2ndFlrSF'] > 0).astype(int)
        X['HasWoodDeck'] = (X['WoodDeckSF'] > 0).astype(int)
        X['HasOpenPorch'] = (X['OpenPorchSF'] > 0).astype(int)
        X['HasMasVnr'] = (X['MasVnrArea'] > 0).astype(int)

        return X

    def _create_polynomial_features(self, X):
        """Create polynomial (squared) features."""
        X['GrLivArea_Squared'] = X['GrLivArea'] ** 2
        X['TotalSF_Squared'] = X['TotalSF'] ** 2
        X['OverallQual_Squared'] = X['OverallQual'] ** 2

        return X

    def _create_neighborhood_features(self, X):
        """Create basic neighborhood features."""
        X['IsExpensiveNeighborhood'] = (
            X['Neighborhood'].isin(EXPENSIVE_NEIGHBORHOODS).astype(int)
        )

        return X

    def _create_temporal_features(self, X):
        """Create temporal (seasonal) features."""
        X['MonthSoldSin'] = np.sin(2 * np.pi * X['MoSold'] / 12)
        X['MonthSoldCos'] = np.cos(2 * np.pi * X['MoSold'] / 12)

        return X

    # ========================================
    # ⭐ PHASE 1: ADVANCED FEATURES
    # ========================================

    def _create_neighborhood_aggregated_features(self, X):
        if not self.neighborhood_stats or 'Neighborhood' not in X.columns:
            return X

        # Helper function to map with fallback
        def safe_map(series, mapping, fallback_key):
            mapped = series.map(mapping)
            # Fill NaN with overall statistic
            if fallback_key in self.neighborhood_stats:
                mapped = mapped.fillna(self.neighborhood_stats[fallback_key])
            return mapped

        # Neighborhood price statistics (if learned during fit)
        if 'median_price' in self.neighborhood_stats:
            X['Neighborhood_MedianPrice'] = safe_map(
                X['Neighborhood'],
                self.neighborhood_stats['median_price'],
                'median_price_overall'
            )

        if 'mean_price' in self.neighborhood_stats:
            X['Neighborhood_MeanPrice'] = safe_map(
                X['Neighborhood'],
                self.neighborhood_stats['mean_price'],
                'mean_price_overall'
            )

        if 'std_price' in self.neighborhood_stats:
            X['Neighborhood_StdPrice'] = safe_map(
                X['Neighborhood'],
                self.neighborhood_stats['std_price'],
                'std_price_overall'
            )

        # Neighborhood area statistics
        if 'mean_area' in self.neighborhood_stats:
            X['Neighborhood_MeanArea'] = safe_map(
                X['Neighborhood'],
                self.neighborhood_stats['mean_area'],
                'mean_area_overall'
            )

            # How does this house compare to neighborhood average?
            X['Area_vs_Neighborhood'] = X['GrLivArea'] / (X['Neighborhood_MeanArea'] + 1)

        # Neighborhood quality statistics
        if 'mean_qual' in self.neighborhood_stats:
            X['Neighborhood_MeanQual'] = safe_map(
                X['Neighborhood'],
                self.neighborhood_stats['mean_qual'],
                'mean_qual_overall'
            )

            # How does this house's quality compare to neighborhood?
            X['Qual_vs_Neighborhood'] = X['OverallQual'] / (X['Neighborhood_MeanQual'] + 1)

        # Neighborhood year built statistics
        if 'mean_year' in self.neighborhood_stats:
            X['Neighborhood_MeanYear'] = safe_map(
                X['Neighborhood'],
                self.neighborhood_stats['mean_year'],
                'mean_year_overall'
            )

            # Is this house newer or older than neighborhood average?
            X['Year_vs_Neighborhood'] = X['YearBuilt'] - X['Neighborhood_MeanYear']

        return X

    def _create_advanced_interactions(self, X):

        # Helper function to check if column is numeric
        def is_numeric(col_name):
            return col_name in X.columns and pd.api.types.is_numeric_dtype(X[col_name])

        # Price-relevant combinations (ONLY use already-numeric features)
        if is_numeric('BsmtQual'):
            X['Qual_Bsmt_Interaction'] = (X['OverallQual'] * X['TotalBsmtSF'] * X['BsmtQual']) / 1000
        else:
            X['Qual_Bsmt_Interaction'] = (X['OverallQual'] * X['TotalBsmtSF']) / 1000

        if is_numeric('ExterQual') and is_numeric('ExterCond'):
            X['Qual_Exterior_Interaction'] = X['OverallQual'] * X['ExterQual'] * X['ExterCond']
        else:
            X['Qual_Exterior_Interaction'] = X['OverallQual']

        # Luxury score (pool + fireplace + high quality + garage)
        X['LuxuryScore'] = (
                (X['PoolArea'] > 0).astype(int) * 3 +
                (X['Fireplaces'] > 0).astype(int) * 2 +
                (X['OverallQual'] >= 8).astype(int) * 5 +
                (X['GarageCars'] >= 3).astype(int) * 2
        )

        # Age × Quality (newer high-quality homes worth more)
        X['Age_Quality_Interaction'] = X['HouseAge'] * X['OverallQual']
        X['Remod_Quality_Interaction'] = X['RemodAge'] * X['OverallQual']

        # Total Quality Score (ONLY numeric quality features)
        quality_sum = X['OverallQual'] + X['OverallCond']

        # Add other quality features only if numeric
        if is_numeric('ExterQual'):
            quality_sum = quality_sum + X['ExterQual']
        if is_numeric('ExterCond'):
            quality_sum = quality_sum + X['ExterCond']
        if is_numeric('BsmtQual'):
            quality_sum = quality_sum + X['BsmtQual']
        if is_numeric('KitchenQual'):
            quality_sum = quality_sum + X['KitchenQual']
        if is_numeric('GarageQual'):
            quality_sum = quality_sum + X['GarageQual']
        if is_numeric('HeatingQC'):
            quality_sum = quality_sum + X['HeatingQC']

        X['TotalQualityScore'] = quality_sum

        # Bathrooms per bedroom ratio (luxury indicator)
        if 'BedroomAbvGr' in X.columns:
            X['BathPerBedroom'] = X['TotalBath'] / (X['BedroomAbvGr'] + 1)
        else:
            X['BathPerBedroom'] = X['TotalBath']

        # Living space quality-weighted
        X['QualityLivingSpace'] = X['GrLivArea'] * X['OverallQual'] / 1000

        # Finished vs unfinished space
        X['TotalFinishedSF'] = X['BsmtFinSF1'] + X['BsmtFinSF2'] + X['1stFlrSF'] + X['2ndFlrSF']
        X['FinishedSpaceRatio'] = X['TotalFinishedSF'] / (X['TotalSF'] + 1)

        # Porch quality interaction
        X['PorchQuality'] = X['TotalPorchSF'] * X['OverallQual'] / 100

        # Garage quality × cars (only if GarageQual is numeric)
        if is_numeric('GarageQual'):
            X['GarageQualityCars'] = X['GarageQual'] * X['GarageCars']
        else:
            X['GarageQualityCars'] = X['GarageCars']

        # Kitchen-Bath quality (only if KitchenQual is numeric)
        if is_numeric('KitchenQual'):
            X['KitchenBathQuality'] = X['KitchenQual'] + X['TotalBath']
        else:
            X['KitchenBathQuality'] = X['TotalBath']

        # Basement quality × area (only if BsmtQual is numeric)
        if is_numeric('BsmtQual'):
            X['BsmtQualityArea'] = X['BsmtQual'] * X['TotalBsmtSF'] / 100
        else:
            X['BsmtQualityArea'] = X['TotalBsmtSF'] / 100

        # Exterior quality × condition (only if both numeric)
        if is_numeric('ExterQual') and is_numeric('ExterCond'):
            X['ExteriorQualityScore'] = X['ExterQual'] * X['ExterCond']
        else:
            X['ExteriorQualityScore'] = 0

        return X

    def _create_polynomial_log_features(self, X):
        """
        Create polynomial and log-transformed features.

        These help with:
        - Reducing skewness (log)
        - Capturing non-linear relationships (polynomial)
        """
        # Log of important continuous features (reduces skewness)
        log_features = ['LotArea', 'GrLivArea', '1stFlrSF', 'TotalBsmtSF']
        for col in log_features:
            if col in X.columns:
                X[f'{col}_Log'] = np.log1p(X[col])

        # Square root features (another way to reduce skewness)
        sqrt_features = ['LotArea', 'LotFrontage']
        for col in sqrt_features:
            if col in X.columns:
                X[f'{col}_Sqrt'] = np.sqrt(X[col])

        # Cubic features for quality (exponential value increase)
        X['OverallQual_Cubed'] = X['OverallQual'] ** 3

        # More polynomial features
        X['TotalSF_Cubed'] = X['TotalSF'] ** 3
        X['TotalBath_Squared'] = X['TotalBath'] ** 2

        # Log of area-related features
        if 'TotalSF' in X.columns:
            X['TotalSF_Log'] = np.log1p(X['TotalSF'])

        # Inverse features (for ratios)
        X['Inverse_LotArea'] = 1 / (X['LotArea'] + 1)
        X['Inverse_GrLivArea'] = 1 / (X['GrLivArea'] + 1)

        return X

    def _create_advanced_temporal_features(self, X):
        """
        Create more sophisticated time-based features.

        Real estate has strong seasonal patterns!
        """
        # Season sold (better grouping than individual months)
        season_map = {
            12: 0, 1: 0, 2: 0,  # Winter
            3: 1, 4: 1, 5: 1,  # Spring
            6: 2, 7: 2, 8: 2,  # Summer
            9: 3, 10: 3, 11: 3  # Fall
        }
        X['Season'] = X['MoSold'].map(season_map)

        # Is it peak season? (May-July are best months for selling)
        X['IsPeakSeason'] = X['MoSold'].isin([5, 6, 7]).astype(int)

        # Is it off-season? (December-February)
        X['IsOffSeason'] = X['MoSold'].isin([12, 1, 2]).astype(int)

        # Years since remodel (depreciation factor)
        X['YearsSinceRemod'] = X['YrSold'] - X['YearRemodAdd']

        # Is recently remodeled? (within 5 years)
        X['IsRecentRemod'] = (X['YearsSinceRemod'] <= 5).astype(int)

        # Age at remodel
        X['AgeAtRemod'] = X['YearRemodAdd'] - X['YearBuilt']

        # Was remodeled when old? (might indicate good maintenance)
        X['WasRemodeledWhenOld'] = ((X['AgeAtRemod'] > 20) & (X['AgeAtRemod'] > 0)).astype(int)

        # Decade built (captures era-specific building styles)
        X['DecadeBuilt'] = (X['YearBuilt'] // 10) * 10

        # Is modern? (built after 2000)
        X['IsModern'] = (X['YearBuilt'] >= 2000).astype(int)

        # Is vintage? (built before 1950)
        X['IsVintage'] = (X['YearBuilt'] < 1950).astype(int)

        return X

    def get_feature_names(self):
        """Get list of created feature names."""
        return self.created_features


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Select features based on importance threshold.

    Optional: Use this to remove low-importance features.
    """

    def __init__(self, threshold=0.001, model=None):
        self.threshold = threshold
        self.model = model
        self.selected_features = None
        self.feature_importances = None

    def fit(self, X, y):
        """Fit feature selector."""
        if self.model is None:
            # If no model provided, keep all features
            if isinstance(X, pd.DataFrame):
                self.selected_features = X.columns.tolist()
            else:
                self.selected_features = list(range(X.shape[1]))
            return self

        # Train model to get feature importances
        self.model.fit(X, y)

        if isinstance(X, pd.DataFrame):
            feature_names = X.columns
        else:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        # Get importances
        importances = self.model.feature_importances_
        self.feature_importances = dict(zip(feature_names, importances))

        # Select features above threshold
        self.selected_features = [
            name for name, importance in self.featur e_importances.items()
            if importance >= self.threshold
        ]

        return self

    def transform(self, X):
        """Transform by selecting features."""
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features]
        else:
            return X[:, self.selected_features]