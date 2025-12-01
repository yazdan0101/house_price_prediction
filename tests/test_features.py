"""
Unit tests for feature engineering.
"""

import unittest
import numpy as np
import pandas as pd

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.features import FeatureEngineer


class TestFeatureEngineer(unittest.TestCase):
    """Test FeatureEngineer transformer."""

    def setUp(self):
        """Create comprehensive sample data with ALL required columns."""
        self.data = pd.DataFrame({
            # Year features
            'YrSold': [2010, 2011, 2012],
            'YearBuilt': [2000, 1990, 2012],
            'YearRemodAdd': [2005, 1990, 2012],
            'GarageYrBlt': [2000, 1990, 2012],
            'MoSold': [6, 12, 3],

            # Area features
            'TotalBsmtSF': [1000, 800, 0],
            '1stFlrSF': [1000, 900, 800],
            '2ndFlrSF': [500, 0, 400],
            'GrLivArea': [1500, 900, 1200],
            'GarageArea': [400, 300, 0],
            'LotArea': [10000, 8000, 12000],
            'PoolArea': [0, 0, 0],
            'WoodDeckSF': [100, 0, 80],
            'OpenPorchSF': [50, 0, 60],
            'EnclosedPorch': [0, 0, 0],
            '3SsnPorch': [0, 0, 0],
            'ScreenPorch': [0, 0, 0],
            'MasVnrArea': [200, 0, 150],

            # Basement features
            'BsmtFinSF1': [500, 400, 0],
            'BsmtFinSF2': [0, 0, 0],
            'BsmtUnfSF': [500, 400, 0],

            # Bathroom features
            'FullBath': [2, 1, 2],
            'HalfBath': [1, 0, 1],
            'BsmtFullBath': [1, 0, 0],
            'BsmtHalfBath': [0, 0, 0],

            # Room features
            'TotRmsAbvGrd': [7, 5, 6],
            'BedroomAbvGr': [3, 2, 3],
            'KitchenAbvGr': [1, 1, 1],
            'Fireplaces': [1, 0, 1],

            # Garage features
            'GarageCars': [2, 1, 0],

            # Quality features (already encoded as numbers)
            'OverallQual': [7, 5, 8],
            'OverallCond': [6, 5, 7],
            'ExterQual': [4, 3, 5],
            'ExterCond': [3, 3, 4],
            'BsmtQual': [4, 3, 0],
            'BsmtCond': [3, 3, 0],
            'HeatingQC': [4, 3, 5],
            'KitchenQual': [4, 3, 5],
            'FireplaceQu': [4, 0, 5],
            'GarageQual': [3, 2, 0],
            'GarageCond': [3, 2, 0],
            'PoolQC': [0, 0, 0],

            # Neighborhood (already encoded as number)
            'Neighborhood': [1, 0, 2]  # NoRidge=1, Other=0, NridgHt=2
        })

    def test_transform_runs_without_error(self):
        """Test that transform completes without error."""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(self.data)

        # Should complete successfully
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result.columns), len(self.data.columns))

    def test_no_missing_values_introduced(self):
        """Test that feature engineering doesn't introduce NaN."""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(self.data)

        # No NaN should be introduced
        self.assertEqual(result.isnull().sum().sum(), 0,
                         "Feature engineering introduced NaN values")

    def test_no_infinite_values_introduced(self):
        """Test that feature engineering doesn't introduce inf."""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(self.data)

        # No inf should be introduced
        self.assertFalse(np.isinf(result).any().any(),
                         "Feature engineering introduced infinite values")

    def test_age_features_created(self):
        """Test that age features are created correctly."""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(self.data)

        # Check HouseAge
        self.assertIn('HouseAge', result.columns)
        self.assertEqual(result['HouseAge'].iloc[0], 10)  # 2010 - 2000

        # Check IsNew
        self.assertIn('IsNew', result.columns)
        self.assertEqual(result['IsNew'].iloc[2], 1)  # Built in 2012, sold in 2012
        self.assertEqual(result['IsNew'].iloc[0], 0)  # Built in 2000, sold in 2010

        # Check IsRemodeled
        self.assertIn('IsRemodeled', result.columns)
        self.assertEqual(result['IsRemodeled'].iloc[0], 1)  # Built 2000, remod 2005
        self.assertEqual(result['IsRemodeled'].iloc[1], 0)  # Built 1990, remod 1990

    def test_total_area_features_created(self):
        """Test that total area features are created correctly."""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(self.data)

        # Check TotalSF
        self.assertIn('TotalSF', result.columns)
        expected_total = 1000 + 1000 + 500  # TotalBsmtSF + 1stFlrSF + 2ndFlrSF
        self.assertEqual(result['TotalSF'].iloc[0], expected_total)

        # Check TotalBath
        self.assertIn('TotalBath', result.columns)
        expected_bath = 2 + 0.5 * 1 + 1 + 0  # FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath
        self.assertEqual(result['TotalBath'].iloc[0], expected_bath)

        # Check TotalPorchSF
        self.assertIn('TotalPorchSF', result.columns)
        expected_porch = 50 + 0 + 0 + 0 + 100  # All porch areas
        self.assertEqual(result['TotalPorchSF'].iloc[0], expected_porch)

    def test_ratio_features_created(self):
        """Test that ratio features are created correctly."""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(self.data)

        # Check AreaPerRoom
        self.assertIn('AreaPerRoom', result.columns)
        expected = 1500 / (7 + 1)  # GrLivArea / (TotRmsAbvGrd + 1)
        self.assertAlmostEqual(result['AreaPerRoom'].iloc[0], expected)

        # Check GarageAreaPerCar
        self.assertIn('GarageAreaPerCar', result.columns)
        expected = 400 / (2 + 1)  # GarageArea / (GarageCars + 1)
        self.assertAlmostEqual(result['GarageAreaPerCar'].iloc[0], expected)

    def test_quality_interactions_created(self):
        """Test that quality interaction features are created."""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(self.data)

        # Check OverallQualCond
        self.assertIn('OverallQualCond', result.columns)
        expected = 7 * 6  # OverallQual * OverallCond
        self.assertEqual(result['OverallQualCond'].iloc[0], expected)

        # Check KitchenScore
        self.assertIn('KitchenScore', result.columns)
        expected = 1 * 4  # KitchenAbvGr * KitchenQual
        self.assertEqual(result['KitchenScore'].iloc[0], expected)

    def test_quality_area_interactions_created(self):
        """Test that quality × area features are created."""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(self.data)

        # Check QualityArea
        self.assertIn('QualityArea', result.columns)
        expected = 7 * 1500  # OverallQual * GrLivArea
        self.assertEqual(result['QualityArea'].iloc[0], expected)

        # Check QualityBath
        self.assertIn('QualityBath', result.columns)
        # QualityBath = OverallQual * TotalBath
        # TotalBath = 2 + 0.5*1 + 1 + 0 = 3.5
        expected = 7 * 3.5
        self.assertEqual(result['QualityBath'].iloc[0], expected)

    def test_boolean_features_created(self):
        """Test that boolean features are created correctly."""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(self.data)

        # Check HasGarage
        self.assertIn('HasGarage', result.columns)
        self.assertEqual(result['HasGarage'].iloc[0], 1)  # Has garage
        self.assertEqual(result['HasGarage'].iloc[2], 0)  # No garage

        # Check HasBsmt
        self.assertIn('HasBsmt', result.columns)
        self.assertEqual(result['HasBsmt'].iloc[0], 1)  # Has basement
        self.assertEqual(result['HasBsmt'].iloc[2], 0)  # No basement

        # Check HasFireplace
        self.assertIn('HasFireplace', result.columns)
        self.assertEqual(result['HasFireplace'].iloc[0], 1)  # Has fireplace
        self.assertEqual(result['HasFireplace'].iloc[1], 0)  # No fireplace

    def test_polynomial_features_created(self):
        """Test that polynomial features are created."""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(self.data)

        # Check GrLivArea_Squared
        self.assertIn('GrLivArea_Squared', result.columns)
        expected = 1500 ** 2
        self.assertEqual(result['GrLivArea_Squared'].iloc[0], expected)

        # Check OverallQual_Squared
        self.assertIn('OverallQual_Squared', result.columns)
        expected = 7 ** 2
        self.assertEqual(result['OverallQual_Squared'].iloc[0], expected)

    def test_temporal_features_created(self):
        """Test that temporal features are created."""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(self.data)

        # Check MonthSoldSin and MonthSoldCos
        self.assertIn('MonthSoldSin', result.columns)
        self.assertIn('MonthSoldCos', result.columns)

        # Verify range [-1, 1]
        self.assertTrue((result['MonthSoldSin'] >= -1).all() and
                        (result['MonthSoldSin'] <= 1).all())
        self.assertTrue((result['MonthSoldCos'] >= -1).all() and
                        (result['MonthSoldCos'] <= 1).all())

    def test_created_features_tracked(self):
        """Test that created features are tracked."""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(self.data)

        # Should have list of created features
        self.assertIsInstance(engineer.created_features, list)
        self.assertGreater(len(engineer.created_features), 0)

        # Check some expected features
        self.assertIn('HouseAge', engineer.created_features)
        self.assertIn('TotalSF', engineer.created_features)
        self.assertIn('QualityArea', engineer.created_features)

    def test_handles_zero_values_safely(self):
        """Test that division by zero is handled safely."""
        # Create data with zeros
        zero_data = self.data.copy()
        zero_data['TotRmsAbvGrd'] = 0
        zero_data['GarageCars'] = 0
        zero_data['LotArea'] = 0

        engineer = FeatureEngineer()
        result = engineer.fit_transform(zero_data)

        # Should not have inf or NaN
        self.assertEqual(result.isnull().sum().sum(), 0)
        self.assertFalse(np.isinf(result).any().any())

    def test_output_shape(self):
        """Test that output has more features than input."""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(self.data)

        # Should have significantly more columns
        original_cols = len(self.data.columns)
        new_cols = len(result.columns)

        self.assertGreater(new_cols, original_cols)
        # Should have at least 40+ new features
        self.assertGreater(new_cols - original_cols, 40)


class TestFeatureEngineerEdgeCases(unittest.TestCase):
    """Test edge cases for FeatureEngineer."""

    def test_handles_missing_optional_columns(self):
        """Test that transformer handles missing optional columns gracefully."""
        # Minimal dataset
        minimal_data = pd.DataFrame({
            'YrSold': [2010, 2011],
            'YearBuilt': [2000, 1990],
            'YearRemodAdd': [2005, 1990],
            'GarageYrBlt': [2000, 1990],
            'MoSold': [6, 12],
            'TotalBsmtSF': [1000, 800],
            '1stFlrSF': [1000, 900],
            '2ndFlrSF': [500, 0],
            'GrLivArea': [1500, 900],
            'OverallQual': [7, 5],
            'OverallCond': [6, 5],
            'TotRmsAbvGrd': [7, 5],
            'GarageArea': [400, 300],
            'GarageCars': [2, 1],
            'FullBath': [2, 1],
            'HalfBath': [1, 0],
            'BsmtFullBath': [1, 0],
            'BsmtHalfBath': [0, 0],
            'LotArea': [10000, 8000],
            'PoolArea': [0, 0],
            'PoolQC': [0, 0],
            'Fireplaces': [0, 0],
            'FireplaceQu': [0, 0],
            'WoodDeckSF': [0, 0],
            'OpenPorchSF': [0, 0],
            'EnclosedPorch': [0, 0],
            '3SsnPorch': [0, 0],
            'ScreenPorch': [0, 0],
            'BsmtFinSF1': [500, 400],
            'BsmtFinSF2': [0, 0],
            'ExterQual': [4, 3],
            'ExterCond': [3, 3],
            'KitchenAbvGr': [1, 1],
            'KitchenQual': [4, 3],
            'GarageQual': [3, 2],
            'GarageCond': [3, 2],
            'BsmtQual': [4, 3],
            'BsmtCond': [3, 3],
            'MasVnrArea': [200, 0],
            'Neighborhood': [1, 0]
        })

        engineer = FeatureEngineer()
        result = engineer.fit_transform(minimal_data)

        # Should complete successfully
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.isnull().sum().sum(), 0)

    def test_all_zeros(self):
        """Test with all zero values."""
        zero_data = pd.DataFrame({
            'YrSold': [2010],
            'YearBuilt': [2010],
            'YearRemodAdd': [2010],
            'GarageYrBlt': [2010],
            'MoSold': [1],
            'TotalBsmtSF': [0],
            '1stFlrSF': [0],
            '2ndFlrSF': [0],
            'GrLivArea': [0],
            'OverallQual': [0],
            'OverallCond': [0],
            'TotRmsAbvGrd': [0],
            'GarageArea': [0],
            'GarageCars': [0],
            'FullBath': [0],
            'HalfBath': [0],
            'BsmtFullBath': [0],
            'BsmtHalfBath': [0],
            'LotArea': [0],
            'PoolArea': [0],
            'PoolQC': [0],
            'Fireplaces': [0],
            'FireplaceQu': [0],
            'WoodDeckSF': [0],
            'OpenPorchSF': [0],
            'EnclosedPorch': [0],
            '3SsnPorch': [0],
            'ScreenPorch': [0],
            'BsmtFinSF1': [0],
            'BsmtFinSF2': [0],
            'ExterQual': [0],
            'ExterCond': [0],
            'KitchenAbvGr': [0],
            'KitchenQual': [0],
            'GarageQual': [0],
            'GarageCond': [0],
            'BsmtQual': [0],
            'BsmtCond': [0],
            'MasVnrArea': [0],
            'Neighborhood': [0]
        })

        engineer = FeatureEngineer()
        result = engineer.fit_transform(zero_data)

        # Should handle all zeros safely
        self.assertEqual(result.isnull().sum().sum(), 0)
        self.assertFalse(np.isinf(result).any().any())


if __name__ == '__main__':
    unittest.main()