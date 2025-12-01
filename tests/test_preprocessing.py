"""
Unit tests for preprocessing transformers.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import MissingValueHandler, CategoricalEncoder, SkewnessFixer


class TestMissingValueHandler(unittest.TestCase):
    """Test MissingValueHandler transformer."""

    def setUp(self):
        """Create sample data with missing values."""
        self.data = pd.DataFrame({
            'PoolQC': ['Ex', 'Gd', np.nan, np.nan],
            'GarageArea': [500, 400, np.nan, 300],
            'LotFrontage': [80, np.nan, 90, 85],
            'Neighborhood': ['A', 'A', 'B', 'B'],
            'SomeFeature': [1, 2, 3, 4]
        })

    def test_none_features_filled(self):
        """Test that NA means None features are filled correctly."""
        handler = MissingValueHandler()
        result = handler.fit_transform(self.data)

        # PoolQC should have 'None' instead of NaN
        self.assertFalse(result['PoolQC'].isnull().any())
        self.assertEqual(result['PoolQC'].iloc[2], 'None')

    def test_zero_features_filled(self):
        """Test that NA means 0 features are filled correctly."""
        handler = MissingValueHandler()
        result = handler.fit_transform(self.data)

        # GarageArea should have 0 instead of NaN
        self.assertFalse(result['GarageArea'].isnull().any())
        self.assertEqual(result['GarageArea'].iloc[2], 0)

    def test_lotfrontage_filled_by_neighborhood(self):
        """Test that LotFrontage is filled with neighborhood median."""
        handler = MissingValueHandler()
        result = handler.fit_transform(self.data)

        # LotFrontage for Neighborhood A should be filled with median of A
        self.assertFalse(result['LotFrontage'].isnull().any())
        self.assertEqual(result['LotFrontage'].iloc[1], 80)  # Median of [80]

    def test_no_missing_values_remaining(self):
        """Test that all missing values are handled."""
        handler = MissingValueHandler()
        result = handler.fit_transform(self.data)

        self.assertEqual(result.isnull().sum().sum(), 0)


class TestCategoricalEncoder(unittest.TestCase):
    """Test CategoricalEncoder transformer."""

    def setUp(self):
        """Create sample data with categorical features."""
        self.data = pd.DataFrame({
            'ExterQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
            'BsmtExposure': ['Gd', 'Av', 'Mn', 'No', 'None'],
            'Neighborhood': ['A', 'B', 'A', 'C', 'B']
        })

    def test_ordinal_encoding(self):
        """Test that ordinal features are encoded with correct order."""
        encoder = CategoricalEncoder()
        result = encoder.fit_transform(self.data)

        # ExterQual should be encoded: Ex=5, Gd=4, TA=3, Fa=2, Po=1
        self.assertEqual(result['ExterQual'].iloc[0], 5)  # Ex
        self.assertEqual(result['ExterQual'].iloc[1], 4)  # Gd
        self.assertEqual(result['ExterQual'].iloc[2], 3)  # TA
        self.assertEqual(result['ExterQual'].iloc[3], 2)  # Fa
        self.assertEqual(result['ExterQual'].iloc[4], 1)  # Po

    def test_label_encoding(self):
        """Test that nominal features are label encoded."""
        encoder = CategoricalEncoder()
        encoder.fit(self.data)
        result = encoder.transform(self.data)

        # Neighborhood should be label encoded
        self.assertTrue(pd.api.types.is_numeric_dtype(result['Neighborhood']))

    def test_all_numeric_output(self):
        """Test that all output columns are numeric."""
        encoder = CategoricalEncoder()
        result = encoder.fit_transform(self.data)

        # All columns should be numeric
        for col in result.columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(result[col]))


class TestSkewnessFixer(unittest.TestCase):
    """Test SkewnessFixer transformer."""

    def setUp(self):
        """Create sample data with skewed feature."""
        # Create a highly skewed feature
        np.random.seed(42)
        normal = np.random.normal(100, 10, 100)
        skewed = np.random.exponential(50, 100)  # Right-skewed

        self.data = pd.DataFrame({
            'normal_feature': normal,
            'skewed_feature': skewed
        })

    def test_identifies_skewed_features(self):
        """Test that skewed features are correctly identified."""
        fixer = SkewnessFixer(threshold=0.5)
        fixer.fit(self.data)

        # skewed_feature should be identified
        self.assertIn('skewed_feature', fixer.skewed_features)
        # normal_feature should not be identified
        self.assertNotIn('normal_feature', fixer.skewed_features)

    def test_reduces_skewness(self):
        """Test that transformation reduces skewness."""
        from scipy.stats import skew

        original_skew = abs(skew(self.data['skewed_feature']))

        fixer = SkewnessFixer(threshold=0.5)
        result = fixer.fit_transform(self.data)

        transformed_skew = abs(skew(result['skewed_feature']))

        # Skewness should be reduced
        self.assertLess(transformed_skew, original_skew)


class TestPreprocessingPipeline(unittest.TestCase):
    """Test complete preprocessing pipeline."""

    def test_pipeline_runs_without_error(self):
        """Test that complete pipeline runs successfully."""
        # Create sample data
        data = pd.DataFrame({
            'PoolQC': ['Ex', np.nan, 'Gd'],
            'GarageArea': [500, np.nan, 400],
            'ExterQual': ['Ex', 'Gd', 'TA'],
            'LotArea': [10000, 15000, 8000],
            'Neighborhood': ['A', 'B', 'A'],
            'LotFrontage': [80, 90, np.nan]
        })

        # Create pipeline
        pipeline = Pipeline([
            ('missing_values', MissingValueHandler()),
            ('categorical_encoding', CategoricalEncoder()),
            ('skewness_fixer', SkewnessFixer())
        ])

        # Should run without error
        result = pipeline.fit_transform(data)

        # Verify output
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(data))
        self.assertEqual(result.isnull().sum().sum(), 0)


if __name__ == '__main__':
    unittest.main()