"""
Prediction script for house price prediction model.

Usage:
    python scripts/predict.py [--model-name house_price_model.pkl] [--output submission.csv]

This script:
1. Loads trained model and preprocessing pipeline
2. Loads test data
3. Generates predictions
4. Creates submission file
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib

from src.config import MODEL_DIR, SUBMISSION_DIR
from src.utils import load_data, create_submission
from src.models import HousePriceModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate predictions for house prices')

    parser.add_argument(
        '--model-name',
        type=str,
        default='house_price_model.pkl',
        help='Model filename (default: house_price_model.pkl)'
    )

    parser.add_argument(
        '--pipeline-name',
        type=str,
        default='preprocessing_pipeline.pkl',
        help='Preprocessing pipeline filename (default: preprocessing_pipeline.pkl)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='submission.csv',
        help='Output submission filename (default: submission.csv)'
    )

    parser.add_argument(
        '--use-ensemble',
        action='store_true',
        default=True,
        help='Use ensemble predictions (default: True)'
    )

    parser.add_argument(
        '--no-ensemble',
        action='store_false',
        dest='use_ensemble',
        help='Use best single model instead of ensemble'
    )

    return parser.parse_args()


def main():
    """Main prediction pipeline."""
    args = parse_args()

    print("=" * 80)
    print("HOUSE PRICE PREDICTION - PREDICTION PIPELINE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_name}")
    print(f"  Pipeline: {args.pipeline_name}")
    print(f"  Output: {args.output}")
    print(f"  Use Ensemble: {args.use_ensemble}")

    # ========================================
    # STEP 1: LOAD MODEL AND PIPELINE
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 1: LOADING MODEL AND PIPELINE")
    print("=" * 80)

    # Load model
    model_path = MODEL_DIR / args.model_name
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("Please run 'python scripts/train.py' first to train the model.")
        sys.exit(1)

    model = HousePriceModel.load(args.model_name)

    # Load preprocessing pipeline
    pipeline_path = MODEL_DIR / args.pipeline_name
    if not pipeline_path.exists():
        print(f"❌ Preprocessing pipeline not found: {pipeline_path}")
        print("Please run 'python scripts/train.py' first.")
        sys.exit(1)

    preprocessing_pipeline = joblib.load(pipeline_path)
    print(f"✅ Preprocessing pipeline loaded from {pipeline_path}")

    # ========================================
    # STEP 2: LOAD TEST DATA
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 2: LOADING TEST DATA")
    print("=" * 80)

    _, test, test_ids = load_data()
    X_test = test.drop(['Id'], axis=1)

    # ========================================
    # STEP 3: PREPROCESS TEST DATA
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 3: PREPROCESSING TEST DATA")
    print("=" * 80)

    X_test_processed = preprocessing_pipeline.transform(X_test)

    print(f"Processed test shape: {X_test_processed.shape}")
    print(f"Missing values: {pd.DataFrame(X_test_processed).isnull().sum().sum()}")

    # ========================================
    # STEP 4: GENERATE PREDICTIONS
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 4: GENERATING PREDICTIONS")
    print("=" * 80)

    # Predictions in log space
    predictions_log = model.predict(X_test_processed, use_ensemble=args.use_ensemble)

    print(f"Generated {len(predictions_log)} predictions")
    print(f"Prediction statistics (log scale):")
    print(f"  Min: {predictions_log.min():.4f}")
    print(f"  Max: {predictions_log.max():.4f}")
    print(f"  Mean: {predictions_log.mean():.4f}")
    print(f"  Median: {np.median(predictions_log):.4f}")

    # ========================================
    # STEP 5: CREATE SUBMISSION
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 5: CREATING SUBMISSION")
    print("=" * 80)

    submission = create_submission(
        test_ids=test_ids,
        predictions=predictions_log,
        filename=args.output,
        inverse_log=True
    )

    # ========================================
    # STEP 6: VERIFY SUBMISSION
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 6: VERIFYING SUBMISSION")
    print("=" * 80)

    # Check format
    assert list(submission.columns) == ['Id', 'SalePrice'], "Wrong columns!"
    assert len(submission) == len(test_ids), "Wrong number of rows!"
    assert not submission['SalePrice'].isnull().any(), "Missing predictions!"
    assert (submission['SalePrice'] > 0).all(), "Negative predictions!"

    print("✅ All verification checks passed!")

    # Display sample
    print("\nSample predictions:")
    print(submission.head(10).to_string(index=False))

    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "=" * 80)
    print("PREDICTION COMPLETE!")
    print("=" * 80)

    print(f"\n📊 Submission Statistics:")
    print(f"   File: {SUBMISSION_DIR / args.output}")
    print(f"   Predictions: {len(submission)}")
    print(f"   Price Range: ${submission['SalePrice'].min():,.0f} - ${submission['SalePrice'].max():,.0f}")
    print(f"   Mean Price: ${submission['SalePrice'].mean():,.0f}")
    print(f"   Median Price: ${submission['SalePrice'].median():,.0f}")

    print("\n🚀 Next Steps:")
    print("   1. Go to: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques")
    print("   2. Click 'Submit Predictions'")
    print(f"   3. Upload: {SUBMISSION_DIR / args.output}")
    print("   4. Check your score!")

    # Show model performance
    if hasattr(model, 'cv_scores'):
        print("\n📈 Expected Performance:")
        best_cv = min(model.cv_scores.values(), key=lambda x: x[0])[0]
        print(f"   CV RMSE: {best_cv:.5f}")
        print(f"   Expected LB: ~{best_cv + 0.001:.5f}")


if __name__ == '__main__':
    main()