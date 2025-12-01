"""
Training script for house price prediction model.

Usage:
    python scripts/train.py [--cv-folds 5] [--use-stacking] [--save-model]

This script:
1. Loads and preprocesses data
2. Trains multiple models with cross-validation
3. Creates ensemble predictions
4. Saves trained model
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE, MODEL_DIR
from src.utils import load_data, remove_outliers, prepare_target, print_cv_scores
from src.preprocessing import MissingValueHandler, CategoricalEncoder, SkewnessFixer
from src.features import FeatureEngineer
from src.models import HousePriceModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train house price prediction model')

    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )

    parser.add_argument(
        '--use-stacking',
        action='store_true',
        default=True,
        help='Use stacking ensemble (default: True)'
    )

    parser.add_argument(
        '--no-stacking',
        action='store_false',
        dest='use_stacking',
        help='Disable stacking ensemble'
    )

    parser.add_argument(
        '--save-model',
        action='store_true',
        default=True,
        help='Save trained model (default: True)'
    )

    parser.add_argument(
        '--model-name',
        type=str,
        default='house_price_model.pkl',
        help='Model filename (default: house_price_model.pkl)'
    )

    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        help='Specific models to train (default: all)'
    )

    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()

    print("=" * 80)
    print("HOUSE PRICE PREDICTION - TRAINING PIPELINE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  CV Folds: {args.cv_folds}")
    print(f"  Use Stacking: {args.use_stacking}")
    print(f"  Save Model: {args.save_model}")
    print(f"  Model Name: {args.model_name}")
    if args.models:
        print(f"  Selected Models: {args.models}")

    # ========================================
    # STEP 1: LOAD DATA
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)

    train, test, test_ids = load_data()

    # ========================================
    # STEP 2: REMOVE OUTLIERS
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 2: REMOVING OUTLIERS")
    print("=" * 80)

    train = remove_outliers(train)

    # ========================================
    # STEP 3: PREPARE TARGET
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 3: PREPARING TARGET")
    print("=" * 80)

    X_train, y_train = prepare_target(train)
    X_test = test.drop(['Id'], axis=1)

    # ========================================
    # STEP 4: PREPROCESSING PIPELINE
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 4: BUILDING PREPROCESSING PIPELINE")
    print("=" * 80)

    preprocessing_pipeline = Pipeline([
        ('missing_values', MissingValueHandler()),
        ('feature_engineering', FeatureEngineer()),
        ('categorical_encoding', CategoricalEncoder()),
        ('skewness_fixer', SkewnessFixer())
    ])

    print("Pipeline steps:")
    for i, (name, transformer) in enumerate(preprocessing_pipeline.steps, 1):
        print(f"  {i}. {name}")

    # ========================================
    # STEP 5: APPLY PREPROCESSING
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 5: APPLYING PREPROCESSING")
    print("=" * 80)

    # Combine for consistent transformation
    ntrain = len(X_train)
    all_data = pd.concat([X_train, X_test], axis=0, sort=False).reset_index(drop=True)

    print(f"Combined data shape: {all_data.shape}")

    # Fit and transform
    all_data_processed = preprocessing_pipeline.fit_transform(all_data)

    # Split back
    X_train_processed = all_data_processed[:ntrain]
    X_test_processed = all_data_processed[ntrain:]

    print(f"Processed train shape: {X_train_processed.shape}")
    print(f"Processed test shape: {X_test_processed.shape}")

    # Verify
    print(f"\nVerification:")
    print(f"  Missing values in train: {pd.DataFrame(X_train_processed).isnull().sum().sum()}")
    print(f"  Missing values in test: {pd.DataFrame(X_test_processed).isnull().sum().sum()}")
    print(f"  All numeric: {pd.DataFrame(X_train_processed).select_dtypes(include=['object']).shape[1] == 0}")

    # ========================================
    # STEP 6: TRAIN MODELS
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 6: TRAINING MODELS")
    print("=" * 80)

    model = HousePriceModel(
        use_stacking=args.use_stacking,
        model_names=args.models
    )

    model.fit(X_train_processed, y_train, cv=args.cv_folds, verbose=True)

    # ========================================
    # STEP 7: DISPLAY RESULTS
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 7: RESULTS")
    print("=" * 80)

    print_cv_scores(model.cv_scores)

    # Get CV scores DataFrame
    scores_df = model.get_cv_scores()
    print("\nDetailed Results:")
    print(scores_df.to_string(index=False))

    # ========================================
    # STEP 8: SAVE MODEL
    # ========================================
    if args.save_model:
        print("\n" + "=" * 80)
        print("STEP 8: SAVING MODEL")
        print("=" * 80)

        model.save(args.model_name)

        # Also save preprocessing pipeline
        import joblib
        pipeline_path = MODEL_DIR / 'preprocessing_pipeline.pkl'
        joblib.dump(preprocessing_pipeline, pipeline_path)
        print(f"✅ Preprocessing pipeline saved to {pipeline_path}")

    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)

    best_cv = min(model.cv_scores.values(), key=lambda x: x[0])[0]
    print(f"\n🏆 Best CV RMSE: {best_cv:.5f}")
    print(f"\nExpected Leaderboard Performance: ~{best_cv + 0.001:.5f}")
    print("\nNext steps:")
    print("  1. Run: python scripts/predict.py")
    print("  2. Submit: outputs/submissions/submission.csv to Kaggle")
    print("  3. Check your score!")


if __name__ == '__main__':
    main()