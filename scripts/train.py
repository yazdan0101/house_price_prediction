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
    print(f"✅ Loaded {len(train)} training samples")
    print(f"✅ Loaded {len(test)} test samples")

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

    print("Fitting preprocessing pipeline on training data...")

    try:
        # Fit ONLY on training data with target (to learn neighborhood stats)
        preprocessing_pipeline.fit(X_train, y_train)
        print("✅ Pipeline fitted successfully")

        print("\nTransforming training data...")
        X_train_processed = preprocessing_pipeline.transform(X_train)
        print("✅ Training data transformed")

        print("Transforming test data...")
        X_test_processed = preprocessing_pipeline.transform(X_test)
        print("✅ Test data transformed")

    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        raise

    print(f"\nProcessed train shape: {X_train_processed.shape}")
    print(f"Processed test shape: {X_test_processed.shape}")

    # Verify
    print(f"\n📊 Verification:")
    train_missing = pd.DataFrame(X_train_processed).isnull().sum().sum()
    test_missing = pd.DataFrame(X_test_processed).isnull().sum().sum()
    train_objects = pd.DataFrame(X_train_processed).select_dtypes(include=['object']).shape[1]

    print(f"  Missing values in train: {train_missing}")
    print(f"  Missing values in test: {test_missing}")
    print(f"  All numeric: {train_objects == 0}")

    if train_missing > 0 or test_missing > 0:
        print("  ⚠️  WARNING: Missing values detected!")
    if train_objects > 0:
        print("  ⚠️  WARNING: Non-numeric columns detected!")

    # ⭐ NEW: Show feature engineering stats
    feature_engineer = preprocessing_pipeline.named_steps['feature_engineering']
    print(f"\n🎨 Feature Engineering Stats:")
    print(f"  Original features: 80")
    print(f"  Features created: {len(feature_engineer.created_features)}")
    print(f"  Total features: {X_train_processed.shape[1]}")

    if feature_engineer.neighborhood_stats:
        print(
            f"  Neighborhood statistics learned: {len([k for k in feature_engineer.neighborhood_stats.keys() if not k.endswith('_overall')])} types")
        if 'median_price' in feature_engineer.neighborhood_stats:
            print(f"  Neighborhoods tracked: {len(feature_engineer.neighborhood_stats['median_price'])}")

    # ========================================
    # STEP 6: TRAIN MODELS
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 6: TRAINING MODELS")
    print("=" * 80)

    try:
        model = HousePriceModel(
            use_stacking=args.use_stacking,
            model_names=args.models
        )

        model.fit(X_train_processed, y_train, cv=args.cv_folds, verbose=True)
        print("✅ Models trained successfully")

    except Exception as e:
        print(f"❌ Error during training: {e}")
        raise

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

        try:
            model.save(args.model_name)

            # Also save preprocessing pipeline
            import joblib
            pipeline_path = MODEL_DIR / 'preprocessing_pipeline.pkl'
            joblib.dump(preprocessing_pipeline, pipeline_path)
            print(f"✅ Preprocessing pipeline saved to {pipeline_path}")

        except Exception as e:
            print(f"❌ Error saving model: {e}")
            raise

    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "=" * 80)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 80)

    best_cv = min(model.cv_scores.values(), key=lambda x: x[0])[0]
    print(f"\n🏆 Best CV RMSE: {best_cv:.5f}")
    print(f"\n📈 Expected Leaderboard Performance: ~{best_cv + 0.001:.5f} to ~{best_cv + 0.003:.5f}")

    # Estimate rank improvement
    if best_cv < 0.114:
        print(f"\n🎯 Target achieved! You should be in Top 1% territory!")
    elif best_cv < 0.116:
        print(f"\n💪 Good progress! Close to Top 1% (target: < 0.114)")

    print("\n📋 Next steps:")
    print("  1. Run: python scripts/predict.py")
    print("  2. Submit: outputs/submissions/submission.csv to Kaggle")
    print("  3. Check your score and compare with CV!")
    print("  4. If LB score >> CV score: possible overfitting")
    print("  5. If LB score << CV score: trust your CV, public LB is noisy")


if __name__ == '__main__':
    main()