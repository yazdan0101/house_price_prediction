"""
Model definitions and ensemble methods for house price prediction.

Contains implementations of:
- Individual models with pipelines
- Stacking ensemble
- Weighted ensemble
- Model evaluation utilities
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import joblib

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb

    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from .config import (
    MODEL_PARAMS, RANDOM_STATE, N_FOLDS,
    ENSEMBLE_WEIGHTS, MODEL_DIR
)


class StackingEnsemble(BaseEstimator, RegressorMixin):
    """
    Stacking ensemble with cross-validated predictions.

    Level 1: Multiple base models generate out-of-fold predictions
    Level 2: Meta-model learns optimal combination

    This is more powerful than simple averaging because the meta-model
    learns when to trust each base model.

    Parameters
    ----------
    base_models : list
        List of base models (sklearn estimators)
    meta_model : estimator
        Meta-model to combine base predictions
    n_folds : int, default=5
        Number of folds for cross-validation

    Attributes
    ----------
    base_models_ : list of lists
        Fitted base models for each fold
    meta_model_ : estimator
        Fitted meta-model

    Examples
    --------
    >>> base = [Lasso(), Ridge(), ElasticNet()]
    >>> meta = Ridge()
    >>> stacking = StackingEnsemble(base, meta, n_folds=5)
    >>> stacking.fit(X_train, y_train)
    >>> predictions = stacking.predict(X_test)
    """

    def __init__(self, base_models, meta_model, n_folds=N_FOLDS):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.base_models_ = None
        self.meta_model_ = None

    def fit(self, X, y):
        """
        Fit stacking ensemble.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values

        Returns
        -------
        self
        """
        # ✅ Convert to numpy arrays to avoid indexing issues
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Initialize
        self.base_models_ = [[] for _ in self.base_models]
        self.meta_model_ = clone(self.meta_model)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=RANDOM_STATE)

        # Generate out-of-fold predictions for meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        print(f"  Training {len(self.base_models)} base models with {self.n_folds}-fold CV...")

        for i, model in enumerate(self.base_models):
            model_name = model.__class__.__name__ if not hasattr(model, 'steps') else 'Pipeline'
            print(f"    Base model {i + 1}/{len(self.base_models)}: {model_name}")

            for fold_num, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
                # Clone and fit model
                instance = clone(model)
                self.base_models_[i].append(instance)

                # Fit on training fold
                instance.fit(X[train_idx], y[train_idx])

                # Generate out-of-fold predictions
                y_pred = instance.predict(X[val_idx])
                out_of_fold_predictions[val_idx, i] = y_pred

        # Train meta-model on out-of-fold predictions
        print(f"  Training meta-model...")
        self.meta_model_.fit(out_of_fold_predictions, y)

        return self

    def predict(self, X):
        """
        Predict using stacking ensemble.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples

        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted values
        """
        # ✅ Convert to numpy array
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Generate predictions from each base model (average across folds)
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_
        ])

        # Meta-model prediction
        return self.meta_model_.predict(meta_features)


class HousePriceModel(BaseEstimator, RegressorMixin):
    """
    Complete house price prediction model with preprocessing and ensembling.

    This is the main model class that combines:
    - Preprocessing pipeline
    - Multiple base models
    - Stacking ensemble
    - Model persistence

    Parameters
    ----------
    use_stacking : bool, default=True
        Whether to use stacking ensemble
    model_names : list, optional
        List of model names to use

    Attributes
    ----------
    models : dict
        Dictionary of fitted models
    stacking : StackingEnsemble
        Fitted stacking ensemble
    cv_scores : dict
        Cross-validation scores for each model
    best_model_name : str
        Name of best single model

    Examples
    --------
    >>> model = HousePriceModel(use_stacking=True)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> model.save('house_price_model.pkl')
    """

    def __init__(self, use_stacking=True, model_names=None):
        self.use_stacking = use_stacking
        self.model_names = model_names
        self.models = {}
        self.stacking = None
        self.cv_scores = {}
        self.best_model_name = None

    def _get_models(self):
        """Initialize models with configured parameters."""
        models = {}

        # Linear models with scaling
        models['lasso'] = Pipeline([
            ('scaler', RobustScaler()),
            ('model', Lasso(**MODEL_PARAMS['lasso']))
        ])

        models['ridge'] = Pipeline([
            ('scaler', RobustScaler()),
            ('model', Ridge(**MODEL_PARAMS['ridge']))
        ])

        models['elasticnet'] = Pipeline([
            ('scaler', RobustScaler()),
            ('model', ElasticNet(**MODEL_PARAMS['elasticnet']))
        ])

        # Tree-based models (no scaling needed)
        models['gradient_boosting'] = GradientBoostingRegressor(
            **MODEL_PARAMS['gradient_boosting']
        )

        if HAS_XGB:
            models['xgboost'] = xgb.XGBRegressor(**MODEL_PARAMS['xgboost'])

        if HAS_LGB:
            models['lightgbm'] = lgb.LGBMRegressor(**MODEL_PARAMS['lightgbm'])

        # Filter by requested model names
        if self.model_names:
            models = {k: v for k, v in models.items() if k in self.model_names}

        return models

    def fit(self, X, y, cv=N_FOLDS, verbose=True):
        """
        Fit all models with cross-validation.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        cv : int, default=5
            Number of cross-validation folds
        verbose : bool, default=True
            Whether to print progress

        Returns
        -------
        self
        """
        # ✅ Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y

        self.models = self._get_models()
        kfold = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

        if verbose:
            print("=" * 80)
            print("TRAINING MODELS")
            print("=" * 80)

        # Train and evaluate each model
        for name, model in self.models.items():
            if verbose:
                print(f"\nTraining {name}...")

            # Cross-validation
            scores = cross_val_score(
                model, X_array, y_array,
                cv=kfold,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            rmse = np.sqrt(-scores.mean())
            std = scores.std()

            self.cv_scores[name] = (rmse, std)

            if verbose:
                print(f"  CV RMSE: {rmse:.5f} (+/- {std:.5f})")

            # Train on full dataset
            model.fit(X_array, y_array)

        # Identify best model
        self.best_model_name = min(self.cv_scores.items(), key=lambda x: x[1][0])[0]

        if verbose:
            print(f"\n🏆 Best model: {self.best_model_name} "
                  f"(RMSE: {self.cv_scores[self.best_model_name][0]:.5f})")

        # Create stacking ensemble if requested
        if self.use_stacking and len(self.models) >= 3:
            if verbose:
                print("\nCreating stacking ensemble...")

            # Select top 3 models as base
            sorted_models = sorted(self.cv_scores.items(), key=lambda x: x[1][0])
            base_model_names = [name for name, _ in sorted_models[:3]]
            base_models = [self.models[name] for name in base_model_names]

            # Use Lasso as meta-model
            meta_model = Pipeline([
                ('scaler', RobustScaler()),
                ('model', Lasso(alpha=0.001, random_state=RANDOM_STATE))
            ])

            self.stacking = StackingEnsemble(
                base_models=base_models,
                meta_model=meta_model,
                n_folds=cv
            )

            # Fit stacking ensemble
            self.stacking.fit(X_array, y_array)

            # Evaluate stacking
            stacking_scores = cross_val_score(
                self.stacking, X_array, y_array,
                cv=kfold,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            stacking_rmse = np.sqrt(-stacking_scores.mean())
            self.cv_scores['stacking'] = (stacking_rmse, stacking_scores.std())

            if verbose:
                print(f"  Base models: {base_model_names}")
                print(f"  Stacking CV RMSE: {stacking_rmse:.5f}")

        if verbose:
            print("\n" + "=" * 80)
            print("TRAINING COMPLETE")
            print("=" * 80)

        return self

    def predict(self, X, use_ensemble=True):
        """
        Generate predictions.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples
        use_ensemble : bool, default=True
            Whether to use weighted ensemble of all models

        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted values
        """
        # ✅ Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        if use_ensemble:
            return self._predict_ensemble(X_array)
        else:
            return self.models[self.best_model_name].predict(X_array)

    def _predict_ensemble(self, X):
        """Generate weighted ensemble predictions."""
        predictions = {}

        # Get predictions from all models
        for name, model in self.models.items():
            predictions[name] = model.predict(X)

        # Add stacking if available
        if self.stacking is not None:
            predictions['stacking'] = self.stacking.predict(X)

        # Calculate weights based on CV performance
        weights = self._calculate_weights()

        # Weighted average
        ensemble_pred = sum(
            weights.get(name, 0) * pred
            for name, pred in predictions.items()
        )

        return ensemble_pred

    def _calculate_weights(self):
        """Calculate ensemble weights based on CV performance."""
        # Inverse RMSE weighting
        weights = {}
        total_inverse_rmse = sum(1 / rmse for rmse, _ in self.cv_scores.values())

        for name, (rmse, _) in self.cv_scores.items():
            weights[name] = (1 / rmse) / total_inverse_rmse

        return weights

    def save(self, filepath):
        """
        Save model to disk.

        Parameters
        ----------
        filepath : str or Path
            Path to save model
        """
        filepath = MODEL_DIR / filepath if not str(filepath).startswith('/') else filepath
        joblib.dump(self, filepath)
        print(f"✅ Model saved to {filepath}")

    @staticmethod
    def load(filepath):
        """
        Load model from disk.

        Parameters
        ----------
        filepath : str or Path
            Path to load model from

        Returns
        -------
        model : HousePriceModel
            Loaded model
        """
        filepath = MODEL_DIR / filepath if not str(filepath).startswith('/') else filepath
        model = joblib.load(filepath)
        print(f"✅ Model loaded from {filepath}")
        return model

    def get_cv_scores(self):
        """
        Get cross-validation scores.

        Returns
        -------
        scores_df : pd.DataFrame
            DataFrame with CV scores for each model
        """
        scores_data = []
        for name, (rmse, std) in sorted(self.cv_scores.items(), key=lambda x: x[1][0]):
            scores_data.append({
                'Model': name,
                'CV_RMSE': rmse,
                'Std': std
            })

        return pd.DataFrame(scores_data)


def evaluate_model(model, X, y, cv=N_FOLDS):
    """
    Evaluate a model using cross-validation.

    Parameters
    ----------
    model : estimator
        Model to evaluate
    X : array-like
        Features
    y : array-like
        Target
    cv : int, default=5
        Number of folds

    Returns
    -------
    mean_rmse : float
        Mean RMSE across folds
    std_rmse : float
        Standard deviation of RMSE
    """
    # ✅ Convert to numpy arrays
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    kfold = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        model, X, y,
        cv=kfold,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    rmse = np.sqrt(-scores)
    return rmse.mean(), rmse.std()


def rmse(y_true, y_pred):
    """
    Calculate RMSE.

    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values

    Returns
    -------
    float
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))