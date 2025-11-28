# 🏠 House Price Prediction - Advanced Regression

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive machine learning solution for predicting house prices using advanced regression techniques and feature engineering. Designed for the [Kaggle House Prices Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

**Target Performance:** RMSE < 0.115 (Top 1%)

---

## 🎯 Features

- **Clean sklearn pipeline architecture** - No data leakage, fully reproducible
- **100+ engineered features** - Domain-driven feature engineering
- **Advanced ensemble methods** - Stacking, blending, weighted averaging
- **Comprehensive preprocessing** - Intelligent missing value handling, outlier removal
- **Hyperparameter optimization** - GridSearchCV with cross-validation
- **Production-ready code** - Modular, tested, documented

---

## 📊 Results

| Model | CV RMSE | Public LB |
|-------|---------|-----------|
| ElasticNet | 0.1165 | - |
| Gradient Boosting | 0.1142 | - |
| XGBoost | 0.1128 | - |
| LightGBM | 0.1135 | - |
| **Stacking Ensemble** | **0.1098** | **0.1125** |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download data
Download `train.csv` and `test.csv` from [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) and place them in the `data/` directory.

### 4. Train models
```bash
python scripts/train.py
```

### 5. Generate predictions
```bash
python scripts/predict.py
```

Submission file will be saved to `outputs/submissions/submission.csv`

---

## 📂 Project Structure
```
house-price-prediction/
├── data/                  # Data files (not tracked)
├── notebooks/             # Jupyter notebooks for exploration
├── src/                   # Source code
│   ├── preprocessing.py   # Data preprocessing transformers
│   ├── features.py        # Feature engineering
│   ├── models.py          # Model definitions and ensembles
│   └── utils.py           # Utility functions
├── scripts/               # Training and prediction scripts
├── tests/                 # Unit tests
├── models/                # Saved model files
└── outputs/               # Outputs (submissions, plots)
```

---

## 🛠️ Usage

### Training from scratch
```python
from src.models import HousePriceModel

# Initialize model
model = HousePriceModel()

# Train with cross-validation
model.fit(X_train, y_train, cv=5)

# Evaluate
cv_score = model.cv_score_
print(f"CV RMSE: {cv_score:.4f}")
```

### Making predictions
```python
# Predict on test set
predictions = model.predict(X_test)

# Create submission
model.create_submission(test_ids, predictions, 'submission.csv')
```

### Custom pipeline
```python
from sklearn.pipeline import Pipeline
from src.preprocessing import MissingValueHandler
from src.features import FeatureEngineer

# Create custom pipeline
pipeline = Pipeline([
    ('missing', MissingValueHandler()),
    ('features', FeatureEngineer()),
    ('model', YourModel())
])

pipeline.fit(X_train, y_train)
```

---

## 🧪 Testing

Run unit tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

---

## 📈 Methodology

### 1. Data Preprocessing
- Intelligent missing value imputation (domain knowledge)
- Outlier removal (GrLivArea > 4000 with low price)
- Log transformation of target variable

### 2. Feature Engineering
- **Age features:** HouseAge, RemodAge, GarageAge
- **Area features:** TotalSF, TotalBath, TotalPorchSF
- **Ratio features:** AreaPerRoom, BsmtFinPercent
- **Quality interactions:** OverallQualCond, QualityArea
- **Boolean features:** HasPool, HasGarage, HasBsmt
- **Polynomial features:** GrLivArea², OverallQual²

See [FEATURES.md](docs/FEATURES.md) for complete feature documentation.

### 3. Models
- Linear models: Lasso, Ridge, ElasticNet
- Tree-based: Gradient Boosting, XGBoost, LightGBM
- Ensemble: Stacking with meta-learner

### 4. Evaluation
- 5-fold cross-validation
- RMSE on log-transformed prices
- Train/validation monitoring

---

## 📚 Documentation

- [Methodology](docs/METHODOLOGY.md) - Detailed approach and techniques
- [Features](docs/FEATURES.md) - Complete feature engineering guide
- [Results](docs/RESULTS.md) - Experiments and results analysis

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Kaggle House Prices Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- Dean De Cock for the Ames Housing dataset
- The Kaggle community for insights and discussions

---

## 📧 Contact

Yazdan Mohammadi - [https://www.yazdan.tech]

Project Link: [https://github.com/yazdan0101/house_price_prediction]

---

**⭐ If you found this helpful, please consider giving it a star!**