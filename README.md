# Crop Yield Forecasting and Alert System for GreenField Farmers Co-op

## 1. Project Overview

This project develops a machine learning pipeline to predict crop yield using agricultural features such as soil type, rainfall, temperature, and fertilizer usage.

The project uses a synthetic dataset to develop and evaluate a baseline regression model. It demonstrates data validation, feature preprocessing, model training, cross-validation, performance evaluation, and feature importance analysis.

**Current stage:** Task 2 — Feature Engineering and Baseline Model Training.

## 2. Objectives

- Generate and validate agricultural data.
- Prepare numerical and categorical features for machine learning.
- Train a Random Forest regression model.
- Evaluate the model using cross-validation and a held-out test set.
- Analyze feature importance.
- Save the trained model and evaluation results for future integration.

## 3. Dataset

Dataset file:

`data/synthetic_crop_yield.csv`

The dataset contains 1,000 records and six columns.

| Column | Description | Role |
|---|---|---|
| `field_id` | Unique identifier for each field | Identifier; excluded from model features |
| `soil_type` | Soil category | Input feature |
| `rainfall` | Rainfall measurement | Input feature |
| `temperature` | Temperature measurement | Input feature |
| `fertilizer` | Fertilizer measurement | Input feature |
| `yield` | Crop yield target | Target variable |

The dataset contains synthetic data generated for development and experimentation.

**Important limitation:** Synthetic data does not fully represent real agricultural conditions. Model performance on this dataset should not be interpreted as validated real-world farming performance.

## 4. Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

## 5. Project Structure

```text
crop-yield-forecasting-and-alert-system-for-greenfield-farmers/
├── data/
│   └── synthetic_crop_yield.csv
├── notebooks/
├── src/
│   ├── generate_data.py
│   └── train_model.py
├── models/
│   └── crop_yield_model.pkl
├── reports/
│   ├── model_metrics.json
│   └── feature_importance.png
├── .gitignore
├── README.md
└── requirements.txt
```

The `Task/` directory contains private task documents and should remain excluded from the public repository.

## 6. Environment Setup

Python 3.10 or another compatible Python version is recommended.

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

## 7. Train the Model

Run the training script from the project root:

```powershell
python src/train_model.py
```

The script performs the following operations:

1. Loads and validates the dataset.
2. Separates input features from the target variable.
3. Applies numerical preprocessing and one-hot encoding to soil categories.
4. Splits the dataset into training and testing sets.
5. Evaluates the training split using five-fold cross-validation.
6. Trains a Random Forest regression pipeline.
7. Calculates test-set MAE and R².
8. Generates a feature importance chart.
9. Saves the trained pipeline and evaluation metrics.

The experiment uses an 80/20 train/test split and a fixed random seed of 42 for reproducibility.

## 8. Model Evaluation

The current baseline model uses `RandomForestRegressor` with 200 trees.

### Cross-Validation Results

| Metric | Result |
|---|---:|
| Number of folds | 5 |
| Mean CV MAE | 0.3549 |
| Mean CV R² | 0.8447 |

### Held-Out Test Results

| Metric | Result |
|---|---:|
| Test MAE | 0.3413 |
| Test R² | 0.8380 |

The target values are interpreted in tonnes per hectare, as assumed by the project.

**MAE (Mean Absolute Error)** measures the average absolute difference between predicted and actual yield values. Lower values indicate smaller errors on the evaluated dataset.

**R² (Coefficient of Determination)** measures model performance relative to a mean-prediction baseline. Its interpretation depends on the dataset and evaluation setup.

These results describe performance on the current synthetic dataset only.

## 9. Feature Importance

The model generates a feature importance chart at:

`reports/feature_importance.png`

The current model ranks rainfall, loamy soil, and temperature among its most important features.

Feature importance describes how the fitted Random Forest uses features internally. It does not establish causal relationships between agricultural inputs and crop yield.

## 10. Generated Artifacts

| Artifact | Description |
|---|---|
| `models/crop_yield_model.pkl` | Saved preprocessing and regression pipeline |
| `reports/model_metrics.json` | Evaluation metrics and feature importance values |
| `reports/feature_importance.png` | Feature importance visualization |

The saved pipeline can be loaded for subsequent predictions using the same preprocessing steps applied during training.

Only load model files from trusted sources because loading untrusted pickle-based artifacts can execute malicious code.

## 11. Example Prediction

The saved model was tested with the following sample input:

| Feature | Value |
|---|---:|
| Soil type | Loamy |
| Rainfall | 500 |
| Temperature | 25 |
| Fertilizer | 100 |

**Predicted crop yield:** 6.862 tonnes/hectare.

This is an example prediction from the current synthetic-data model, not a verified estimate for a real farm.

## 12. Limitations and Future Improvements

- Replace or supplement synthetic data with reliable real-world agricultural datasets.
- Validate feature units and measurement ranges against the selected data source.
- Evaluate performance across different soil types, weather conditions, and farming regions.
- Compare the baseline model with suitable alternative regression algorithms.
- Improve model monitoring and prediction validation.
- Integrate the trained pipeline into a prediction API.
- Develop an alert system only after defining and validating appropriate agricultural thresholds.

## 13. Current Status

The baseline machine learning pipeline has been implemented and evaluated on the synthetic crop yield dataset.

| Component | Status |
|---|---|
| Synthetic dataset generation | Completed |
| Dataset validation | Completed |
| Random Forest baseline training | Completed |
| Five-fold cross-validation | Completed |
| Test-set evaluation | Completed |
| Feature importance visualization | Completed |
| Model serialization | Completed |
| Sample prediction verification | Completed |

### Next Steps

- Validate the model using reliable real-world agricultural data.
- Compare the baseline model with alternative regression algorithms.
- Integrate the trained pipeline into a prediction API.
- Develop an alert system using appropriately validated agricultural thresholds.

**Note:** Current evaluation results are based on synthetic data and do not establish real-world agricultural accuracy.