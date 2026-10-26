# Crop Yield Forecasting and Alert System for GreenField Farmers Co-op

## 1. Project Overview

This project develops a machine learning pipeline to predict crop yield using agricultural features such as soil type, rainfall, temperature, and fertilizer usage.

The project is organized into two main tasks:

- **Task 1 — Synthetic Dataset Generation:** Generate a synthetic agricultural dataset for model development and experimentation.
- **Task 2 — Feature Engineering and Baseline Model Training:** Preprocess the dataset, train a Random Forest regression model, evaluate its performance, and save the trained model and evaluation reports.

The current implementation focuses on dataset generation and baseline model training. A production-ready alert system has not yet been implemented.

**Current Stage:** Task 1 completed; Task 2 baseline model training and evaluation completed.

## 2. Project Objectives

- Generate a synthetic agricultural dataset.
- Validate the dataset before model training.
- Prepare numerical and categorical features for machine learning.
- Train a Random Forest regression model.
- Evaluate the model using five-fold cross-validation and a held-out test set.
- Analyze feature importance.
- Save the trained model and evaluation results.
- Prepare the project for future prediction API and alert system integration.

## 3. Task 1 — Synthetic Dataset Generation

### Objective

The first task focuses on generating a synthetic agricultural dataset containing field information and crop yield measurements. This dataset provides the input required for the subsequent machine learning task.

### Data Generation Script

The dataset generation script is located at:

`src/generate_data.py`

### Generated Dataset

The generated dataset is saved at:

`data/synthetic_crop_yield.csv`

The dataset contains **1,000 records and six columns**.

### Dataset Schema

| Column | Description | Role |
|---|---|---|
| `field_id` | Unique identifier for each field | Identifier; excluded from model features |
| `soil_type` | Soil category | Input feature |
| `rainfall` | Rainfall measurement | Input feature |
| `temperature` | Temperature measurement | Input feature |
| `fertilizer` | Fertilizer measurement | Input feature |
| `yield` | Crop yield target | Target variable |

### Generate the Dataset

Run the following command from the project root:

```powershell
python src/generate_data.py
```

This runs the data generation script and creates the dataset used in Task 2.

If the dataset already exists and you do not need to regenerate it, you can proceed directly to model training.

### Task 1 Output

- **Script:** `src/generate_data.py`
- **Dataset:** `data/synthetic_crop_yield.csv`
- **Records:** 1,000
- **Columns:** 6

### Dataset Limitation

The dataset contains synthetic data generated for development and experimentation. It may not fully represent real agricultural conditions. Therefore, the model's performance must not be interpreted as verified real-world farming accuracy.

## 4. Task 2 — Feature Engineering and Baseline Model Training

### Objective

Task 2 uses the dataset generated in Task 1 to build and evaluate a machine learning model for crop yield prediction.

### Training Script

The training script is located at:

`src/train_model.py`

### Model and Preprocessing Pipeline

The project uses a `RandomForestRegressor` with 200 trees.

The preprocessing and training pipeline includes:

- **Numerical features:** Rainfall, temperature, and fertilizer.
- **Missing numerical values:** Handled using median imputation.
- **Numerical scaling:** Applied using `StandardScaler`.
- **Categorical feature:** Soil type.
- **Missing categorical values:** Handled using most-frequent imputation.
- **Categorical encoding:** Applied using one-hot encoding.
- **Train/test split:** 80% training data and 20% test data.
- **Random seed:** 42 for reproducibility.
- **Cross-validation:** Five-fold cross-validation on the training split.

The trained pipeline includes both preprocessing and the regression model so that the same transformations can be applied during future predictions.

## 5. Technology Stack

- Python 3.10
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

## 6. Project Structure

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

## 7. Environment Setup

Python 3.10 or another compatible Python version is recommended.

### Create a Virtual Environment

Run the following command from the project root:

```powershell
python -m venv .venv
```

### Activate the Environment

In Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

## 8. Running the Project

Run the tasks in the following order when starting with a fresh dataset.

### Step 1: Generate the Dataset

```powershell
python src/generate_data.py
```

This runs Task 1 and generates the synthetic agricultural dataset.

### Step 2: Train and Evaluate the Model

```powershell
python src/train_model.py
```

This runs Task 2 and performs the following operations:

1. Loads and validates the dataset.
2. Separates the input features from the target variable.
3. Preprocesses numerical features.
4. Encodes categorical soil types.
5. Splits the dataset into training and testing sets.
6. Performs five-fold cross-validation on the training data.
7. Trains the Random Forest regression pipeline.
8. Evaluates the model on the held-out test set.
9. Generates a feature importance chart.
10. Saves the trained model and evaluation metrics.

## 9. Model Evaluation

The current baseline model was evaluated using five-fold cross-validation and a separate held-out test set.

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

### Evaluation Metrics

**Mean Absolute Error (MAE)** measures the average absolute difference between predicted and actual crop yield values. Lower values indicate smaller prediction errors.

**R² (Coefficient of Determination)** measures model performance relative to a mean-prediction baseline. Its interpretation depends on the dataset and evaluation setup.

These results describe the model's performance on the current synthetic dataset only.

## 10. Feature Importance

The model generates a feature importance chart at:

`reports/feature_importance.png`

The current model's leading feature importance values are:

| Feature | Importance |
|---|---:|
| Rainfall | 0.3665 |
| Loamy soil | 0.3016 |
| Temperature | 0.2220 |
| Fertilizer | 0.0747 |
| Clay soil | 0.0179 |

The chart provides a visual summary of the features used by the fitted Random Forest model.

Feature importance does not establish causal relationships between agricultural inputs and crop yield. The rankings may also change when the model is trained on different data.

## 11. Generated Artifacts

The project saves the following outputs:

| Artifact | Description |
|---|---|
| `data/synthetic_crop_yield.csv` | Synthetic agricultural dataset generated in Task 1 |
| `models/crop_yield_model.pkl` | Saved preprocessing and regression pipeline |
| `reports/model_metrics.json` | Evaluation metrics and feature importance values |
| `reports/feature_importance.png` | Feature importance visualization |

The saved model can be loaded for subsequent predictions using the same preprocessing steps applied during training.

**Security note:** Only load model files from trusted sources because loading untrusted pickle-based artifacts can execute malicious code.

## 12. Example Prediction

The saved model was tested using the following sample input:

| Feature | Value |
|---|---:|
| Soil type | Loamy |
| Rainfall | 500 |
| Temperature | 25 |
| Fertilizer | 100 |

**Predicted crop yield:** 6.862 tonnes/hectare.

This is an example prediction from the current synthetic-data model, not a verified estimate for a real farm.

## 13. Limitations and Future Improvements

- Replace or supplement synthetic data with reliable real-world agricultural datasets.
- Validate feature units and measurement ranges against the selected data source.
- Evaluate model performance across different soil types, weather conditions, and farming regions.
- Compare the baseline model with alternative regression algorithms.
- Improve prediction validation and model monitoring.
- Integrate the trained pipeline into a prediction API.
- Develop an alert system after defining and validating appropriate agricultural thresholds.
- Evaluate the system using real-world agricultural data before practical deployment.

## 14. Current Project Status

| Component | Status |
|---|---|
| Task 1 — Synthetic dataset generation | Completed |
| Dataset validation | Completed |
| Task 2 — Feature preprocessing | Completed |
| Random Forest baseline training | Completed |
| Five-fold cross-validation | Completed |
| Held-out test evaluation | Completed |
| Feature importance visualization | Completed |
| Model serialization | Completed |
| Sample prediction verification | Completed |
| Real-world agricultural validation | Future work |
| Prediction API integration | Future work |
| Agricultural alert system | Future work |

## 15. Conclusion

The project establishes an initial machine learning pipeline for crop yield forecasting. Task 1 generates the synthetic dataset, while Task 2 uses that dataset to preprocess features, train a Random Forest regression model, evaluate its performance, and save the resulting artifacts.

The current results demonstrate the workflow on synthetic data. Further validation using reliable real-world agricultural data is necessary before using the model for farming decisions or agricultural alerts.