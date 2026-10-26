
from pathlib import Path
import json

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import (
    KFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

RANDOM_SEED = 42
TEST_SIZE = 0.20
N_SPLITS = 5

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "synthetic_crop_yield.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = MODEL_DIR / "crop_yield_model.pkl"
METRICS_PATH = REPORT_DIR / "model_metrics.json"
IMPORTANCE_PATH = REPORT_DIR / "feature_importance.png"


# --------------------------------------------------
# 2. Load and validate dataset
# --------------------------------------------------

def load_dataset() -> pd.DataFrame:
    """Load the CSV and validate its required columns."""

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Generate it first using: python src/generate_data.py"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {
        "field_id",
        "soil_type",
        "rainfall",
        "temperature",
        "fertilizer",
        "yield",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    if len(df) < 50:
        raise ValueError("Dataset is too small for the configured evaluation.")

    if df["field_id"].isna().any():
        raise ValueError("field_id contains missing values.")

    if df["field_id"].duplicated().any():
        raise ValueError("Duplicate field_id values detected.")

    if df["soil_type"].isna().any():
        raise ValueError("soil_type contains missing values.")

    numeric_columns = ["rainfall", "temperature", "fertilizer", "yield"]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

        if not np.isfinite(df[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} contains non-finite values.")

    if (df["yield"] <= 0).any():
        raise ValueError("Yield must be positive.")

    if (df["rainfall"] < 0).any():
        raise ValueError("Rainfall cannot be negative.")

    if (df["fertilizer"] < 0).any():
        raise ValueError("Fertilizer cannot be negative.")

    return df


# --------------------------------------------------
# 3. Feature engineering and preprocessing
# --------------------------------------------------

def build_pipeline() -> Pipeline:
    """
    Build preprocessing and regression steps.

    Categorical soil types are one-hot encoded.
    Numeric features receive median imputation and scaling.
    Scaling is not required by Random Forest, but keeps the
    preprocessing pipeline reusable for other estimators.
    """

    numeric_features = [
        "rainfall",
        "temperature",
        "fertilizer",
    ]

    categorical_features = ["soil_type"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", model),
        ]
    )


# --------------------------------------------------
# 4. Feature importance plot
# --------------------------------------------------

def save_feature_importance(
    fitted_pipeline: Pipeline,
) -> list[dict[str, float | str]]:
    """Save feature importances and return them as records."""

    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    regressor = fitted_pipeline.named_steps["regressor"]

    feature_names = preprocessor.get_feature_names_out()
    importances = regressor.feature_importances_

    order = np.argsort(importances)

    plt.figure(figsize=(10, 6))
    plt.barh(
        feature_names[order],
        importances[order],
    )
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(IMPORTANCE_PATH, dpi=150)
    plt.close()

    return [
        {
            "feature": str(feature_names[index]),
            "importance": float(importances[index]),
        }
        for index in np.argsort(importances)[::-1]
    ]


# --------------------------------------------------
# 5. Train, evaluate, and save
# --------------------------------------------------

def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()

    print("=" * 60)
    print("CROP YIELD FORECASTING - BASELINE MODEL")
    print("=" * 60)

    print(f"Dataset rows: {len(df)}")
    print(f"Dataset columns: {len(df.columns)}")
    print(f"Dataset path: {DATA_PATH}")

    # field_id identifies a record; it is not a predictive feature.
    X = df[["soil_type", "rainfall", "temperature", "fertilizer"]]
    y = df["yield"]

    # Keep the final test set separate from cross-validation.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )

    pipeline = build_pipeline()

    # Cross-validation uses only training data.
    cv = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        },
        n_jobs=1,
        error_score="raise",
    )

    cv_mae_scores = -cv_results["test_mae"]
    cv_r2_scores = cv_results["test_r2"]

    print("\nCross-validation results:")
    for fold_number, (mae, r2) in enumerate(
        zip(cv_mae_scores, cv_r2_scores),
        start=1,
    ):
        print(f"Fold {fold_number}: MAE={mae:.4f}, R2={r2:.4f}")

    print(f"Mean CV MAE: {cv_mae_scores.mean():.4f}")
    print(f"Mean CV R2: {cv_r2_scores.mean():.4f}")

    # Fit on the complete training split.
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    test_mae = mean_absolute_error(y_test, predictions)
    test_r2 = r2_score(y_test, predictions)

    print("\nHeld-out test results:")
    print(f"Test MAE: {test_mae:.4f} tonnes/hectare")
    print(f"Test R2: {test_r2:.4f}")

    feature_importance = save_feature_importance(pipeline)

    # Save the complete fitted pipeline, including preprocessing.
    joblib.dump(pipeline, MODEL_PATH)

    metrics = {
        "model": "RandomForestRegressor",
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "cross_validation_folds": N_SPLITS,
        "cross_validation_mae": [
            float(score) for score in cv_mae_scores
        ],
        "mean_cross_validation_mae": float(cv_mae_scores.mean()),
        "cross_validation_r2": [
            float(score) for score in cv_r2_scores
        ],
        "mean_cross_validation_r2": float(cv_r2_scores.mean()),
        "test_mae": float(test_mae),
        "test_r2": float(test_r2),
        "target_unit": "tonnes per hectare",
        "dataset_type": "synthetic",
        "feature_importance": feature_importance,
    }

    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\nSaved artifacts:")
    print(f"Model: {MODEL_PATH}")
    print(f"Metrics: {METRICS_PATH}")
    print(f"Feature importance plot: {IMPORTANCE_PATH}")

    print("\nTop features:")
    for item in feature_importance[:5]:
        print(
            f"{item['feature']}: {item['importance']:.4f}"
        )

    print("\nTraining and evaluation completed successfully.")


if __name__ == "__main__":
    main()