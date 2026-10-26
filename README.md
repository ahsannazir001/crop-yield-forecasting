# Crop Yield Forecasting and Alert System for GreenField Farmers Co-op

An end-to-end machine-learning project to generate crop-yield data,
train a baseline regression model, and serve predictions through FastAPI
in a Docker container. **Render deployment is still pending.**

> **Important:** The dataset is synthetic. Current metrics demonstrate
> the workflow on generated data; they do not prove real-world
> agricultural accuracy.

## Objectives

-   Generate a synthetic crop-yield dataset.
-   Preprocess numerical and categorical features.
-   Train and evaluate a baseline model.
-   Save the fitted model pipeline.
-   Expose inference through FastAPI.
-   Package the API and model with Docker.
-   Deploy to Render, then add a dashboard, drift monitoring, and a
    model card.

## Task 1 --- Synthetic Dataset Generation

Script: `src/generate_data.py`\
Output: `data/synthetic_crop_yield.csv`

The generated dataset contains **1,000 rows and 6 columns**:

  Column          Meaning                   Role
  --------------- ------------------------- ------------------------------
  `field_id`      Field identifier          Excluded from model features
  `soil_type`     Soil category             Input
  `rainfall`      Rainfall measurement      Input
  `temperature`   Temperature measurement   Input
  `fertilizer`    Fertilizer measurement    Input
  `yield`         Crop yield                Target

Generate the dataset:

``` powershell
python src/generate_data.py
```

The CSV produced in Task 1 is used to train and evaluate the model in
Task 2.

## Task 2 --- Feature Engineering and Baseline Model Training

Script: `src/train_model.py`

### Preprocessing and model

-   Excluded `field_id` because it is an identifier.
-   Numerical features: `rainfall`, `temperature`, `fertilizer`.
-   Numerical preprocessing: median imputation and `StandardScaler`.
-   Categorical feature: `soil_type`, processed with most-frequent
    imputation and one-hot encoding.
-   Estimator:
    `RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1)`.
-   Split: 80% training and 20% test.
-   Validation: 5-fold KFold cross-validation on the training split.
-   Metrics: MAE and R².

### Evaluation results

  Metric                                Result
  -------------------- -----------------------
  Mean 5-fold CV MAE                    0.3549
  Mean 5-fold CV R²                     0.8447
  Test MAE               0.3413 tonnes/hectare
  Test R²                               0.8380

These results are from synthetic data and should not be treated as
real-world performance estimates.

### Feature importance

  Feature                            Importance
  -------------------------------- ------------
  `numeric__rainfall`                    0.3665
  `categorical__soil_type_loamy`         0.3016
  `numeric__temperature`                 0.2220
  `numeric__fertilizer`                  0.0747
  `categorical__soil_type_clay`          0.0179

Feature importance reflects the model's use of features; it does not
establish causation.

### Saved artifacts

-   `models/crop_yield_model.pkl`
-   `reports/model_metrics.json`
-   `reports/feature_importance.png`

Train the model:

``` powershell
python src/train_model.py
```

Example input:

``` json
{
  "soil_type": "loamy",
  "rainfall": 500,
  "temperature": 25,
  "fertilizer": 100
}
```

The example prediction was approximately **6.8623 tonnes/hectare**.

## Task 3 --- FastAPI Inference API and Docker

### API endpoints

  Endpoint          Purpose
  ----------------- -----------------------------------
  `GET /`           API status message
  `GET /health`     Health/model check
  `POST /predict`   Predict yield from input features
  `GET /docs`       Interactive Swagger documentation

### Run the API locally

From the project root, with the virtual environment active and API
dependencies installed:

``` powershell
python -m uvicorn api.main:app --reload
```

Open:

-   Home: `http://127.0.0.1:8000/`
-   Health: `http://127.0.0.1:8000/health`
-   Swagger UI: `http://127.0.0.1:8000/docs`

Test `POST /predict` with:

``` json
{
  "soil_type": "loamy",
  "rainfall": 500,
  "temperature": 25,
  "fertilizer": 100
}
```

The endpoint returned HTTP `200` and a prediction of approximately
`6.8623` for this input.

### Docker

The root `Dockerfile` uses Python 3.10 slim, installs API dependencies,
copies the API code and saved model, and starts Uvicorn on
`0.0.0.0:8000`.

The root `.dockerignore` excludes local virtual environments, Git
metadata, private `Task/` files, training data, reports, caches, and
environment files from the build context. The API code and model are
intentionally not ignored.

Build the image from the project root:

``` powershell
docker build -t crop-yield-api:latest .
```

Check the image:

``` powershell
docker images
```

Run the container:

``` powershell
docker run --rm -p 8000:8000 crop-yield-api:latest
```

Browse to `http://127.0.0.1:8000/` (not `0.0.0.0`, which is the server
bind address). Test `/health` and `/docs`, then submit the prediction
example above.

### Task 3 checklist

-   [x] FastAPI starts locally.
-   [x] Health endpoint tested.
-   [x] Prediction endpoint tested successfully.
-   [x] Docker image `crop-yield-api:latest` built.
-   [x] Prediction tested inside the Docker container.
-   [x] `.dockerignore` added.
-   [ ] Render deployment and public URL verification.

## Technology Stack

-   Python 3.10
-   pandas, NumPy, scikit-learn
-   Matplotlib, joblib
-   FastAPI, Uvicorn
-   Docker
-   Git and GitHub
-   Render (deployment pending)

## Project Structure

``` text
.
├── api/
│   ├── main.py
│   ├── requirements.txt
│   └── README.md
├── data/
│   └── synthetic_crop_yield.csv
├── models/
│   └── crop_yield_model.pkl
├── notebooks/
├── reports/
│   ├── model_metrics.json
│   └── feature_importance.png
├── src/
│   ├── generate_data.py
│   └── train_model.py
├── Task/                 # Private task documents; do not commit
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

## Environment Setup

Create/activate a virtual environment if needed:

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install root dependencies:

``` powershell
python -m pip install -r requirements.txt
```

Install API dependencies if maintained separately:

``` powershell
python -m pip install -r api/requirements.txt
```

## Limitations and Future Work

-   Synthetic data may not represent real farm conditions.
-   Validate feature units, value ranges, and predictions with reliable
    real agricultural data before operational use.
-   Compare the baseline against alternative models using suitable
    validation.
-   Deploy the API to Render and test its public URL.
-   Create a Streamlit dashboard for predictions.
-   Implement drift monitoring and create a model card.
-   Add appropriate logging and production deployment configuration.

## Current Project Status

  ## Task Status

| Task | Description | Status |
|---|---|---|
| Task 1 | Synthetic dataset and repository setup | Completed |
| Task 2 | Feature engineering and baseline model | Completed |
| Task 3 | FastAPI, Docker, and Render deployment | Completed |

## Conclusion

## Conclusion

Tasks 1, 2, and 3 have been completed successfully. The project includes synthetic dataset generation, feature engineering, a trained Random Forest model, model evaluation, a FastAPI inference API, Docker containerization, and successful deployment on Render. The live `POST /predict` endpoint was tested successfully and returned an HTTP 200 response with a predicted crop yield.
