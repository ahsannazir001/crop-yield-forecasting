# Crop Yield Forecasting and Alert System for GreenField Farmers Co-op

An end-to-end machine learning project that generates a synthetic crop-yield dataset, trains a Random Forest regression model, serves predictions through FastAPI, and provides an interactive Streamlit dashboard for crop-yield forecasting.

The inference API is containerized with Docker and deployed on Render. The Streamlit dashboard provides an interface for entering agricultural field features, viewing predictions, and exploring prediction history.

> **Important:** The dataset is synthetic. The reported metrics demonstrate the workflow on generated data; they do not establish real-world agricultural accuracy.

## Project Objectives

- Generate a synthetic crop-yield dataset.
- Preprocess numerical and categorical features.
- Train and evaluate a machine learning regression model.
- Save the fitted preprocessing and model pipeline.
- Expose predictions through FastAPI.
- Package the API with Docker.
- Deploy the inference API on Render.
- Build a Streamlit dashboard for predictions.
- Maintain a session-based history of predictions and visualize them.
- Prepare for future drift monitoring and model-card documentation.

## Project Links

- **GitHub Repository:** [https://github.com/ahsannazir001/crop-yield-forecasting](https://github.com/ahsannazir001/crop-yield-forecasting)
- **Live FastAPI:** [https://crop-yield-forecasting-0j5q.onrender.com/](https://crop-yield-forecasting-0j5q.onrender.com/)
- **API Documentation (Swagger):** [https://crop-yield-forecasting-0j5q.onrender.com/docs](https://crop-yield-forecasting-0j5q.onrender.com/docs)
- **Prediction Endpoint:** [https://crop-yield-forecasting-0j5q.onrender.com/predict](https://crop-yield-forecasting-0j5q.onrender.com/predict)
- **Streamlit Dashboard:** Add the public URL after deployment.

## Task 1 — Synthetic Dataset Generation

**Script:** `src/generate_data.py`

**Output:** `data/synthetic_crop_yield.csv`

The generated dataset contains **1,000 rows and 6 columns**.

| Column        | Meaning                 | Role                         |
| ------------- | ----------------------- | ---------------------------- |
| `field_id`    | Field identifier        | Excluded from model features |
| `soil_type`   | Soil category           | Input feature                |
| `rainfall`    | Rainfall measurement    | Input feature                |
| `temperature` | Temperature measurement | Input feature                |
| `fertilizer`  | Fertilizer measurement  | Input feature                |
| `yield`       | Crop yield              | Target variable              |

### Generate the Dataset

From the project root, run:

```powershell
python src/generate_data.py
```

The generated CSV is used to train and evaluate the machine learning model in Task 2.

## Task 2 — Feature Engineering and Model Training

**Training script:** `src/train_model.py`

### Input Features

The model uses four predictive features:

- `soil_type`
- `rainfall`
- `temperature`
- `fertilizer`

The `field_id` column is excluded because it is an identifier rather than a predictive feature.

The target variable is `yield`, representing crop yield in tonnes per hectare.

### Data Preprocessing

A Scikit-learn preprocessing pipeline handles numerical and categorical features separately.

**Numerical features:**

- Rainfall
- Temperature
- Fertilizer

Numerical preprocessing uses:

- `SimpleImputer(strategy="median")` for missing values.
- `StandardScaler` for feature scaling.

**Categorical feature:**

- Soil type

Categorical preprocessing uses:

- `SimpleImputer(strategy="most_frequent")` for missing values.
- `OneHotEncoder` to convert soil categories into numerical features.

The preprocessing steps are integrated into the model pipeline to ensure consistent transformations during training and inference.

### Train-Test Split and Validation

- Training data: **80% (800 records)**
- Testing data: **20% (200 records)**
- Validation: **5-fold K-Fold cross-validation** on the training split.
- Evaluation metrics: Mean Absolute Error (MAE) and R².

### Machine Learning Algorithm

The project uses `RandomForestRegressor` with the following configuration:

```python
RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
```

The trained pipeline is saved using Joblib for reuse by the prediction API.

### Model Evaluation Results

| Metric             |                Result |
| ------------------ | --------------------: |
| Mean 5-fold CV MAE |                0.3549 |
| Mean 5-fold CV R²  |                0.8447 |
| Test MAE           | 0.3413 tonnes/hectare |
| Test R²            |                0.8380 |

These results were obtained using synthetic data and should not be interpreted as real-world performance estimates.

### Feature Importance

| Feature                        | Importance |
| ------------------------------ | ---------: |
| `numeric__rainfall`            |     0.3665 |
| `categorical__soil_type_loamy` |     0.3016 |
| `numeric__temperature`         |     0.2220 |
| `numeric__fertilizer`          |     0.0747 |
| `categorical__soil_type_clay`  |     0.0179 |

Feature importance reflects how the trained model uses features; it does not establish causation.

### Saved Artifacts

- `models/crop_yield_model.pkl`
- `reports/model_metrics.json`
- `reports/feature_importance.png`

### Train the Model

```powershell
python src/train_model.py
```

## Task 3 — FastAPI Inference API and Docker

The trained machine learning pipeline is exposed through a FastAPI application.

### API Endpoints

| Endpoint        | Purpose                                |
| --------------- | -------------------------------------- |
| `GET /`         | API status message                     |
| `GET /health`   | Health/model check                     |
| `POST /predict` | Predict crop yield from input features |
| `GET /docs`     | Interactive Swagger documentation      |

### Live API

**API Home:**
[https://crop-yield-forecasting-0j5q.onrender.com/](https://crop-yield-forecasting-0j5q.onrender.com/)

**Health Check:**
[https://crop-yield-forecasting-0j5q.onrender.com/health](https://crop-yield-forecasting-0j5q.onrender.com/health)

**Swagger Documentation:**
[https://crop-yield-forecasting-0j5q.onrender.com/docs](https://crop-yield-forecasting-0j5q.onrender.com/docs)

**Prediction Endpoint:**
[https://crop-yield-forecasting-0j5q.onrender.com/predict](https://crop-yield-forecasting-0j5q.onrender.com/predict)

### Sample Prediction Input

Send the following JSON payload to `POST /predict`:

```json
{
  "soil_type": "loamy",
  "rainfall": 500,
  "temperature": 25,
  "fertilizer": 100
}
```

The previously tested endpoint returned HTTP `200` and a predicted yield of approximately **6.8623 tonnes/hectare** for this input.

### Run the API Locally

From the project root, activate your virtual environment and install the required dependencies.

Run:

```powershell
python -m uvicorn api.main:app --reload
```

Open the following local URLs:

- Home: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- Health: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Docker

The root `Dockerfile` uses Python 3.10 slim, installs the API dependencies, copies the API code and saved model, and starts Uvicorn on `0.0.0.0:8000`.

The root `.dockerignore` excludes local virtual environments, Git metadata, private task files, training data, reports, caches, and environment files from the Docker build context.

Build the Docker image:

```powershell
docker build -t crop-yield-api:latest .
```

Check the image:

```powershell
docker images
```

Run the container:

```powershell
docker run --rm -p 8000:8000 crop-yield-api:latest
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) to access the local API.

### Task 3 Checklist

- [x] FastAPI application implemented.
- [x] Health endpoint tested.
- [x] Prediction endpoint tested successfully.
- [x] Docker image built.
- [x] Prediction tested inside the Docker container.
- [x] `.dockerignore` added.
- [x] API deployed to Render.
- [x] Live prediction endpoint tested successfully.

## Task 4 — Streamlit Dashboard for Predictions

The Streamlit dashboard provides a user-friendly interface for interacting with the deployed machine learning API.

### Dashboard Features

- Interactive inputs for soil type, rainfall, temperature, and fertilizer.
- Integration with the deployed FastAPI `/predict` endpoint.
- Predicted crop yield displayed in tonnes per hectare.
- Historical prediction bar chart.
- Prediction history table containing input features, predicted yield, and timestamp.
- API connection check.
- Error handling for unsuccessful requests and connection failures.

### Dashboard Technology Stack

- Python
- Streamlit
- Requests
- Pandas
- FastAPI
- Scikit-learn
- Git and GitHub
- Streamlit Community Cloud for deployment

### Dashboard Structure

```text
webapp/
├── app.py
├── requirements.txt
└── README.md
```

### Run the Dashboard Locally

From the project root, run:

```powershell
python -m pip install -r webapp/requirements.txt
python -m streamlit run webapp/app.py
```

Alternatively, navigate to the `webapp` directory and run:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The dashboard normally opens at:

[http://localhost:8501](http://localhost:8501)

### Dashboard API Configuration

The dashboard uses the deployed FastAPI service by default:

`https://crop-yield-forecasting-0j5q.onrender.com`

The API base URL can be configured from the dashboard sidebar.

### Historical Prediction Storage

Successful predictions are stored in Streamlit session state and displayed in a chart and table.

**Limitation:** Prediction history is session-based. It is not stored in a permanent database and may be lost when the session resets.

### Streamlit Deployment

The dashboard is intended for deployment on Streamlit Community Cloud.

**Live Streamlit Dashboard:** https://greenfield-yield-forecast.streamlit.app/

Detailed dashboard installation and usage instructions are available in `webapp/README.md`.

### Task 4 Checklist

- [x] Streamlit dashboard implemented.
- [x] Field input form implemented.
- [x] FastAPI prediction endpoint integrated.
- [x] Prediction results displayed.
- [x] Historical predictions chart implemented.
- [x] Prediction history table implemented.
- [x] Dashboard deployed to Streamlit Community Cloud.
- [x] Public dashboard URL verified.

## Technology Stack

| Technology                | Purpose                            |
| ------------------------- | ---------------------------------- |
| Python 3.10               | Main programming language          |
| Pandas                    | Data handling                      |
| NumPy                     | Numerical operations               |
| Scikit-learn              | Preprocessing and model training   |
| Random Forest Regression  | Crop-yield prediction              |
| SimpleImputer             | Missing-value handling             |
| OneHotEncoder             | Categorical encoding               |
| StandardScaler            | Numerical feature scaling          |
| Matplotlib                | Feature-importance visualization   |
| Joblib                    | Model persistence                  |
| FastAPI                   | Prediction API                     |
| Uvicorn                   | API server                         |
| Requests                  | HTTP communication with the API    |
| Streamlit                 | Interactive prediction dashboard   |
| Docker                    | API containerization               |
| Render                    | API hosting                        |
| Git and GitHub            | Version control and source hosting |
| Streamlit Community Cloud | Planned dashboard hosting          |

## Project Structure

```text
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
├── webapp/
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
├── Task/
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

The `Task/` directory contains private task documents and should not be committed if it contains confidential material.

## Environment Setup

Create and activate a virtual environment if needed:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the root dependencies:

```powershell
python -m pip install -r requirements.txt
```

Install the API dependencies if maintained separately:

```powershell
python -m pip install -r api/requirements.txt
```

Install the dashboard dependencies:

```powershell
python -m pip install -r webapp/requirements.txt
```

## Limitations and Future Work

- The dataset is synthetic and may not represent real farm conditions.
- Predictions require validation against reliable real-world agricultural data before operational use.
- Feature units and acceptable value ranges should be verified against agricultural domain requirements.
- Alternative models should be evaluated using suitable validation methods.
- Historical dashboard predictions are session-based rather than persistently stored.
- Drift monitoring and a model card remain future tasks.
- Production use would benefit from additional logging, monitoring, and deployment configuration.

## Current Project Status

| Task   | Description                                          | Status                                    |
| ------ | ---------------------------------------------------- | ----------------------------------------- |
| Task 1 | Synthetic dataset generation and repository setup    | Completed                                 |
| Task 2 | Feature engineering and baseline model training      | Completed                                 |
| Task 3 | FastAPI inference API, Docker, and Render deployment | Completed                                 |
| Task 4 | Streamlit dashboard for predictions                  | Dashboard implemented; deployment pending |
| Task 5 | Drift monitoring and model card                      | Not started                               |

## Conclusion

Tasks 1, 2, and 3 have been completed. The project includes synthetic dataset generation, preprocessing, Random Forest model training and evaluation, a FastAPI inference API, Docker containerization, and a successfully tested Render deployment.

Task 4 adds an interactive Streamlit dashboard that accepts agricultural field inputs, calls the deployed API, displays predicted crop yield, and visualizes historical predictions during the current session. Public Streamlit deployment remains pending.

The dataset is synthetic, so the model's reported metrics should not be interpreted as evidence of real-world agricultural accuracy.
