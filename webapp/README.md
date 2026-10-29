# 🌱 Crop Yield Forecasting Dashboard

## Overview

The Crop Yield Forecasting Dashboard is an interactive web application built with Streamlit. It allows users to enter agricultural field information and obtain crop yield predictions from a deployed FastAPI machine learning service.

The dashboard connects to the existing Crop Yield Forecasting API and displays predicted yield in tonnes per hectare.

## Features

- **Interactive Input Form:** Enter soil type, rainfall, temperature, and fertilizer values.
- **Machine Learning Predictions:** Send input features to the deployed FastAPI prediction endpoint.
- **Prediction Results:** Display the predicted crop yield in tonnes per hectare.
- **Historical Predictions:** Maintain a history of successful predictions during the current Streamlit session.
- **Interactive Chart:** Visualize historical yield predictions using a bar chart.
- **Prediction History Table:** Review previous predictions, input values, and timestamps.
- **API Health Check:** Check whether the API is responding.
- **Error Handling:** Display messages for connection failures, timeouts, and unsuccessful API responses.

## Technologies Used

- Python
- Streamlit
- Requests
- Pandas
- FastAPI
- Scikit-learn
- Render
- Git and GitHub

## Project Structure

```text
webapp/
├── app.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10 or a compatible Python version
- Internet connection
- Access to the deployed FastAPI service

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ahsannazir001/crop-yield-forecasting.git
```

### 2. Navigate to the Dashboard Directory

```bash
cd crop-yield-forecasting/webapp
```

### 3. Create and Activate a Virtual Environment (Optional)

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the Dashboard Locally

Execute:

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open this address in your browser to use the dashboard.

## How to Use

1. Open the dashboard.
2. Select the soil type.
3. Enter the rainfall value.
4. Enter the temperature in degrees Celsius.
5. Enter the fertilizer value.
6. Click **Predict Crop Yield**.
7. View the predicted crop yield in tonnes per hectare.
8. Generate additional predictions to populate the historical chart and prediction history table.

The application sends the input values to the FastAPI `/predict` endpoint and displays the returned prediction.

## API Configuration

The dashboard uses the following deployed API by default:

**Live API:**  
https://crop-yield-forecasting-0j5q.onrender.com/

**API Documentation:**  
https://crop-yield-forecasting-0j5q.onrender.com/docs

**Prediction Endpoint:**  
https://crop-yield-forecasting-0j5q.onrender.com/predict

The API base URL can also be configured from the dashboard sidebar.

## Sample Input

| Feature | Example Value |
|---|---:|
| Soil Type | loamy |
| Rainfall | 500 |
| Temperature | 25 |
| Fertilizer | 100 |

For this example, the previously tested API returned a predicted yield of approximately **6.8623 tonnes per hectare**. Actual results depend on the deployed model and submitted input values.

## Historical Predictions

The dashboard stores successful predictions in Streamlit session state and displays them in a bar chart and table.

**Limitation:** Prediction history is session-based. It is not stored in a permanent database and may be lost when the session resets.

## Deployment

The dashboard is designed for deployment on Streamlit Community Cloud.

Deployment configuration:

- **Repository:** `ahsannazir001/crop-yield-forecasting`
- **Branch:** `main`
- **Main file path:** `webapp/app.py`

The repository must contain the dashboard code and its `requirements.txt` file before deployment.

## Important Note

The machine learning model was trained using a synthetically generated dataset. Predictions are intended for demonstration and educational purposes and should not be treated as verified real-world agricultural forecasts.

## Author

**Ahsan Nazir**

Crop Yield Forecasting and Alert System for GreenField Farmers Co-op