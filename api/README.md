# Crop Yield Forecasting API

FastAPI service for predicting crop yield using a trained Random Forest regression pipeline.

## Features

- Loads the trained model from `models/crop_yield_model.pkl`.
- Accepts JSON input through the `/predict` endpoint.
- Validates incoming field features.
- Returns predicted crop yield in tonnes per hectare.
- Provides health and interactive documentation endpoints.

## Input Features

| Feature | Type | Validation |
|---|---|---|
| soil_type | String | Must not be empty |
| rainfall | Number | Must be non-negative |
| temperature | Number | Must be finite |
| fertilizer | Number | Must be non-negative |

## Endpoints

- `GET /` — API information
- `GET /health` — Health check
- `POST /predict` — Predict crop yield
- `GET /docs` — Interactive API documentation

## Run Locally

From the repository root:

```bash
python -m pip install -r api/requirements.txt
python -m uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` to test the API.

## Example Request

```json
{
  "soil_type": "loamy",
  "rainfall": 500,
  "temperature": 25,
  "fertilizer": 100
}
```

## Example Response

```json
{
  "yield": 6.862
}
```

The response value above is illustrative. The actual prediction depends on the trained model.

**Note:** The model was trained using a synthetic dataset. Predictions should not be treated as verified agricultural advice.