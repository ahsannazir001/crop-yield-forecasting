
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "crop_yield_model.pkl"

# API application
app = FastAPI(
    title="Crop Yield Forecasting API",
    description=(
        "Predict crop yield using soil type, rainfall, "
        "temperature, and fertilizer inputs."
    ),
    version="1.0.0",
)


class CropFeatures(BaseModel):
    """Input schema for crop yield prediction."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    soil_type: str = Field(min_length=1, max_length=50)
    rainfall: float = Field(ge=0)
    temperature: float
    fertilizer: float = Field(ge=0)


class PredictionResponse(BaseModel):
    """Output schema for crop yield prediction."""

    yield_value: float = Field(
        serialization_alias="yield",
        description="Predicted yield in tonnes per hectare",
    )


# Load the trained pipeline once when the application starts.
if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return {
        "message": "Crop Yield Forecasting API is running",
        "docs": "/docs",
        "prediction_endpoint": "/predict",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CropFeatures):
    """Predict crop yield from validated field features."""

    try:
        input_data = pd.DataFrame(
            [
                {
                    "soil_type": features.soil_type.strip(),
                    "rainfall": features.rainfall,
                    "temperature": features.temperature,
                    "fertilizer": features.fertilizer,
                }
            ]
        )

        prediction = float(model.predict(input_data)[0])

        return PredictionResponse(yield_value=prediction)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the API logs.",
        ) from exc
