# Crop Yield Forecasting and Alert System

## Project Overview

This project is being developed for GreenField Farmers Co-op to demonstrate an end-to-end machine learning pipeline for crop yield forecasting.

**Current milestone:** Task 1 — Synthetic Dataset Generation and Repository Setup.

The current dataset is synthetic and intended for development and experimentation. It does not represent actual farm measurements.

## Dataset

The generated dataset is saved at:

`data/synthetic_crop_yield.csv`

It contains 1,000 synthetic field records and the following columns:

| Column | Description |
|---|---|
| `field_id` | Unique identifier for each field |
| `soil_type` | Soil category: loamy, sandy, or clay |
| `rainfall` | Synthetic rainfall amount in millimetres |
| `temperature` | Synthetic temperature in degrees Celsius |
| `fertilizer` | Synthetic fertilizer quantity in the generator's illustrative units |
| `yield` | Simulated crop yield in tonnes per hectare |

The yield values are generated using illustrative mathematical relationships involving soil type, rainfall, temperature, fertilizer, and random variation. They are not validated agricultural estimates.

## Technology Stack

- Python
- NumPy
- Pandas
- Git and GitHub

## Project Structure

```text
crop-yield-forecasting/
├── data/
│   └── synthetic_crop_yield.csv
├── notebooks/
├── src/
│   └── generate_data.py
├── .gitignore
└── README.md
```

## Setup

Python 3.10 or newer is recommended for this project.

Create and activate a virtual environment on Windows:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

Install the dependencies:

```bat
python -m pip install numpy pandas
```

## Generate the Dataset

From the project root, run:

```bat
python src/generate_data.py
```

The script uses a fixed random seed so the same configuration produces reproducible data. It validates basic dataset properties and writes the CSV file to `data/synthetic_crop_yield.csv`.

## Verify the Dataset

Run:

```bat
python -c "import pandas as pd; df=pd.read_csv('data/synthetic_crop_yield.csv'); print(df.shape); print(df.head())"
```

The expected shape is `(1000, 6)`.

## Limitations

This dataset is artificially generated. Model performance on this data cannot establish real-world agricultural accuracy. Real farm data and domain validation are required before using predictions for farming decisions.

## Project Roadmap

1. Generate synthetic crop yield data and set up the repository.
2. Perform feature engineering and train a baseline model.
3. Build and deploy a FastAPI inference service.
4. Create a Streamlit prediction dashboard.
5. Implement drift monitoring and document the model.
