
from pathlib import Path

import numpy as np
import pandas as pd


# Reproducible results: same seed produces the same dataset.
RANDOM_SEED = 42
N_ROWS = 1000

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "synthetic_crop_yield.csv"


def generate_dataset(n_rows: int = N_ROWS) -> pd.DataFrame:
    """Generate a synthetic crop-yield dataset for demonstration."""

    if n_rows < 500:
        raise ValueError("The dataset must contain at least 500 rows.")

    rng = np.random.default_rng(RANDOM_SEED)

    soil_types = np.array(["loamy", "sandy", "clay"])
    soil_type = rng.choice(
        soil_types,
        size=n_rows,
        p=[0.45, 0.30, 0.25],
    )

    rainfall = rng.uniform(100, 900, size=n_rows)
    temperature = rng.uniform(10, 40, size=n_rows)
    fertilizer = rng.uniform(20, 200, size=n_rows)

    # Illustrative crop-yield relationships, not real farm measurements.
    soil_effect = np.select(
        [
            soil_type == "loamy",
            soil_type == "sandy",
            soil_type == "clay",
        ],
        [1.0, -0.5, 0.2],
        default=0.0,
    )

    rainfall_effect = 2.0 * np.exp(
        -((rainfall - 500) / 250) ** 2
    )
    temperature_effect = 1.5 * np.exp(
        -((temperature - 25) / 8) ** 2
    )
    fertilizer_effect = (
        0.8 * (fertilizer / 100)
        - 0.15 * (fertilizer / 100) ** 2
    )

    noise = rng.normal(loc=0, scale=0.35, size=n_rows)

    yield_tonnes_per_hectare = (
        2.0
        + soil_effect
        + rainfall_effect
        + temperature_effect
        + fertilizer_effect
        + noise
    )

    yield_tonnes_per_hectare = np.clip(
        yield_tonnes_per_hectare,
        0.1,
        None,
    )

    return pd.DataFrame(
        {
            "field_id": [
                f"FIELD_{i:04d}" for i in range(1, n_rows + 1)
            ],
            "soil_type": soil_type,
            "rainfall": np.round(rainfall, 2),
            "temperature": np.round(temperature, 2),
            "fertilizer": np.round(fertilizer, 2),
            "yield": np.round(yield_tonnes_per_hectare, 3),
        }
    )


def main() -> None:
    """Generate, validate, and save the CSV dataset."""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = generate_dataset()

    assert len(df) >= 500
    assert df["field_id"].is_unique
    assert not df.isnull().any().any()
    assert (df["yield"] > 0).all()

    df.to_csv(OUTPUT_PATH, index=False)

    print("Dataset generated successfully!")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    print("\nMissing values:")
    print(df.isnull().sum().to_string())
    print("\nDataset summary:")
    print(df.describe(include="all").to_string())


if __name__ == "__main__":
    main()
