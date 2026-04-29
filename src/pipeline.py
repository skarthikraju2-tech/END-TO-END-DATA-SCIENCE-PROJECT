from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data/raw"
PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data/processed"


def collect_data(save_path: Path = RAW_DATA_DIR / "california_housing.csv") -> pd.DataFrame:
    """Collect the California Housing dataset and persist raw CSV data."""
    dataset = fetch_california_housing(as_frame=True)
    df = dataset.frame.copy()
    df["MedHouseVal"] = df["MedHouseVal"] * 100000

    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived housing features for model training."""
    df = df.copy()
    df["rooms_per_household"] = df["AveRooms"] / df["HouseAge"].replace(0, 1)
    df["bedrooms_per_room"] = df["AveBedrms"] / df["AveRooms"].replace(0, 1)
    df["population_per_household"] = df["Population"] / df["HouseAge"].replace(0, 1)
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split the dataset into training and test sets."""
    y = df["MedHouseVal"]
    X = df.drop(columns=["MedHouseVal"])
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def build_pipeline(feature_names):
    """Build a preprocessing + model pipeline."""
    transformer = ColumnTransformer(
        transformers=[("num", StandardScaler(), feature_names)],
        remainder="drop",
    )

    pipeline = Pipeline(
        [
            ("scaler", transformer),
            ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
        ]
    )
    return pipeline
