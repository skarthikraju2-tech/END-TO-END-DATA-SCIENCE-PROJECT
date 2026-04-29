from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_squared_error

from src.pipeline import add_engineered_features, build_pipeline, collect_data, split_data


MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODEL_DIR / "housing_model.joblib"


def train_and_save_model():
    """Train the housing price model and save the pipeline."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = collect_data()
    df = add_engineered_features(df)
    X_train, X_test, y_train, y_test = split_data(df)

    feature_names = [column for column in X_train.columns]
    pipeline = build_pipeline(feature_names)
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")
    print(f"Test RMSE: ${rmse:,.2f}")


if __name__ == "__main__":
    train_and_save_model()
