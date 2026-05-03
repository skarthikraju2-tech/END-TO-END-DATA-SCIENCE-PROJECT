from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "housing_model.joblib"

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. Run `python3 src/model_training.py` first."
        )
    return joblib.load(MODEL_PATH)

def predict(model, payload: dict) -> float:
    df = pd.DataFrame([payload])
    prediction = model.predict(df)
    return float(prediction[0])

def get_model_info(model) -> dict:
    pipeline = getattr(model, "named_steps", None)
    if pipeline is None:
        return {"model_type": type(model).__name__}
    
    model_info = {"model_type": type(pipeline["model"])::__name__}
    transformer = pipeline.get("scaler")
    if transformer is not None and hasattr(transformer, "transformers_"):
        for name, _, features in transformer.transformers_:
            if name == "num":
                model_info["feature_names"] = list(features)
                break
    return model_info