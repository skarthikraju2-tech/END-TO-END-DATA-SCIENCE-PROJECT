# End-to-End Data Science Project

This repository contains a complete end-to-end data science pipeline for predicting California housing prices using the built-in `sklearn` California Housing dataset.

## Project overview

- **Data collection**: collect and persist the California Housing dataset in CSV form
- **Preprocessing**: engineer additional features and scale numeric inputs
- **Model training**: train a `RandomForestRegressor` pipeline and evaluate its performance
- **Deployment**: expose prediction capabilities via a FastAPI REST API

## Structure

- `src/pipeline.py` - data collection, feature engineering, preprocessing pipeline setup
- `src/model_training.py` - training and model serialization script
- `app/main.py` - FastAPI application exposing the prediction endpoint
- `app/model.py` - model loader and prediction helper
- `requirements.txt` - Python dependencies
- `models/` - trained model artifact location
- `data/` - raw and processed dataset storage

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the model

Run the training module to collect data, preprocess it, train the model, and save the artifact:

```bash
python3 -m src.model_training
```

## Run the API

Start the FastAPI server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open the web UI in your browser:

- http://127.0.0.1:8000/

The dashboard now includes model metadata and API usage details.

## Model metadata endpoint

- `GET /model-info` - returns the model type and feature list used by the pipeline

## Example API request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"MedInc":8.3252,"HouseAge":41.0,"AveRooms":6.9841,"AveBedrms":1.0238,"Population":322.0,"AveOccup":2.5556,"Latitude":37.88,"Longitude":-122.23}'
```

## API endpoints

- `GET /` - health check
- `POST /predict` - predict median house price for a sample of California housing features

## Notes

This project is designed to be reproducible locally with minimal setup and no external dataset downloads beyond `sklearn`.
