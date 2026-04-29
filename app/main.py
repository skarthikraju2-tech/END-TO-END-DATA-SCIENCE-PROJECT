from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.model import get_model_info, load_model, predict

app = FastAPI(
    title="California Housing Price Predictor",
    description="Predict median house values for California housing data using a trained regression pipeline.",
    version="1.0.0",
)

model = None
model_info = {}


class HousingFeatures(BaseModel):
    MedInc: float = Field(..., description="Median income in block group")
    HouseAge: float = Field(..., description="Median house age in block group")
    AveRooms: float = Field(..., description="Average rooms per household")
    AveBedrms: float = Field(..., description="Average bedrooms per household")
    Population: float = Field(..., description="Block group population")
    AveOccup: float = Field(..., description="Average occupants per household")
    Latitude: float = Field(..., description="Block group latitude")
    Longitude: float = Field(..., description="Block group longitude")


@app.on_event("startup")
def startup_event():
    global model, model_info
    model = load_model()
    model_info = get_model_info(model)


@app.get("/model-info")
def model_info_endpoint():
    if not model_info:
        raise HTTPException(status_code=503, detail="Model metadata is not available")
    return model_info


@app.get("/", response_class=HTMLResponse)
def read_root():
    return """<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>California Housing Price Predictor</title>
    <style>
        :root { color-scheme: light; }
        body { font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 0; background: #f5f7fb; color: #111827; }
        .container { max-width: 1000px; margin: 0 auto; padding: 2rem; }
        .card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 1rem; padding: 1.75rem; box-shadow: 0 10px 25px rgba(15, 23, 42, 0.05); margin-bottom: 1.5rem; }
        h1, h2, h3 { margin: 0 0 1rem; }
        h1 { font-size: 2.2rem; }
        .grid { display: grid; gap: 1.25rem; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }
        label { display: block; margin-bottom: 0.5rem; font-size: 0.95rem; font-weight: 600; }
        input { width: 100%; padding: 0.85rem 1rem; border: 1px solid #d1d5db; border-radius: 0.75rem; background: #f9fafb; font-size: 1rem; }
        button { width: 100%; padding: 0.95rem 1.2rem; border: none; border-radius: 0.85rem; background: #2563eb; color: #ffffff; font-weight: 700; cursor: pointer; transition: background 0.2s ease; }
        button:hover { background: #1d4ed8; }
        .result { margin-top: 1rem; padding: 1rem 1.25rem; border-radius: 0.85rem; background: #eef2ff; border: 1px solid #c7d2fe; font-weight: 600; }
        .meta-list { list-style: none; padding: 0; margin: 0; }
        .meta-list li { padding: 0.45rem 0; border-bottom: 1px solid #e5e7eb; }
        .meta-list li:last-child { border-bottom: none; }
        .hint { color: #6b7280; margin: 0 0 1rem; }
        .section-header { display: flex; justify-content: space-between; align-items: center; gap: 1rem; }
    </style>
</head>
<body>
    <div class=\"container\">
        <div class=\"card\">
            <div class=\"section-header\"><div><h1>California Housing Price Predictor</h1><p class=\"hint\">Interactive model dashboard backed by a trained FastAPI regression pipeline.</p></div></div>
            <p>Use the form below to score a new California housing observation, then review the model metadata and prediction output.</p>
        </div>

        <div class=\"grid\">
            <div class=\"card\">
                <h2>Prediction form</h2>
                <form id=\"prediction-form\">
                    <label for=\"MedInc\">Median Income</label>
                    <input type=\"number\" id=\"MedInc\" name=\"MedInc\" step=\"any\" value=\"8.3252\" required>
                    <label for=\"HouseAge\">House Age</label>
                    <input type=\"number\" id=\"HouseAge\" name=\"HouseAge\" step=\"any\" value=\"41.0\" required>
                    <label for=\"AveRooms\">Average Rooms</label>
                    <input type=\"number\" id=\"AveRooms\" name=\"AveRooms\" step=\"any\" value=\"6.9841\" required>
                    <label for=\"AveBedrms\">Average Bedrooms</label>
                    <input type=\"number\" id=\"AveBedrms\" name=\"AveBedrms\" step=\"any\" value=\"1.0238\" required>
                    <label for=\"Population\">Population</label>
                    <input type=\"number\" id=\"Population\" name=\"Population\" step=\"any\" value=\"322.0\" required>
                    <label for=\"AveOccup\">Average Occupants</label>
                    <input type=\"number\" id=\"AveOccup\" name=\"AveOccup\" step=\"any\" value=\"2.5556\" required>
                    <label for=\"Latitude\">Latitude</label>
                    <input type=\"number\" id=\"Latitude\" name=\"Latitude\" step=\"any\" value=\"37.88\" required>
                    <label for=\"Longitude\">Longitude</label>
                    <input type=\"number\" id=\"Longitude\" name=\"Longitude\" step=\"any\" value=\"-122.23\" required>
                    <button type=\"submit\">Predict</button>
                </form>
                <div class=\"result\" id=\"result\">Prediction result will appear here.</div>
            </div>

            <div class=\"card\">
                <h2>Model dashboard</h2>
                <p class=\"hint\">This page fetches metadata from the API and displays the current model configuration.</p>
                <ul class=\"meta-list\" id=\"model-details\">
                    <li>Loading model details…</li>
                </ul>
                <h3>API endpoints</h3>
                <ul class=\"meta-list\">
                    <li><strong>GET /</strong> - dashboard page</li>
                    <li><strong>POST /predict</strong> - score a feature vector</li>
                    <li><strong>GET /model-info</strong> - model metadata</li>
                    <li><strong>GET /health</strong> - API health status</li>
                </ul>
            </div>
        </div>
    </div>

    <script>
        const form = document.getElementById('prediction-form');
        const resultEl = document.getElementById('result');
        const modelDetails = document.getElementById('model-details');

        async function loadModelInfo() {
            try {
                const response = await fetch('/model-info');
                const metadata = await response.json();
                if (!response.ok) throw new Error(metadata.detail || 'Unable to load model details');

                const items = [];
                items.push(`<li><strong>Model type:</strong> ${metadata.model_type}</li>`);
                items.push(`<li><strong>Feature count:</strong> ${metadata.feature_names?.length ?? 'unknown'}</li>`);
                if (metadata.feature_names) {
                    items.push(`<li><strong>Features:</strong> ${metadata.feature_names.join(', ')}</li>`);
                }
                modelDetails.innerHTML = items.join('');
            } catch (error) {
                modelDetails.innerHTML = `<li>Error loading model info: ${error.message}</li>`;
            }
        }

        form.addEventListener('submit', async (event) => {
            event.preventDefault();
            const data = Object.fromEntries(new FormData(form).entries());
            Object.keys(data).forEach(key => data[key] = parseFloat(data[key]));

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data),
                });
                const payload = await response.json();
                if (!response.ok) throw new Error(payload.detail || 'Prediction failed');
                resultEl.textContent = 'Predicted median house value: $' + payload.predicted_median_house_value.toFixed(2);
            } catch (error) {
                resultEl.textContent = 'Error: ' + error.message;
            }
        });

        loadModelInfo();
    </script>
</body>
</html>"""


@app.get("/health")
def health_check():
    return {"message": "California Housing Price Predictor API", "status": "running"}


@app.post("/predict")
def predict_house_value(features: HousingFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    result = predict(model, features.dict())
    return {"predicted_median_house_value": result}
