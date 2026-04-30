import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="GuardianAI | Fraud Detection",
    description="API to predict fraudulent transactions using an XGBoost model.",
    version="1.0.0"
)

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model artifacts
MODEL_PATH = "fraud_detection_pipeline.pkl"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file {MODEL_PATH} not found!")

try:
    artifacts = joblib.load(MODEL_PATH)
    model = artifacts['model']
    scaler = artifacts['scaler']
    features = artifacts['features']
    threshold = artifacts['best_threshold']
    # Load label encoder for MCC
    le = artifacts.get('label_encoder')
    print("Model and artifacts loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load model artifacts: {e}")

# Define the request body schema
class Transaction(BaseModel):
    amt: float
    dist_to_merch: float
    time_since_last_txn: float
    txn_count_1hr: int
    amt_z_score: float
    amt_ratio_to_avg: float
    user_txn_count: int
    user_avg_amt: float
    user_std_amt: float
    mcc_enc: float = None  # Optional if raw mcc is provided
    mcc: str = None        # Optional raw MCC string (e.g., "5411")

# Serve Static Files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse('static/index.html')

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        data_dict = transaction.model_dump()
        
        # Handle MCC Encoding
        if data_dict.get('mcc') is not None and le is not None:
            try:
                # The encoder was trained on strings
                mcc_val = str(data_dict['mcc'])
                if mcc_val in le.classes_:
                    data_dict['mcc_enc'] = float(le.transform([mcc_val])[0])
                else:
                    # Default to 0 or average if unknown
                    data_dict['mcc_enc'] = 0.0
            except:
                data_dict['mcc_enc'] = data_dict.get('mcc_enc', 0.0)
        
        if data_dict.get('mcc_enc') is None:
            data_dict['mcc_enc'] = 0.0

        # Convert input to DataFrame
        input_data = pd.DataFrame([data_dict])
        
        # Ensure feature order matches training
        input_data = input_data[features]
        
        # Scale the data
        input_scaled = scaler.transform(input_data)
        
        # Get probability
        prob = model.predict_proba(input_scaled)[0][1]
        
        # Determine flag based on threshold
        fraud_flag = int(prob >= threshold)
        
        return {
            "fraud_probability": round(float(prob), 4),
            "is_fraud": bool(fraud_flag),
            "decision": "🚨 FRAUD" if fraud_flag else "✅ LEGIT",
            "threshold_used": threshold,
            "mcc_encoded_used": data_dict['mcc_enc']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # Railway sets the PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
