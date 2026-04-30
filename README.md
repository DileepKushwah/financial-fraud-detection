# Financial Fraud Detection Deployment

This repository contains a FastAPI application for predicting fraudulent transactions using a trained XGBoost model.

## Features
- **FastAPI**: High-performance REST API.
- **XGBoost**: State-of-the-art fraud detection model.
- **Docker**: Containerized for easy deployment.
- **Railway Ready**: Configured for seamless deployment on Railway.app.

## Project Structure
- `main.py`: The FastAPI application logic.
- `fraud_detection_pipeline.pkl`: The trained model and preprocessing artifacts.
- `requirements.txt`: Python dependencies.
- `Dockerfile`: Container configuration.

## How to Run Locally

### 1. Set Up Virtual Environment
```bash
# Create venv
python -m venv venv

# Activate venv (Windows)
.\venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the API & Frontend
```bash
python main.py
```
The **Professional UI** will be available at `http://localhost:8000`. 
The **Interactive API Docs** are at `http://localhost:8000/docs`.

## User Interface Features
- **Glassmorphism Design**: Sleek and modern translucent cards.
- **Dynamic Risk Gauge**: Visualizes fraud probability in real-time.
- **Responsive Layout**: Optimized for both desktop and mobile devices.
- **FastAPI Backend**: Sub-millisecond inference performance.

## Deployment to Railway.app

### 1. Push to GitHub
1. Initialize a git repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```
2. Create a new repository on GitHub and push your code:
   ```bash
   git remote add origin <your-github-repo-url>
   git branch -M main
   git push -u origin main
   ```

### 2. Connect to Railway
1. Go to [Railway.app](https://railway.app/).
2. Create a **New Project**.
3. Select **Deploy from GitHub repo**.
4. Choose this repository.
5. Railway will automatically detect the `Dockerfile` and start the build and deployment process.

## API Usage

### Root Endpoint
- **URL**: `/`
- **Method**: `GET`
- **Response**: Health check and model info.

### Prediction Endpoint
- **URL**: `/predict`
- **Method**: `POST`
- **Payload Example**:
  ```json
  {
    "amt": 150.0,
    "dist_to_merch": 5.4,
    "time_since_last_txn": 3600,
    "txn_count_1hr": 1,
    "amt_z_score": 0.5,
    "amt_ratio_to_avg": 1.2,
    "user_txn_count": 50,
    "user_avg_amt": 100.0,
    "user_std_amt": 20.0,
    "mcc_enc": 5411.0
  }
  ```
- **Response Example**:
  ```json
  {
    "fraud_probability": 0.0234,
    "is_fraud": false,
    "decision": "✅ LEGIT",
    "threshold_used": 0.25
  }
  ```
