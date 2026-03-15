from fastapi import FastAPI
import joblib
import pandas as pd

from NetworkPacket import NetworkPacket

app = FastAPI(title="Trojan Detection IDS", version="1.0")

print("[INFO] Loading ML models into server memory...")
rf_model = joblib.load("../preparing_data/ml_training/trained_models/random_forest.joblib")
lgbm_model = joblib.load("../preparing_data/ml_training/trained_models/lightgbm.joblib")
print("[INFO] Models loaded successfully! Server is ready.")

@app.post("/analyze")
def analyze_packet(packet: NetworkPacket):
    """
    Analyzes network traffic features to detect potential Trojan activity.

    This endpoint receives a payload containing the 15 preprocessed network
    and host features. It processes the data through the trained Random Forest
    and LightGBM models to calculate the probability of the packet being malicious.

    Args:
        packet (NetworkPacket): The data payload containing the extracted features.

    Returns:
        dict: A JSON response containing the Trojan probability estimated
              by both models.
    """
    df = pd.DataFrame([packet.features])

    rf_prob = rf_model.predict_proba(df)[0][1]
    lgbm_prob = lgbm_model.predict_proba(df)[0][1]

    return {
        "status": "success",
        "results": {
            "random_forest": {
                "trojan_probability": round(rf_prob, 4)
            },
            "lightgbm": {
                "trojan_probability": round(lgbm_prob, 4)
            }
        }
    }