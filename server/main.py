from fastapi import FastAPI
import joblib
import pandas as pd

from NetworkPacket import NetworkPacket

app = FastAPI(title="Trojan Detection IDS", version="2.0")

FEATURE_NAMES = [
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Total Fwd Packets", "Total Backward Packets",
    "Fwd Packets Length Total", "Bwd Packets Length Total",
    "Packet Length Min", "Packet Length Max", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance",
    "Flow Bytes/s", "Flow Packets/s"
]

print("[INFO] Loading ML models into server memory...")
rf_model = joblib.load("../preparing_data/ml_training/trained_models/random_forest.joblib")
lgbm_model = joblib.load("../preparing_data/ml_training/trained_models/lightgbm.joblib")
xgb_model = joblib.load("../preparing_data/ml_training/trained_models/xgboost.joblib")
et_model = joblib.load("../preparing_data/ml_training/trained_models/extra_trees.joblib")
knn_model = joblib.load("../preparing_data/ml_training/trained_models/knn.joblib")
print("[INFO] All 5 models loaded successfully! Server is ready.")


@app.post("/analyze")
def analyze_packet(packet: NetworkPacket):
    """
    Analyzes network traffic features to detect potential Trojan activity.

    This endpoint receives a payload containing the 15 preprocessed network
    features. It processes the data through all five trained ML models
    to calculate the probability of the packet being malicious.

    Args:
        packet (NetworkPacket): The data payload containing the extracted features.

    Returns:
        dict: A JSON response containing the Trojan probability estimated
              by all five models.
    """
    df = pd.DataFrame([packet.features], columns=FEATURE_NAMES)

    rf_prob = rf_model.predict_proba(df)[0][1]
    lgbm_prob = lgbm_model.predict_proba(df)[0][1]
    xgb_prob = xgb_model.predict_proba(df)[0][1]
    et_prob = et_model.predict_proba(df)[0][1]
    knn_prob = knn_model.predict_proba(df)[0][1]

    return {
        "status": "success",
        "results": {
            "random_forest": {
                "trojan_probability": round(rf_prob, 4)
            },
            "lightgbm": {
                "trojan_probability": round(lgbm_prob, 4)
            },
            "xgboost": {
                "trojan_probability": round(xgb_prob, 4)
            },
            "extra_trees": {
                "trojan_probability": round(et_prob, 4)
            },
            "knn": {
                "trojan_probability": round(knn_prob, 4)
            }
        }
    }