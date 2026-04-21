"""
FastAPI Backend — Trojan Detection IDS.

Loads all five trained ML models (Random Forest, LightGBM, XGBoost,
Logistic Regression, MLP) and exposes a single POST endpoint that
returns the Trojan probability from every model for a given set of
network-traffic features.
"""

from fastapi import FastAPI
import joblib
import pandas as pd

from NetworkPacket import NetworkPacket

app = FastAPI(title="Trojan Detection IDS", version="2.0")

# ── Feature names (must match the training column order) ──────────
FEATURE_NAMES = [
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Total Fwd Packets", "Total Backward Packets",
    "Fwd Packets Length Total", "Bwd Packets Length Total",
    "Packet Length Min", "Packet Length Max", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance",
    "Flow Bytes/s", "Flow Packets/s",
]

# ── Load all trained models at server startup ─────────────────────
MODEL_DIR = "../preparing_data/ml_training/trained_models"

print("[INFO] Loading ML models into server memory...")
rf_model   = joblib.load(f"{MODEL_DIR}/random_forest.joblib")
lgbm_model = joblib.load(f"{MODEL_DIR}/lightgbm.joblib")
xgb_model  = joblib.load(f"{MODEL_DIR}/xgboost.joblib")
lr_model   = joblib.load(f"{MODEL_DIR}/logistic_regression.joblib")
mlp_model  = joblib.load(f"{MODEL_DIR}/mlp.joblib")
print("[INFO] All 5 models loaded successfully! Server is ready.")


@app.post("/analyze")
def analyze_packet(packet: NetworkPacket):
    """
    Analyzes network traffic features to detect potential Trojan activity.

    Receives a payload containing 15 preprocessed network features and
    passes them through all five trained classifiers. Returns a JSON
    response with the Trojan probability estimated by each model.

    Args:
        packet (NetworkPacket): The data payload with extracted features.

    Returns:
        dict: JSON containing Trojan probabilities from all five models.
    """
    df = pd.DataFrame([packet.features], columns=FEATURE_NAMES)

    # Get Trojan probability (class 1) from each model.
    # Cast to native Python float so FastAPI's JSON encoder can serialise them
    # (XGBoost returns numpy.float32 which is not JSON-serialisable).
    rf_prob   = float(rf_model.predict_proba(df)[0][1])
    lgbm_prob = float(lgbm_model.predict_proba(df)[0][1])
    xgb_prob  = float(xgb_model.predict_proba(df)[0][1])
    lr_prob   = float(lr_model.predict_proba(df)[0][1])
    mlp_prob  = float(mlp_model.predict_proba(df)[0][1])

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
            "logistic_regression": {
                "trojan_probability": round(lr_prob, 4)
            },
            "mlp": {
                "trojan_probability": round(mlp_prob, 4)
            },
        },
    }