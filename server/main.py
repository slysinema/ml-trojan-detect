from fastapi import FastAPI
import joblib
import pandas as pd

from NetworkPacket import NetworkPacket

app = FastAPI(title="Trojan Detection IDS", version="1.0")

print("[INFO] Загрузка моделей в память сервера...")
rf_model = joblib.load("../preparing_data/ml_training/trained_models/random_forest.joblib")
lgbm_model = joblib.load("../preparing_data/ml_training/trained_models/lightgbm.joblib")
print("[INFO] Модели успешно загружены! Сервер готов к работе.")

@app.post("/analyze")
def analyze_packet(packet: NetworkPacket):
    """
    Принимает 15 признаков сетевого пакета и возвращает вердикт от двух моделей.
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