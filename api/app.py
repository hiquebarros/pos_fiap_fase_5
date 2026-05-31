import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
from flask import Flask, jsonify, request

from preprocessing import prepare_patient_input

app = Flask(__name__)
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/model_data"))


@lru_cache(maxsize=1)
def load_artifacts():
    pipeline = joblib.load(MODEL_DIR / "obesity_model.joblib")
    label_encoder = None
    label_encoder_path = MODEL_DIR / "label_encoder.joblib"
    if label_encoder_path.exists():
        label_encoder = joblib.load(label_encoder_path)

    metrics = {}
    metrics_path = MODEL_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    include_bmi = metrics.get("include_bmi_in_production", True)

    return pipeline, label_encoder, metrics, include_bmi


def decode_prediction(prediction, label_encoder):
    if label_encoder is not None:
        return label_encoder.inverse_transform(prediction.astype(int))[0]
    return prediction[0]


def get_class_labels(label_encoder, pipeline, n_classes: int):
    if label_encoder is not None:
        return label_encoder.classes_.tolist()
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "classes_"):
        return classifier.classes_.tolist()
    return [f"class_{i}" for i in range(n_classes)]


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/metrics")
def metrics():
    *_, metrics_data, _ = load_artifacts()
    return jsonify(metrics_data)


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "JSON body required"}), 400

    try:
        pipeline, label_encoder, _, include_bmi = load_artifacts()
        features = prepare_patient_input(payload, include_bmi=include_bmi)
        bmi = round(float(payload["Weight"]) / (float(payload["Height"]) ** 2), 2)
        prediction = decode_prediction(pipeline.predict(features), label_encoder)
        probabilities = pipeline.predict_proba(features)[0]
        classes = get_class_labels(label_encoder, pipeline, len(probabilities))

        return jsonify(
            {
                "prediction": prediction,
                "bmi": bmi,
                "probabilities": [
                    {"class": cls, "probability": round(float(prob), 4)}
                    for cls, prob in zip(classes, probabilities)
                ],
            }
        )
    except FileNotFoundError:
        return jsonify({"error": "Model not found. Run trainer first."}), 503
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
