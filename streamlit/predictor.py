import json
import os
from functools import lru_cache
from pathlib import Path

import joblib

from preprocessing import prepare_patient_input

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR))


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


def get_metrics_local() -> dict:
    *_, metrics, _ = load_artifacts()
    return metrics


def _class_labels(label_encoder, pipeline, n_classes: int) -> list:
    if label_encoder is not None:
        return label_encoder.classes_.tolist()
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "classes_"):
        return classifier.classes_.tolist()
    return [f"class_{i}" for i in range(n_classes)]


def predict_local(payload: dict) -> dict:
    pipeline, label_encoder, _, include_bmi = load_artifacts()
    features = prepare_patient_input(payload, include_bmi=include_bmi)
    bmi = round(float(payload["Weight"]) / (float(payload["Height"]) ** 2), 2)

    raw_prediction = pipeline.predict(features)
    if label_encoder is not None:
        prediction = label_encoder.inverse_transform(raw_prediction.astype(int))[0]
    else:
        prediction = raw_prediction[0]

    probabilities = pipeline.predict_proba(features)[0]
    classes = _class_labels(label_encoder, pipeline, len(probabilities))

    return {
        "prediction": prediction,
        "bmi": bmi,
        "probabilities": [
            {"class": cls, "probability": round(float(prob), 4)}
            for cls, prob in zip(classes, probabilities)
        ],
    }
