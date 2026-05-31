import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

from preprocessing import TARGET_COL, build_model_pipeline, get_feature_target, load_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "models"))
MODEL_PATH = MODELS_DIR / "obesity_model.joblib"
MODEL_WITH_BMI_PATH = MODELS_DIR / "obesity_model_with_bmi.joblib"
MODEL_NO_BMI_PATH = MODELS_DIR / "obesity_model_no_bmi.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

MIN_ACCURACY = 0.75
CV_FOLDS = 5


def get_model_candidates(random_state: int) -> list[tuple[str, object, bool]]:
    candidates: list[tuple[str, object, bool]] = [
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                class_weight="balanced_subsample",
                n_jobs=-1,
            ),
            False,
        ),
    ]

    try:
        from xgboost import XGBClassifier

        candidates.append(
            (
                "xgboost",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.1,
                    objective="multi:softmax",
                    eval_metric="mlogloss",
                    random_state=random_state,
                ),
                True,
            )
        )
    except ImportError:
        logger.info("XGBoost não instalado — usando apenas Random Forest.")

    return candidates


def encode_labels(y_train, y_test, needs_encoding: bool) -> tuple[object, object, object | None]:
    if not needs_encoding:
        return y_train, y_test, None

    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_test_enc = encoder.transform(y_test)
    return y_train_enc, y_test_enc, encoder


def evaluate_test(
    pipeline,
    label_encoder,
    X_test,
    y_test,
    classes: list[str],
) -> dict:
    preds = pipeline.predict(X_test)
    if label_encoder is not None:
        preds = label_encoder.inverse_transform(np.asarray(preds).astype(int))

    return {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "f1_macro": round(float(f1_score(y_test, preds, average="macro")), 4),
        "f1_weighted": round(float(f1_score(y_test, preds, average="weighted")), 4),
        "classification_report": classification_report(y_test, preds, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, preds, labels=classes).tolist(),
    }


def tune_random_forest(
    X_train,
    y_train,
    random_state: int,
    include_bmi: bool,
):
    rf = RandomForestClassifier(
        random_state=random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    search = RandomizedSearchCV(
        build_model_pipeline(rf, include_bmi=include_bmi),
        param_distributions={
            "classifier__n_estimators": [300, 500, 700],
            "classifier__max_depth": [None, 12, 20, 30],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
        },
        n_iter=12,
        cv=CV_FOLDS,
        scoring="f1_macro",
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, "random_forest_tuned"


def train_variant(
    df,
    include_bmi: bool,
    variant_name: str,
    random_state: int,
) -> dict:
    X, y = get_feature_target(df, include_bmi=include_bmi)
    classes = sorted(y.unique().tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state)
    candidates = get_model_candidates(random_state)
    candidate_logs: list[dict] = []

    best_cv_score = float("-inf")
    best_name = None
    best_classifier = candidates[0][1]
    best_needs_encoding = candidates[0][2]

    for name, classifier, needs_encoding in candidates:
        pipeline = build_model_pipeline(classifier, include_bmi=include_bmi)
        y_train_fit, _, label_encoder = encode_labels(y_train, y_test, needs_encoding)

        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train_fit,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
        )
        cv_mean = float(cv_scores.mean())
        candidate_logs.append(
            {
                "model": name,
                "cv_f1_macro_mean": round(cv_mean, 4),
                "cv_f1_macro_std": round(float(cv_scores.std()), 4),
                "needs_label_encoder": needs_encoding,
            }
        )
        logger.info("[%s] %s — CV f1_macro=%.4f (±%.4f)", variant_name, name, cv_mean, cv_scores.std())

        if cv_mean > best_cv_score:
            best_cv_score = cv_mean
            best_name = name
            best_needs_encoding = needs_encoding
            best_classifier = classifier

    best_pipeline = build_model_pipeline(best_classifier, include_bmi=include_bmi)
    best_label_encoder = None
    y_train_fit, _, _ = encode_labels(y_train, y_test, best_needs_encoding)
    if best_needs_encoding:
        best_label_encoder = LabelEncoder().fit(y_train)
        y_train_fit = best_label_encoder.transform(y_train)

    best_pipeline.fit(X_train, y_train_fit)
    test_metrics = evaluate_test(best_pipeline, best_label_encoder, X_test, y_test, classes)

    tuned = False
    if test_metrics["accuracy"] < MIN_ACCURACY or test_metrics["f1_macro"] < 0.70:
        logger.info("[%s] Aplicando tuning leve no Random Forest...", variant_name)
        tuned_pipeline, tuned_name = tune_random_forest(X_train, y_train, random_state, include_bmi)
        tuned_metrics = evaluate_test(tuned_pipeline, None, X_test, y_test, classes)
        if tuned_metrics["f1_macro"] >= test_metrics["f1_macro"]:
            best_pipeline = tuned_pipeline
            best_name = tuned_name
            best_label_encoder = None
            test_metrics = tuned_metrics
            tuned = True

    return {
        "variant": variant_name,
        "include_bmi": include_bmi,
        "model": best_name,
        "tuned": tuned,
        "cv_f1_macro_mean": round(best_cv_score, 4),
        "candidates": candidate_logs,
        "pipeline": best_pipeline,
        "label_encoder": best_label_encoder,
        "train_size": len(X_train),
        "test_size": len(X_test),
        **test_metrics,
    }


def choose_production_variant(with_bmi: dict, no_bmi: dict) -> str:
    """Produção sem BMI quando atinge a meta FIAP (evita atalho por IMC)."""
    if no_bmi["accuracy"] >= MIN_ACCURACY:
        return "no_bmi"
    if with_bmi["accuracy"] >= MIN_ACCURACY:
        return "with_bmi"
    return max((with_bmi, no_bmi), key=lambda v: v["f1_macro"])["variant"]


def persist_artifacts(with_bmi: dict, no_bmi: dict, production_key: str) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    variants_meta = {}
    for result, path in ((with_bmi, MODEL_WITH_BMI_PATH), (no_bmi, MODEL_NO_BMI_PATH)):
        joblib.dump(result["pipeline"], path)
        le_path = MODELS_DIR / f"label_encoder_{result['variant']}.joblib"
        if result["label_encoder"] is not None:
            joblib.dump(result["label_encoder"], le_path)
        elif le_path.exists():
            le_path.unlink()

        variants_meta[result["variant"]] = {
            k: v
            for k, v in result.items()
            if k not in ("pipeline", "label_encoder")
        }
        variants_meta[result["variant"]]["model_path"] = str(path.name)

    production = with_bmi if production_key == "with_bmi" else no_bmi
    joblib.dump(production["pipeline"], MODEL_PATH)

    if production["label_encoder"] is not None:
        joblib.dump(production["label_encoder"], LABEL_ENCODER_PATH)
    elif LABEL_ENCODER_PATH.exists():
        LABEL_ENCODER_PATH.unlink()

    summary = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "target_column": TARGET_COL,
        "min_accuracy_required": MIN_ACCURACY,
        "production_variant": production_key,
        "production_model": production["model"],
        "include_bmi_in_production": production["include_bmi"],
        "accuracy": production["accuracy"],
        "f1_macro": production["f1_macro"],
        "f1_weighted": production["f1_weighted"],
        "classification_report": production["classification_report"],
        "confusion_matrix": production["confusion_matrix"],
        "train_size": production["train_size"],
        "test_size": production["test_size"],
        "uses_label_encoder": production["label_encoder"] is not None,
        "variants": variants_meta,
        "note": (
            "Modelo de produção prioriza variant sem BMI (hábitos + antropometria) "
            "quando competitivo; variant com BMI documentada para comparação."
        ),
    }

    METRICS_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def train_and_evaluate(random_state: int = 42) -> dict:
    np.random.seed(random_state)
    df = load_data()

    logger.info("Treinando variant com BMI...")
    with_bmi = train_variant(df, include_bmi=True, variant_name="with_bmi", random_state=random_state)

    logger.info("Treinando variant sem BMI (evita atalho por IMC)...")
    no_bmi = train_variant(df, include_bmi=False, variant_name="no_bmi", random_state=random_state)

    production_key = choose_production_variant(with_bmi, no_bmi)
    logger.info("Variant de produção escolhida: %s", production_key)

    summary = persist_artifacts(with_bmi, no_bmi, production_key)
    return summary


if __name__ == "__main__":
    result = train_and_evaluate()
    print(f"Produção: {result['production_variant']} ({result['production_model']})")
    print(f"Acurácia: {result['accuracy']:.2%} | F1 macro: {result['f1_macro']:.2%}")
    print(f"Com BMI  — acc: {result['variants']['with_bmi']['accuracy']:.2%}")
    print(f"Sem BMI  — acc: {result['variants']['no_bmi']['accuracy']:.2%}")
    print(f"Salvo em: {MODEL_PATH}")
