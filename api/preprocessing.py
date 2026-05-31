import os
from pathlib import Path

import pandas as pd

TARGET_COL = "Obesity"
DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "obesity.csv"
DATA_PATH = Path(os.environ.get("DATA_PATH", DEFAULT_DATA))

NUMERIC_WITH_BMI = ["Age", "Height", "Weight", "BMI", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
NUMERIC_NO_BMI = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
CATEGORICAL_FEATURES = ["Gender", "family_history", "FAVC", "SMOKE", "SCC", "MTRANS"]
ORDINAL_FEATURES = ["CAEC", "CALC"]

CAEC_ORDER = ["no", "Sometimes", "Frequently", "Always"]
CALC_ORDER = ["no", "Sometimes", "Frequently", "Always"]

ROUND_COLS = {
    "FCVC": (1, 3),
    "NCP": (1, 4),
    "CH2O": (1, 3),
    "FAF": (0, 3),
    "TUE": (0, 2),
}


def numeric_features(include_bmi: bool = True) -> list[str]:
    return NUMERIC_WITH_BMI if include_bmi else NUMERIC_NO_BMI


def feature_columns(include_bmi: bool = True) -> list[str]:
    return numeric_features(include_bmi) + CATEGORICAL_FEATURES + ORDINAL_FEATURES


def load_data(path: Path | str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or DATA_PATH)
    return clean_data(df)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    for col, (low, high) in ROUND_COLS.items():
        data[col] = data[col].round().clip(low, high)

    data["BMI"] = data["Weight"] / (data["Height"] ** 2)

    return data


def prepare_patient_input(payload: dict, include_bmi: bool = True) -> pd.DataFrame:
    row = payload.copy()
    height = float(row["Height"])
    weight = float(row["Weight"])
    row["BMI"] = weight / (height**2)

    for col, (low, high) in ROUND_COLS.items():
        row[col] = int(round(float(row[col])))
        row[col] = max(low, min(high, row[col]))

    cols = feature_columns(include_bmi)
    return pd.DataFrame([{col: row[col] for col in cols}])
