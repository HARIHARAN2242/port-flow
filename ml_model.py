"""
PortFlow ML layer.
Trains on a public historical port-dwell dataset when available.

Important limitation:
The public benchmark covers five US port complexes, not Tamil Nadu ports.
The model is a real supervised ML model, but predictions for Chennai/Ennore/
Kattupalli are out-of-domain estimates and must not be presented as locally
validated operational forecasts.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = Path(os.getenv("PORTFLOW_DATA_PATH", BASE_DIR / "monthly_dwell_2015_2025.csv"))
MODEL_PATH = BASE_DIR / "portflow_dwell_model.joblib"
DATA_URL = "https://zenodo.org/records/21203605/files/monthly_dwell_2015_2025.csv?download=1"

TARGET = "MedianDwellDays"
NUMERIC_FEATURES = ["UniqueVessels", "TotalObservations", "AvgVesselLength", "MonthNum"]
CATEGORICAL_FEATURES = ["Port"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = [
    "Port", "YearMonth", "UniqueVessels", "MedianDwellDays",
    "TotalObservations", "AvgVesselLength"
]

def load_public_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the Zenodo CSV from disk, or try downloading it once."""
    path = Path(path or DATA_PATH)
    if path.exists():
        df = pd.read_csv(path)
    else:
        try:
            df = pd.read_csv(DATA_URL)
            df.to_csv(path, index=False)
        except Exception as exc:
            raise FileNotFoundError(
                f"Could not find {path.name} and could not download the public dataset. "
                "Download monthly_dwell_2015_2025.csv from the Zenodo record and place "
                f"it beside app.py. Details: {exc}"
            ) from exc

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df.copy()
    df["YearMonth"] = pd.to_datetime(df["YearMonth"], errors="coerce")
    for c in ["UniqueVessels", "MedianDwellDays", "TotalObservations", "AvgVesselLength"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["MonthNum"] = df["YearMonth"].dt.month
    df = df.dropna(subset=FEATURES + [TARGET])
    df = df[(df["UniqueVessels"] >= 0) & (df["MedianDwellDays"] >= 0)]
    if len(df) < 30:
        raise ValueError(f"Only {len(df)} valid records found; at least 30 are required.")
    return df

def train_model(data_path: str | Path | None = None) -> dict[str, Any]:
    """Chronological holdout evaluation, then fit final model on all valid rows."""
    df = load_public_dataset(data_path).sort_values("YearMonth").reset_index(drop=True)
    split = max(int(len(df) * 0.8), 1)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    if len(test_df) < 5:
        raise ValueError("Not enough held-out rows for evaluation.")

    def make_pipeline():
        prep = ColumnTransformer(
            transformers=[
                ("num", "passthrough", NUMERIC_FEATURES),
                ("port", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ]
        )
        return Pipeline([
            ("prep", prep),
            ("model", RandomForestRegressor(
                n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
            )),
        ])

    evaluation_model = make_pipeline()
    evaluation_model.fit(train_df[FEATURES], train_df[TARGET])
    y_true = test_df[TARGET].to_numpy()
    y_pred = evaluation_model.predict(test_df[FEATURES])
    metrics = {
        "rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "mae_days": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "rmse_days": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "r2": round(float(r2_score(y_true, y_pred)), 3) if len(test_df) >= 2 else None,
        "dataset": "Zenodo historical monthly port-dwell dataset (US ports, 2015–2025)",
    }

    final_model = make_pipeline()
    final_model.fit(df[FEATURES], df[TARGET])
    bundle = {
        "model": final_model,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "ports": sorted(df["Port"].astype(str).unique().tolist()),
        "metrics": metrics,
        "training_rows": len(df),
        "year_month_min": str(df["YearMonth"].min().date()),
        "year_month_max": str(df["YearMonth"].max().date()),
        "dataset_path": str(Path(data_path or DATA_PATH)),
        "target": TARGET,
        "target_description": "Median vessel dwell time in port, in days",
    }
    joblib.dump(bundle, MODEL_PATH)
    return metrics

def load_model_bundle() -> dict[str, Any] | None:
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None

def get_dataset_status() -> dict[str, Any]:
    bundle = load_model_bundle()
    if bundle:
        return {"ready": True, **bundle.get("metrics", {}), "ports": bundle.get("ports", [])}
    try:
        df = load_public_dataset()
        return {"ready": False, "rows_available": int(len(df)), "ports": sorted(df["Port"].unique().tolist())}
    except Exception as exc:
        return {"ready": False, "error": str(exc)}

def predict_port_dwell(port_name: str, unique_vessels: float | None = None,
                       total_observations: float | None = None,
                       avg_vessel_length: float | None = None,
                       month_num: int | None = None) -> dict[str, Any]:
    """Predict median dwell days for a supported benchmark port."""
    bundle = load_model_bundle()
    if not bundle:
        raise RuntimeError("The model is not trained. Run: python train_model.py")
    df = load_public_dataset()
    port = str(port_name)
    if port not in bundle["ports"]:
        raise ValueError(f"Unsupported benchmark port '{port}'. Choose one of: {', '.join(bundle['ports'])}")
    port_rows = df[df["Port"].astype(str) == port]
    month_num = int(month_num or pd.Timestamp.today().month)
    # For a selected port, use its historical median feature values for the selected calendar month.
    month_rows = port_rows[port_rows["MonthNum"] == month_num]
    ref = month_rows if not month_rows.empty else port_rows
    row = {
        "UniqueVessels": float(unique_vessels if unique_vessels is not None else ref["UniqueVessels"].median()),
        "TotalObservations": float(total_observations if total_observations is not None else ref["TotalObservations"].median()),
        "AvgVesselLength": float(avg_vessel_length if avg_vessel_length is not None else ref["AvgVesselLength"].median()),
        "MonthNum": month_num,
        "Port": port,
    }
    x = pd.DataFrame([row], columns=FEATURES)
    dwell_days = max(0.0, float(bundle["model"].predict(x)[0]))
    # This is a relative dwell-risk indicator, not an official congestion percentage.
    port_median = float(port_rows["MedianDwellDays"].median())
    port_p90 = float(port_rows["MedianDwellDays"].quantile(0.90))
    if port_p90 <= port_median:
        risk = 0.0
    else:
        risk = float(np.clip((dwell_days - port_median) / (port_p90 - port_median) * 100, 0, 100))
    return {
        "port": port,
        "predicted_dwell_days": round(dwell_days, 2),
        "predicted_dwell_hours": round(dwell_days * 24, 1),
        "dwell_risk_index": round(risk, 1),
        "port_historical_median_days": round(port_median, 2),
        "month_num": month_num,
        "is_local_tamil_nadu_forecast": False,
        "warning": "Historical benchmark for a US port; not calibrated for Tamil Nadu ports.",
    }

# Backward-compatible wrapper for the original app.py signature.
# It deliberately refuses to pretend the public benchmark can produce a validated TN forecast.
def predict_ship_arrival_and_congestion(vessel_name, port_name, container_count, speed_knots, weather_score):
    bundle = load_model_bundle()
    if not bundle:
        raise RuntimeError("Train the public-data model first: python train_model.py")
    port_aliases = {
        "Chennai Port": "Houston",
        "Port Chennai": "Houston",
        "Kamarajar Port (Ennore)": "Houston",
        "Port Ennore": "Houston",
        "Kattupalli Port": "Savannah",
        "Port Kattupalli": "Savannah",
        "VOC Port (Tuticorin)": "Houston",
    }
    benchmark_port = port_aliases.get(port_name, port_name)
    result = predict_port_dwell(benchmark_port)
    # Legacy keys are retained so existing UI integrations do not crash.
    # The numbers are explicitly benchmark proxies, not local vessel ETA forecasts.
    return {
        "eta_delay_hours": result["predicted_dwell_hours"],
        "congestion_pct": result["dwell_risk_index"],
        "ai_summary": (
            f"Benchmark estimate for {benchmark_port}: median vessel dwell is predicted at "
            f"{result['predicted_dwell_days']} days; relative dwell-risk index "
            f"{result['dwell_risk_index']}/100. {result['warning']}"
        ),
        "predicted_dwell_days": result["predicted_dwell_days"],
        "dwell_risk_index": result["dwell_risk_index"],
        "model_type": "RandomForestRegressor trained on public historical dwell data",
        "warning": result["warning"],
    }

def predict_congestion(container_count, speed_knots, weather_score):
    result = predict_ship_arrival_and_congestion(
        "Demo vessel", "Chennai Port", container_count, speed_knots, weather_score
    )
    return result["congestion_pct"]
