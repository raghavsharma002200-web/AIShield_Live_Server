"""
predict.py
============
Loads the trained hybrid pipeline (once) and exposes simple functions for
scoring traffic, used both by app.py (the web UI) and as a standalone CLI
for batch scoring:

    python predict.py --input path/to/traffic.csv [--output out.csv]

Functions importable by app.py:
    get_pipeline()                 -> cached (ensemble, preprocessor, selector)
    predict_single(feature_dict)   -> dict with verdict + scores for one record
    predict_csv_bytes(file_bytes)  -> DataFrame of predictions for an uploaded CSV
"""
import argparse
import io
import os

import pandas as pd

import utils

_PIPELINE_CACHE = None


def get_pipeline(models_dir=None):
    """Loads (and caches) the trained ensemble + preprocessor + feature selector."""
    global _PIPELINE_CACHE
    if _PIPELINE_CACHE is None:
        models_dir = models_dir or utils.CONFIG["paths"]["models_dir"]
        _PIPELINE_CACHE = utils.load_pipeline(models_dir)
    return _PIPELINE_CACHE


def reload_pipeline(models_dir=None):
    """Forces a fresh load -- call this after retraining if the app is already running."""
    global _PIPELINE_CACHE
    _PIPELINE_CACHE = None
    return get_pipeline(models_dir)


def predict_single(feature_dict: dict) -> dict:
    """Scores one record (given as a dict of raw feature -> value)."""
    ensemble, pre, selector = get_pipeline()
    df = pd.DataFrame([feature_dict])
    # Ensure every expected raw column is present, defaulting sensibly if not.
    for col in utils.NSL_KDD_COLUMNS:
        if col not in df.columns:
            df[col] = 0 if col not in utils.CATEGORICAL_OPTIONS else utils.CATEGORICAL_OPTIONS[col][0]
    out = utils.predict_dataframe(df, ensemble, pre, selector)
    row = out.iloc[0].to_dict()
    return row


def predict_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Scores a full DataFrame of raw traffic records."""
    ensemble, pre, selector = get_pipeline()
    df_raw = df_raw.drop(columns=[c for c in ["label"] if c in df_raw.columns])
    return utils.predict_dataframe(df_raw, ensemble, pre, selector)


def predict_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    """Scores an uploaded CSV file (as raw bytes) -- used by the /predict_csv web route."""
    df_raw = pd.read_csv(io.BytesIO(file_bytes))
    results = predict_dataframe(df_raw)
    # Attach the original input columns alongside the predictions for context
    return pd.concat([df_raw.reset_index(drop=True), results.reset_index(drop=True)], axis=1)


def main():
    parser = argparse.ArgumentParser(description="Score traffic records with AIShield's hybrid ensemble")
    parser.add_argument("--input", required=True, help="CSV file of traffic records to classify")
    parser.add_argument("--output", default=None, help="Where to save predictions CSV")
    parser.add_argument("--models-dir", default=None)
    args = parser.parse_args()

    get_pipeline(args.models_dir)
    df = pd.read_csv(args.input)
    out = predict_dataframe(df)

    n_attack = (out["verdict"] == "ATTACK").sum()
    print(f"Scored {len(out)} records: {n_attack} ATTACK, {len(out) - n_attack} NORMAL.")
    print(out.head(10).to_string())

    output_path = args.output or (os.path.splitext(args.input)[0] + "_predictions.csv")
    out.to_csv(output_path, index=False)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
