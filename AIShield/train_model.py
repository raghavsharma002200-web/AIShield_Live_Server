"""
train_model.py
================
Trains AIShield's hybrid ensemble (Random Forest + XGBoost + SVM + Neural
Network + Isolation Forest) and saves every artifact needed for prediction
into models/.

Usage (CLI):
    python train_model.py                          # auto-detect dataset in dataset/, else synthesize
    python train_model.py --dataset synthetic       # force synthetic data
    python train_model.py --n-samples 60000         # bigger synthetic dataset
    python train_model.py --dataset nsl_kdd         # use NSL-KDD (place KDDTrain+.txt / KDDTest+.txt in dataset/)

`run_training(...)` is also imported directly by app.py to power the
dashboard's "Train Now" button, so this module never runs training logic
just by being imported -- only main() (the CLI) or an explicit
run_training() call does.
"""
import argparse
import os
import time

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import utils


def run_training(dataset=None, data_dir=None, n_samples=40000, attack_ratio=0.35,
                  models_dir=None, verbose=True):
    """
    Loads (or generates) a dataset, trains the full hybrid ensemble, and saves
    every artifact needed for prediction. Returns a small dict of hold-out
    metrics. Safe to call both from the CLI (main(), below) and from
    app.py's /api/train route.
    """
    def log(*a):
        if verbose:
            print(*a)

    config = utils.CONFIG
    data_dir = data_dir or config["data"]["data_dir"]
    models_dir = models_dir or config["paths"]["models_dir"]
    preferred = dataset or config["data"]["preferred_dataset"]

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    if preferred == "synthetic" and not os.path.exists(os.path.join(data_dir, "synthetic_train.csv")):
        log(f"Generating {n_samples} rows of synthetic data...")
        df = utils.generate_synthetic(n_samples, attack_ratio)
        split = int(len(df) * 0.75)
        df.iloc[:split].to_csv(os.path.join(data_dir, "synthetic_train.csv"), index=False)
        df.iloc[split:].to_csv(os.path.join(data_dir, "synthetic_test.csv"), index=False)

    log("=" * 70)
    log("STEP 1/4 -- Loading dataset")
    log("=" * 70)
    X, y_binary, y_category, source = utils.load_dataset(data_dir, preferred)

    log("\n" + "=" * 70)
    log("STEP 2/4 -- Preprocessing (clean, encode, scale, split)")
    log("=" * 70)
    X_train, X_test, y_train, y_test, cat_train, cat_test = utils.split_dataset(
        X, y_binary, y_category,
        test_size=config["data"]["test_size"], random_state=config["data"]["random_state"],
    )
    pre = utils.Preprocessor(scale_method=config["feature_engineering"]["scale_method"])
    X_train_t = pre.fit_transform(X_train)
    X_test_t = pre.transform(X_test)

    log("\n" + "=" * 70)
    log("STEP 3/4 -- Feature selection")
    log("=" * 70)
    selector = utils.FeatureSelector(
        k=config["feature_engineering"]["k_features"],
        enabled=config["feature_engineering"]["select_k_best"],
    )
    X_train_sel = selector.fit_transform(X_train_t, y_train)
    X_test_sel = selector.transform(X_test_t)
    log(f"Selected {X_train_sel.shape[1]} / {X_train_t.shape[1]} features.")

    log("\n" + "=" * 70)
    log("STEP 4/4 -- Training hybrid ensemble")
    log("=" * 70)
    t0 = time.time()
    y_train_reset = y_train.reset_index(drop=True) if hasattr(y_train, "reset_index") else y_train
    ensemble = utils.HybridEnsemble(config)
    ensemble.fit(X_train_sel, y_train_reset, verbose=verbose)
    log(f"Training complete in {time.time() - t0:.1f}s")

    result = ensemble.score(X_test_sel)
    y_test_arr = y_test.values if hasattr(y_test, "values") else y_test
    pred = result["verdict"]
    metrics = {
        "accuracy": accuracy_score(y_test_arr, pred),
        "precision": precision_score(y_test_arr, pred, zero_division=0),
        "recall": recall_score(y_test_arr, pred, zero_division=0),
        "f1": f1_score(y_test_arr, pred, zero_division=0),
        "source": source,
        "n_train": len(X_train_sel),
        "n_test": len(X_test_sel),
        "train_seconds": round(time.time() - t0, 1),
    }
    log("\n" + "-" * 70)
    log("HOLD-OUT TEST METRICS (hybrid fusion)")
    log("-" * 70)
    log(f"  Accuracy : {metrics['accuracy']:.4f}")
    log(f"  Precision: {metrics['precision']:.4f}")
    log(f"  Recall   : {metrics['recall']:.4f}")
    log(f"  F1-score : {metrics['f1']:.4f}")

    log("\nSaving model artifacts...")
    utils.save_pipeline(models_dir, ensemble, pre, selector, extra={"source": source})
    log(f"Done. Artifacts saved to '{models_dir}/'.")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train AIShield's hybrid IDS ensemble")
    parser.add_argument("--dataset", default=None,
                         choices=["nsl_kdd", "cic_ids", "unsw_nb15", "synthetic"],
                         help="Which dataset to prefer (default: from CONFIG, falls back to synthetic)")
    parser.add_argument("--data-dir", default=None, help="Folder containing the dataset (default: dataset/)")
    parser.add_argument("--n-samples", type=int, default=40000, help="Rows if generating synthetic data")
    parser.add_argument("--attack-ratio", type=float, default=0.35)
    parser.add_argument("--models-dir", default=None, help="Where to save trained models (default: models/)")
    args = parser.parse_args()

    run_training(
        dataset=args.dataset, data_dir=args.data_dir, n_samples=args.n_samples,
        attack_ratio=args.attack_ratio, models_dir=args.models_dir, verbose=True,
    )
    print("Now run: python app.py   (or python predict.py --input <file.csv>)")


if __name__ == "__main__":
    main()
