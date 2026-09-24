# 🛡️ AIShield: Hybrid ML-Based Network Intrusion Detection System

A Flask web dashboard around a hybrid ML pipeline that classifies network traffic as
**✅ Normal** or **🚨 Attack**, combining four supervised classifiers with an
unsupervised anomaly detector.

```
Network Traffic → Preprocessing → Feature Selection → ML Models → Hybrid Fusion → Alert
```

## Why "Hybrid"?

1. **Supervised ensemble** (soft-voting): Random Forest + XGBoost + SVM + Neural Network —
   trained to recognize known attack signatures (DoS, Probe/Port-Scan, R2L/Brute-Force, U2R).
2. **Unsupervised anomaly layer**: Isolation Forest, trained only on normal traffic — flags
   traffic that statistically "looks weird" even if no supervised model has seen that exact
   attack pattern before.
3. **Fusion**: `final_score = α · supervised_score + (1 − α) · anomaly_score`, thresholded into
   a final verdict. This is what gives AIShield better robustness than any single model alone.

## Project Structure

```
AIShield/
├── app.py                # Flask web app (dashboard + JSON endpoints, incl. /api/train)
├── train_model.py        # CLI + run_training(): loads/generates data, trains the ensemble, saves to models/
├── predict.py            # Loads trained pipeline; used by app.py AND as a standalone batch-scoring CLI
├── utils.py              # Shared core: data loading/synthesis, preprocessing, all 5 models, fusion logic
├── requirements.txt
├── .replit                # Replit run command + port mapping + deployment (gunicorn) config
├── run.sh                 # Replit's Run button executes this: pip install, then start the app
├── models/               # Trained model artifacts land here after training (.joblib files)
├── dataset/              # Put NSL-KDD / CIC-IDS2017 / UNSW-NB15 files here (or let it auto-generate synthetic data)
├── templates/
│   └── index.html        # Dashboard UI (single-record form + CSV batch upload + Train Now button)
├── static/
│   ├── style.css         # Dark cybersecurity dashboard theme
│   └── script.js         # Form submission, sample loading, CSV upload, training -- all via fetch()
└── README.md
```

> **Why is there a `utils.py` even though it wasn't in the requested layout?**
> `train_model.py` (which builds the models) and `predict.py` / `app.py` (which load and use
> them) need to share the *exact same* class definitions — that's a hard requirement of
> Python's `joblib`/pickle serialization, not a style choice. Keeping one shared module avoids
> duplicating hundreds of lines of preprocessing/model code between the two scripts.

## 🚀 Run on Replit

1. Create a new Replit and import/upload this project (or paste in the files).
2. Click **Run**. `run.sh` installs `requirements.txt` and starts `app.py` on
   `0.0.0.0:$PORT` (mapped to port 8080 → 80 in `.replit`), so the webview opens
   automatically once it boots.
3. No model exists yet on a fresh Repl, so the dashboard shows an amber
   **"No trained model yet"** badge with an **⚡ Train Now** button — click it.
   It trains the full hybrid ensemble on an 8,000-row synthetic dataset
   (~15-30s on a typical Repl) directly from the browser, no Shell needed.
4. Once training finishes the page reloads and you can use both the
   **Single Record** form and **Batch CSV Upload** tabs right in the webview.

Prefer the Shell instead? `python train_model.py` works exactly the same way and
lets you pass a bigger `--n-samples` or point at a real dataset (see below).

To publish it as a standing Replit **Deployment** (rather than just the
dev webview), use Replit's Deploy button — `.replit`'s `[deployment]` section
already runs it via `gunicorn` instead of Flask's dev server.

## Quick Start (local / any host)

```bash
pip install -r requirements.txt

# 1. Train the hybrid ensemble (auto-generates a synthetic dataset if dataset/ is empty):
python train_model.py

# 2. Launch the web dashboard:
python app.py
# -> open http://127.0.0.1:8080  (or whichever $PORT is set)

# 3. (Optional) Score a CSV from the command line instead of the browser:
python predict.py --input dataset/some_traffic.csv --output results.csv
```

## Using the Dashboard

- **Single Record tab**: fill in the 41 traffic features (grouped into Basic Connection,
  Content, Time-based Traffic, and Host-based Traffic features), or click a **Load sample**
  chip (Normal / DoS / Probe / R2L / U2R) to auto-fill a realistic example, then hit
  **Analyze Traffic**. You'll see the final fused verdict, the supervised-vs-anomaly score
  breakdown, and each individual model's vote.
- **Batch (CSV Upload) tab**: upload a CSV with the same raw feature columns (a `label`
  column, if present, is ignored) to score many records at once.

## Using Real Datasets

By default AIShield trains on a synthetic, NSL-KDD-shaped dataset (built automatically the
first time you run `train_model.py`, or on demand with `python train_model.py --dataset synthetic`)
so you can try the whole system with zero setup. To use a real dataset instead, drop the files
into `dataset/` and tell `train_model.py` which one to prefer:

| Dataset | Files expected in `dataset/` | Command |
|---|---|---|
| **NSL-KDD** | `KDDTrain+.txt`, `KDDTest+.txt` | `python train_model.py --dataset nsl_kdd` |
| **UNSW-NB15** | `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv` | `python train_model.py --dataset unsw_nb15` |
| **CIC-IDS2017 / CSE-CIC-IDS2018** | any `*.csv` with a `Label` column | `python train_model.py --dataset cic_ids` |

After training on a new dataset, restart `app.py` (or just re-run `train_model.py` — the app
reads fresh model files from `models/` on each server restart).

## API Endpoints (used internally by the dashboard, but usable directly too)

| Method | Route | Body | Response |
|---|---|---|---|
| `GET`  | `/` | — | HTML dashboard |
| `GET`  | `/health` | — | `{"status": "ok", "model_trained": true/false}` |
| `POST` | `/predict` | JSON: `{"duration": 0, "protocol_type": "tcp", ...}` | JSON verdict + scores |
| `POST` | `/predict_csv` | multipart form, field `file` = CSV | JSON summary + row-level predictions |
| `GET`  | `/sample/<kind>` | — (`kind` ∈ normal/dos/probe/r2l/u2r) | JSON of a sample record |
| `POST` | `/api/train` | JSON: `{"n_samples": 8000}` (optional) | Trains a starter model synchronously; returns hold-out metrics |

Example with `curl`:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"duration": 0, "protocol_type": "tcp", "service": "private", "flag": "S0",
       "src_bytes": 0, "dst_bytes": 0, "count": 400, "srv_count": 400,
       "serror_rate": 0.95, "same_srv_rate": 0.99}'
```
(Any fields you omit default to `0` / the first category option — see `utils.NSL_KDD_COLUMNS`.)

## Detected Attack Categories

- **DoS/DDoS** — e.g. neptune, smurf, back, teardrop, land
- **Probe / Port Scanning** — e.g. satan, ipsweep, nmap, portsweep
- **R2L (Remote-to-Local) / Brute Force** — e.g. guess_passwd, ftp_write, warezclient
- **U2R (User-to-Root) / Privilege escalation / Botnet-style** — e.g. buffer_overflow, rootkit, perl

## Notes on `xgboost`

If `xgboost` isn't installed, `utils.py` automatically falls back to scikit-learn's
`GradientBoostingClassifier` so training never breaks — install `xgboost`
(already in `requirements.txt`) for the real, faster, higher-accuracy version.

## Production Notes

- `app.py` runs Flask's built-in dev server (`debug=True`) — fine for local testing, but for
  real deployment run it behind a WSGI server instead, e.g.:
  ```bash
  gunicorn -w 4 -b 0.0.0.0:8000 app:app
  ```
- Retraining is CLI-only by design (`train_model.py`) rather than a web route — training can
  take a while and shouldn't block a web request or be triggerable by an unauthenticated user.

## Extending AIShield

- **New model**: add a class with `.fit()` / `.predict_proba()` in `utils.py`, register it in
  `HybridEnsemble.supervised_models` and give it a weight in `CONFIG["hybrid_fusion"]["weights"]`.
- **New dataset**: add a loader function in `utils.py` following the existing
  `_load_nsl_kdd` / `_load_unsw_nb15` pattern — everything downstream is dataset-agnostic.
- **Real packet capture**: replace the CSV/manual-form input path with a live feature
  extractor (e.g. built on `scapy` or `CICFlowMeter`) that calls `predict.predict_single()`
  per flow — the ML pipeline itself needs no changes.
