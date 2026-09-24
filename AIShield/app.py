"""
app.py
========
AIShield's Flask web app -- Replit-ready.

Routes:
    GET  /              -> dashboard (single-record form + CSV batch upload)
    POST /predict        -> score one record, submitted as JSON  {feature: value, ...}
    POST /predict_csv     -> score an uploaded CSV file, returns JSON list of rows
    GET  /sample/<kind>   -> kind in {normal, dos, probe, r2l, u2r}; returns a sample
                             record (JSON) used by the "Load Sample" buttons
    POST /api/train        -> trains a starter model from inside the browser (no shell
                             access needed -- this is what powers the "Train Now" button
                             shown when no model exists yet)
    GET  /health          -> simple JSON health/status check (is a model trained?)

Run locally:
    python app.py
Then open http://127.0.0.1:8080

On Replit: click the Run button (see .replit / run.sh) -- the webview opens automatically.
"""
import os
import threading
import time
from collections import deque, Counter

from flask import Flask, render_template, request, jsonify

import utils
import predict as predictor
from train_model import run_training

app = Flask(__name__)

# ---------------------------------------------------------------------
# LIVE SERVER MONITOR
# Monitors HTTP requests received by this Flask application itself.
# This is web-server monitoring, NOT Wi-Fi/packet capture.
# ---------------------------------------------------------------------
_LIVE_MAX_EVENTS = 300
_live_events = deque(maxlen=_LIVE_MAX_EVENTS)
_live_lock = threading.Lock()
_live_monitoring = True

def _live_suspicion(method, path, status_code, response_ms):
    """Simple transparent web-request heuristic for the Live Server tab."""
    score = 0
    reasons = []

    suspicious_paths = (
        "/admin", "/wp-admin", "/wp-login", "/.env", "/phpmyadmin",
        "/login", "/config", "/etc/passwd", "/shell", "/debug"
    )
    p = path.lower().split("?", 1)[0]

    if any(p.startswith(x) for x in suspicious_paths):
        score += 55
        reasons.append("suspicious endpoint")

    if status_code >= 400:
        score += 15
        reasons.append(f"HTTP {status_code}")

    if response_ms > 1500:
        score += 10
        reasons.append("slow response")

    return min(score, 100), reasons

@app.before_request
def _live_request_start():
    request._aishield_start = time.perf_counter()

@app.after_request
def _live_request_record(response):
    start = getattr(request, "_aishield_start", time.perf_counter())
    response_ms = round((time.perf_counter() - start) * 1000, 1)

    if _live_monitoring and request.path != "/api/live/status":
        score, reasons = _live_suspicion(
            request.method, request.path, response.status_code, response_ms
        )
        event = {
            "time": time.strftime("%H:%M:%S"),
            "method": request.method,
            "path": request.path,
            "status": response.status_code,
            "response_ms": response_ms,
            "risk": score,
            "verdict": "SUSPICIOUS" if score >= 50 else "NORMAL",
            "reason": ", ".join(reasons) if reasons else "normal request pattern",
        }
        with _live_lock:
            _live_events.appendleft(event)

    return response

MODELS_DIR = utils.CONFIG["paths"]["models_dir"]

# Guards against two overlapping /api/train requests (e.g. a double-click)
_train_lock = threading.Lock()
_training_in_progress = False


def _model_is_trained() -> bool:
    required = ["preprocessor.joblib", "feature_selector.joblib", "random_forest.joblib"]
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in required)


@app.route("/")
def index():
    return render_template(
        "index.html",
        feature_groups=utils.FEATURE_GROUPS,
        categorical_options=utils.CATEGORICAL_OPTIONS,
        model_ready=_model_is_trained(),
    )


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_trained": _model_is_trained()})


@app.route("/predict", methods=["POST"])
def predict_route():
    if not _model_is_trained():
        return jsonify({"error": "No trained model found yet. Click 'Train Now' above, "
                                  "or run `python train_model.py` from the Shell."}), 400

    payload = request.get_json(force=True, silent=True) or {}
    try:
        result = predictor.predict_single(payload)
    except Exception as exc:  # noqa: BLE001 -- surface a clean error to the UI
        return jsonify({"error": f"Prediction failed: {exc}"}), 400

    return jsonify(result)


@app.route("/predict_csv", methods=["POST"])
def predict_csv_route():
    if not _model_is_trained():
        return jsonify({"error": "No trained model found yet. Click 'Train Now' above, "
                                  "or run `python train_model.py` from the Shell."}), 400

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    try:
        out_df = predictor.predict_csv_bytes(file.read())
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Could not process CSV: {exc}"}), 400

    n_attack = int((out_df["verdict"] == "ATTACK").sum())
    preview_cols = ["verdict", "final_score", "supervised_score", "anomaly_score"]
    preview = out_df[preview_cols].round(4).to_dict(orient="records")

    return jsonify({
        "n_rows": len(out_df),
        "n_attack": n_attack,
        "n_normal": len(out_df) - n_attack,
        "rows": preview[:200],   # cap what we ship to the browser
        "truncated": len(out_df) > 200,
    })


@app.route("/sample/<kind>")
def sample_route(kind):
    if kind not in ("normal", "dos", "probe", "r2l", "u2r"):
        return jsonify({"error": "Unknown sample kind."}), 400
    row = utils.sample_record(kind)
    row = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
    return jsonify(row)


@app.route("/api/live/status")
def live_status():
    """Return the latest requests observed by the AIShield Flask server."""
    with _live_lock:
        events = list(_live_events)
    counts = Counter(e["verdict"] for e in events)
    return jsonify({
        "monitoring": _live_monitoring,
        "total": len(events),
        "normal": counts.get("NORMAL", 0),
        "suspicious": counts.get("SUSPICIOUS", 0),
        "events": events[:100],
    })

@app.route("/api/live/clear", methods=["POST"])
def live_clear():
    with _live_lock:
        _live_events.clear()
    return jsonify({"status": "ok"})

@app.route("/api/live/toggle", methods=["POST"])
def live_toggle():
    global _live_monitoring
    payload = request.get_json(silent=True) or {}
    _live_monitoring = bool(payload.get("enabled", True))
    return jsonify({"monitoring": _live_monitoring})

@app.route("/api/train", methods=["POST"])
def train_route():
    """
    Trains a starter model directly from the browser -- lets AIShield work on
    Replit (or anywhere else) with just the Run button, no Shell access needed.
    Uses a modest synthetic dataset by default so it finishes in well under a
    minute even on a small Replit instance.
    """
    global _training_in_progress

    if not _train_lock.acquire(blocking=False):
        return jsonify({"error": "Training is already in progress -- please wait."}), 409

    try:
        _training_in_progress = True
        payload = request.get_json(silent=True) or {}
        n_samples = int(payload.get("n_samples", 8000))
        n_samples = max(2000, min(n_samples, 60000))  # keep it sane for free-tier instances

        metrics = run_training(dataset="synthetic", n_samples=n_samples, verbose=False)
        predictor.reload_pipeline()  # make the freshly trained model available immediately

        return jsonify({"status": "ok", "metrics": metrics})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Training failed: {exc}"}), 500
    finally:
        _training_in_progress = False
        _train_lock.release()


if __name__ == "__main__":
    # Replit sets $PORT automatically; 8080 matches the port mapped in .replit.
    # host="0.0.0.0" is required for Replit's proxy/webview to reach the app.
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("AISHIELD_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=debug)
