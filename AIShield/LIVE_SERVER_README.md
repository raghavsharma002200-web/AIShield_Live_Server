# AIShield Live Server Monitor

This version adds a **Live Server** tab.

## What it monitors
It monitors HTTP requests received by the AIShield Flask server itself. It records:
- HTTP method
- endpoint
- response status
- response time
- a transparent heuristic risk score
- NORMAL or SUSPICIOUS verdict

It does **not** capture Wi-Fi/network packets and it does not claim that an HTTP request is an NSL-KDD flow.

## Run
From the AIShield project folder:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:8080
```

Open **Live Server** and generate activity by:
- refreshing the dashboard
- loading a sample
- clicking Analyze Traffic
- uploading a CSV

The table refreshes automatically.

## Important
The Live Server heuristic is separate from the NSL-KDD ML prediction pipeline. Your existing Single Record and Batch CSV features continue to use the trained ML models.
