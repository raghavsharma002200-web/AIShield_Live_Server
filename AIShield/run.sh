#!/usr/bin/env bash
# Replit's "Run" button executes this. Installing on every run keeps things
# simple/robust across Replit's package-manager versions; pip's cache makes
# repeat installs fast after the first one.
set -e
pip install -q -r requirements.txt
python3 app.py
