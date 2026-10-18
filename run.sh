#!/usr/bin/env bash
# TrustGate launcher.  Usage: ./run.sh [port]
set -e
cd "$(dirname "$0")"
PORT="${1:-8000}"
echo "machine: $(uname -m)  python: $(python3 --version)"
if [ ! -d .venv ]; then python3 -m venv .venv; fi
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt || { echo "pip install failed -> try: pip install flask  (see README fallback)"; exit 1; }
echo "starting on http://0.0.0.0:$PORT"
exec python -m uvicorn app:app --host 0.0.0.0 --port "$PORT"
