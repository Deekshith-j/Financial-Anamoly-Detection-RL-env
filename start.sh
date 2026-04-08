#!/bin/bash
set -e

# Hugging Face exposes exactly ONE port: 7860
# Gradio is now mounted directly inside the FastAPI app.

echo "[start.sh] Starting FastAPI server (with mounted Gradio UI) on port 7860..."
export ENV_BASE_URL="http://127.0.0.1:7860"
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
