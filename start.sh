#!/bin/bash
set -e

# Hugging Face exposes exactly ONE port: 7860
# So we run Gradio on 7860 (public) and FastAPI on 7861 (internal).

echo "[start.sh] Starting FastAPI server on internal port 7861..."
python -m uvicorn server.main:app --host 0.0.0.0 --port 7861 &
FASTAPI_PID=$!

echo "[start.sh] Waiting for FastAPI to be ready..."
sleep 3

echo "[start.sh] Starting Gradio UI on public port 7860..."
export ENV_BASE_URL="http://localhost:7861"
python app.py &
GRADIO_PID=$!

echo "[start.sh] Both services running — FastAPI PID=$FASTAPI_PID  Gradio PID=$GRADIO_PID"

# Wait for either process to exit and propagate exit code
wait $FASTAPI_PID $GRADIO_PID
