#!/bin/bash
set -e

echo "[start.sh] Starting FastAPI server on port 7860..."
python -m uvicorn server.main:app --host 0.0.0.0 --port 7860 &
FASTAPI_PID=$!

echo "[start.sh] Waiting for FastAPI to be ready..."
sleep 3

echo "[start.sh] Starting Gradio UI on port 7861..."
python app.py &
GRADIO_PID=$!

echo "[start.sh] Both services running — FastAPI PID=$FASTAPI_PID  Gradio PID=$GRADIO_PID"

# Wait for either process to exit and propagate exit code
wait $FASTAPI_PID $GRADIO_PID
