"""
FastAPI server exposing the OpenEnv HTTP interface.
One global FinancialAnomalyEnv instance per task_id.
"""

from fastapi import FastAPI, Query, HTTPException
from server.env import FinancialAnomalyEnv, TASK_IDS
from server.models import FinancialAction, StepResult, ResetResult

app = FastAPI(
    title="Financial Anomaly Detection — OpenEnv",
    version="1.0.0",
    description=(
        "Real RL environment for financial fraud detection. "
        "Each episode generates fresh transactions procedurally. "
        "Ground truth is server-side only — never sent to the agent."
    ),
)

_envs: dict[str, FinancialAnomalyEnv] = {}


def _get_env(task_id: str) -> FinancialAnomalyEnv:
    if task_id not in TASK_IDS:
        raise HTTPException(
            status_code=400, detail=f"Unknown task_id: {task_id}"
        )
    if task_id not in _envs:
        _envs[task_id] = FinancialAnomalyEnv()
    return _envs[task_id]


@app.post("/reset", response_model=ResetResult)
def reset(
    task_id: str = Query(default="duplicate_detection"),
    seed: int = Query(default=None),
):
    """Reset the environment and generate a fresh episode."""
    env = _get_env(task_id)
    return env.reset(task_id=task_id, seed=seed)


@app.post("/step", response_model=StepResult)
def step(
    action: FinancialAction,
    task_id: str = Query(default="duplicate_detection"),
):
    """Submit an action and receive reward + feedback."""
    env = _get_env(task_id)
    return env.step(action)


@app.get("/state")
def state(task_id: str = Query(default="duplicate_detection")):
    """Get the current environment state (no ground truth)."""
    env = _get_env(task_id)
    return env.state()


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "tasks": TASK_IDS,
        "description": "Financial Anomaly Detection OpenEnv",
    }


@app.get("/tasks")
def tasks():
    """List all available task IDs."""
    return {"tasks": TASK_IDS}

# Mount Gradio UI at root so both FastAPI and Gradio run on a single port for HF Spaces
import gradio as gr
from app import demo

app = gr.mount_gradio_app(app, demo, path="/")

def main():
    """CLI entrypoint to run the server."""
    import uvicorn
    import os
    os.environ["ENV_BASE_URL"] = "http://127.0.0.1:7860"
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
