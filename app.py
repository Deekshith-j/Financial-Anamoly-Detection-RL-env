"""
Gradio interface for the Financial Anomaly Detection OpenEnv.
Wraps inference logic with a live-streaming UI.
Prints to stdout (judging compliance) AND updates Gradio in real time.

Imports all env/llm helpers from inference.py — no circular imports.
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

import requests
import gradio as gr
from openai import OpenAI

from inference import (
    env_reset,
    env_step,
    call_llm,
    run_episode,
    log_start,
    log_step,
    log_end,
    TASKS,
    BENCHMARK,
    MAX_STEPS,
)

API_BASE_URL_DEFAULT = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME_DEFAULT   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN_DEFAULT     = os.getenv("HF_TOKEN",     "")
ENV_BASE_URL_DEFAULT = os.getenv("ENV_BASE_URL", "http://localhost:7860")


# ── Streaming episode runner ─────────────────────────────────────────────────────

def run_episode_streaming(task_id, api_base, model, token, env_url):
    """
    Generator function — yields updated UI state after each step.
    Also prints mandatory stdout logs for judging.
    Yields: (log_text, score, status, steps, rewards_str, last_action, txn_rows, summary_rows)
    """
    os.environ["API_BASE_URL"] = api_base
    os.environ["MODEL_NAME"]   = model
    os.environ["HF_TOKEN"]     = token
    os.environ["ENV_BASE_URL"] = env_url

    client    = OpenAI(base_url=api_base, api_key=token or "dummy")
    log_lines = []
    rewards   = []
    last_action: dict = {}
    txn_rows  = []
    steps_taken = 0
    score = 0.0

    def add_log(line: str) -> None:
        log_lines.append(line)
        print(line, flush=True)

    add_log(f"[START] task={task_id} env={BENCHMARK} model={model}")

    try:
        # Verify environment is reachable before starting
        try:
            requests.get(f"{env_url}/health", timeout=5).raise_for_status()
        except Exception as e:
            add_log(f"[ERROR] Cannot reach env at {env_url}: {e}")
            yield (
                "\n".join(log_lines), 0.0, "ENV UNREACHABLE",
                0, "", {}, [], []
            )
            return

        result = env_reset(task_id)
        obs    = result["observation"]
        done   = False

        # Build initial transaction table rows
        txn_rows = [
            [
                t["id"],
                t["timestamp"][:16],
                f"${t['amount']:.2f}",
                t["account_id"],
                t["merchant"],
                t["merchant_category"],
                t["transaction_type"],
                t["channel"],
                t["country"],
            ]
            for t in obs["transactions"]
        ]

        batch = obs.get("batch_stats", {})
        add_log(
            f"[INFO] Episode {obs['episode_id']} | "
            f"{batch.get('n_transactions', '?')} transactions | "
            f"{batch.get('n_accounts', '?')} accounts | "
            f"Total: ${batch.get('total_amount', '?')}"
        )

        # Yield immediately after reset so the transaction table populates
        yield (
            "\n".join(log_lines), 0.0, "Running...",
            0, "", {}, txn_rows, []
        )

        history = []
        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            add_log(f"[INFO] Step {step}/{MAX_STEPS} — calling LLM...")
            yield (
                "\n".join(log_lines), score, "Running...",
                steps_taken, str(rewards), last_action, txn_rows, []
            )

            action = call_llm(client, obs, history)
            sr     = env_step(task_id, action)

            reward      = sr.get("reward", 0.0)
            done        = sr.get("done", False)
            info        = sr.get("info", {})
            obs         = sr.get("observation", obs)
            rewards.append(reward)
            steps_taken = step
            score       = info.get("best_score", reward)
            last_action = action

            flags   = action.get("flagged_ids", [])
            summary = (
                f"flags={flags[:3]}"
                f"{'...' if len(flags) > 3 else ''}"
                f" disp={action.get('disposition', 'review')}"
                f" conf={action.get('confidence', 0):.2f}"
            )
            log_line = (
                f"[STEP]  step={step} action={summary} "
                f"reward={reward:.2f} done={str(done).lower()} error=null"
            )
            add_log(log_line)

            breakdown = info.get("breakdown", {})
            add_log(f"[FEEDBACK] {json.dumps(breakdown)}")

            history.append({"role": "assistant", "content": json.dumps(action)})

            yield (
                "\n".join(log_lines), score, "Running...",
                steps_taken, str([round(r, 2) for r in rewards]),
                last_action, txn_rows, []
            )

    except Exception as e:
        add_log(f"[ERROR] {e}")

    finally:
        success = score >= 0.5
        end_line = (
            f"[END]   success={str(success).lower()} "
            f"steps={steps_taken} score={score:.2f} "
            f"rewards={','.join(f'{r:.2f}' for r in rewards)}"
        )
        add_log(end_line)

        summary_rows = [[task_id, f"{score:.3f}", str(success), steps_taken]]
        yield (
            "\n".join(log_lines),
            score,
            "✅ SUCCESS" if success else "⚠ BELOW THRESHOLD",
            steps_taken,
            str([round(r, 2) for r in rewards]),
            last_action,
            txn_rows,
            summary_rows,
        )


def run_all_tasks_fn(api_base, model, token, env_url):
    """Run all 3 tasks sequentially and return a summary table."""
    os.environ["API_BASE_URL"] = api_base
    os.environ["MODEL_NAME"]   = model
    os.environ["HF_TOKEN"]     = token
    os.environ["ENV_BASE_URL"] = env_url

    client = OpenAI(base_url=api_base, api_key=token or "dummy")
    rows = []
    for task_id in TASKS:
        s = run_episode(client, task_id)
        rows.append([task_id, f"{s:.3f}", str(s >= 0.5)])
    return rows


# ── Gradio Layout ──────────────────────────────────────────────────────────────

TXN_COLS = [
    "ID", "Timestamp", "Amount", "Account",
    "Merchant", "Category", "Type", "Channel", "Country",
]

CSS = """
body { background: #0f1117; }
.gradio-container { max-width: 1400px !important; }
#episode_log textarea {
    font-family: 'Courier New', monospace;
    font-size: 12px;
    background: #1a1d27;
    color: #a8ff78;
}
"""

with gr.Blocks(
    title="Financial Anomaly Detection — OpenEnv",
    theme=gr.themes.Base(
        primary_hue="emerald",
        neutral_hue="slate",
    ),
    css=CSS,
) as demo:

    gr.HTML("""
    <div style="text-align:center; padding: 24px 0 8px;">
        <h1 style="font-size:2rem; font-weight:700; color:#a8ff78; margin:0;">
            💹 Financial Anomaly Detection — OpenEnv
        </h1>
        <p style="color:#94a3b8; margin-top:8px;">
            Real RL environment — each episode generates fresh transactions procedurally.
            No hardcoded data. Ground truth is server-side only.
        </p>
    </div>
    """)

    # ── Configuration Row ────────────────────────────────────────────────────
    with gr.Row():
        api_base_in = gr.Textbox(
            label="🔗 API Base URL",
            value=API_BASE_URL_DEFAULT,
            scale=3,
            visible=False,
        )
        model_in = gr.Textbox(
            label="🤖 Model Name",
            value=MODEL_NAME_DEFAULT,
            scale=3,
            visible=False,
        )
        token_in = gr.Textbox(
            label="🔑 HF Token",
            type="password",
            value=HF_TOKEN_DEFAULT,
            scale=2,
            visible=False,
        )
        env_url_in = gr.Textbox(
            label="🌐 Env Base URL",
            value=ENV_BASE_URL_DEFAULT,
            scale=2,
            visible=False,
        )

    # ── Task Selection + Controls ────────────────────────────────────────────
    with gr.Row():
        task_sel = gr.Radio(
            choices=TASKS,
            value="duplicate_detection",
            label="🎯 Select Task",
            scale=4,
        )
        run_btn     = gr.Button("▶ Run Episode",    variant="primary", scale=1, min_width=140)
        run_all_btn = gr.Button("⚡ Run All 3 Tasks", variant="secondary", scale=1, min_width=160)

    # ── Live Episode Log ─────────────────────────────────────────────────────
    episode_log = gr.Textbox(
        label="📋 Episode Log (live)",
        lines=20,
        max_lines=40,
        interactive=False,
        elem_id="episode_log",
    )

    # ── Metrics Row ──────────────────────────────────────────────────────────
    with gr.Row():
        score_out   = gr.Number(label="🏆 Best Score (0–1)", precision=3, scale=1)
        status_out  = gr.Textbox(label="📊 Status", interactive=False, scale=1)
        steps_out   = gr.Number(label="👣 Steps Taken", precision=0, scale=1)
        rewards_out = gr.Textbox(label="💰 Rewards per Step", interactive=False, scale=2)

    # ── Transaction Batch Table ──────────────────────────────────────────────
    txn_table = gr.Dataframe(
        headers=TXN_COLS,
        label="📄 Transaction Batch (generated fresh each episode)",
        interactive=False,
        wrap=True,
        row_count=(10, "dynamic"),
    )

    # ── Agent Action JSON ────────────────────────────────────────────────────
    with gr.Row():
        action_json = gr.JSON(
            label="🕵️ Agent's Last Action",
            scale=1,
        )

    # ── All Tasks Summary ────────────────────────────────────────────────────
    summary_table = gr.Dataframe(
        headers=["Task", "Score", "Success", "Steps"],
        label="📈 All Tasks Summary",
        interactive=False,
    )

    # ── Wire Up Buttons ──────────────────────────────────────────────────────
    run_btn.click(
        fn=run_episode_streaming,
        inputs=[task_sel, api_base_in, model_in, token_in, env_url_in],
        outputs=[
            episode_log, score_out, status_out, steps_out,
            rewards_out, action_json, txn_table, summary_table,
        ],
    )

    run_all_btn.click(
        fn=run_all_tasks_fn,
        inputs=[api_base_in, model_in, token_in, env_url_in],
        outputs=[summary_table],
    )

    gr.HTML("""
    <div style="text-align:center; padding:12px; color:#475569; font-size:0.8rem;">
        Financial Anomaly Detection OpenEnv · Procedural RL · Ground truth is server-side only
    </div>
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
