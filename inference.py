"""
Inference script — Financial Anomaly Detection OpenEnv.
Mandatory stdout format strictly followed.

Environment variables:
  API_BASE_URL  — default: "https://router.huggingface.co/v1"
  MODEL_NAME    — default: "Qwen/Qwen2.5-72B-Instruct"
  HF_TOKEN      — your Hugging Face API key
  ENV_BASE_URL  — default: "http://localhost:7860"
"""

import os
import json
import textwrap
from typing import List, Optional

import requests
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK    = "financial-anomaly-env"
MAX_STEPS    = 3
TASKS        = ["duplicate_detection", "pattern_fraud", "laundering_chain"]


# ── Mandatory stdout loggers ────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error) -> None:
    print(
        f"[STEP]  step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    r_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END]   success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={r_str}",
        flush=True,
    )


# ── Environment HTTP helpers ────────────────────────────────────────────────────
def env_reset(task_id: str, seed: Optional[int] = None) -> dict:
    params: dict = {"task_id": task_id}
    if seed is not None:
        params["seed"] = seed
    r = requests.post(f"{ENV_BASE_URL}/reset", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(task_id: str, action_dict: dict) -> dict:
    r = requests.post(
        f"{ENV_BASE_URL}/step",
        params={"task_id": task_id},
        json=action_dict,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# ── System prompt ───────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are a senior AML compliance officer.
    Respond ONLY with a valid JSON object, no markdown:
    {
      "flagged_ids":   ["TXN_00001", ...],
      "anomaly_types": ["structuring" or "laundering", ...],
      "report_text":   "<SAR report — see template below>",
      "disposition":   "block",
      "confidence":    0.85
    }

    SAR REPORT TEMPLATE (use this exact structure for laundering_chain task):
    SUSPICIOUS ACTIVITY REPORT
    Account chain identified: [ACC_XXXX] -> [ACC_XXXX] -> [ACC_XXXX] -> [ACC_XXXX]
    Phase 1 - Structuring: Multiple transfers just below $10,000 CTR threshold from [origin] to [inter1].
    Phase 2 - Layering: Consolidated funds moved via [inter1] to [inter2] labeled as consulting/invoices.
    Phase 3 - Integration: Final transfer to international wire or offshore entity.
    Total funds: $[X]. Recommendation: BLOCK and file SAR.

    CRITICAL: flagged_ids and anomaly_types must be same length.
    - Valid anomaly_type labels: duplicate | velocity_attack | round_number | unusual_merchant | after_hours | structuring | laundering
    For laundering_chain: use 'structuring' for Phase 1, 'laundering' for Phase 2 and 3.
    Look for transfers just under $10,000 — these are the structuring transactions.
    Look for 'consulting fees', 'invoice', 'advisory' descriptions — these are layering.
    Look for international wires to offshore entities — these are integration.
""").strip()


# ── LLM call ────────────────────────────────────────────────────────────────────
def call_llm(client: OpenAI, obs: dict, history: list) -> dict:
    """
    Serialize observation, call LLM, return parsed JSON action dict.
    Falls back to a safe no-op action on any error.
    """
    txns_data = [
        {
            "id":                t["id"],
            "timestamp":         t["timestamp"][:16],
            "amount":            t["amount"],
            "account_id":        t["account_id"],
            "counterparty_id":   t["counterparty_id"],
            "merchant":          t["merchant"],
            "merchant_category": t["merchant_category"],
            "transaction_type":  t["transaction_type"],
            "channel":           t["channel"],
        }
        for t in obs["transactions"]
    ]

    # Account profiles — strip internal risk_score before sending to agent
    profiles_data = {
        aid: {
            "account_type":      p["account_type"],
            "age_days":          p["age_days"],
            "avg_monthly_txn":   p["avg_monthly_txn"],
            "typical_merchants": p["typical_merchants"],
            "typical_hours":     p["typical_hours"],
        }
        for aid, p in obs["account_profiles"].items()
    }

    user_msg = textwrap.dedent(f"""
        TASK: {obs['task_instruction']}
        EPISODE: {obs['episode_id']}  STEP: {obs['step_number']}
        BATCH STATS: {json.dumps(obs['batch_stats'])}
        FEEDBACK FROM LAST STEP: {obs.get('feedback') or 'None — this is your first attempt.'}

        ACCOUNT PROFILES:
        {json.dumps(profiles_data, indent=2)}

        TRANSACTIONS ({len(txns_data)} total):
        {json.dumps(txns_data, indent=2)}

        Respond with JSON only.
    """).strip()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-4:])  # Keep last 2 turns (4 messages)
    messages.append({"role": "user", "content": user_msg})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.1,
            max_tokens=1800,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if model wraps output
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)

        # Safety net: ensure parallel lists
        flags  = parsed.get("flagged_ids", [])
        types  = parsed.get("anomaly_types", [])
        min_l  = min(len(flags), len(types))
        parsed["flagged_ids"]   = flags[:min_l]
        parsed["anomaly_types"] = types[:min_l]

        # Safety net: ensure report_text is not empty for laundering task
        if not parsed.get("report_text", "").strip():
            parsed["report_text"] = (
                "No suspicious activity identified in this batch. "
                "All transactions appear consistent with account profiles."
            )
        return parsed
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return {
            "flagged_ids":   [],
            "anomaly_types": [],
            "report_text":   "Fallback empty report.",
            "disposition":   "review",
            "confidence":    0.0,
        }


# ── Episode runner ───────────────────────────────────────────────────────────────
def run_episode(client: OpenAI, task_id: str) -> float:
    """
    Run a single episode for the given task_id.
    Prints mandatory [START] / [STEP] / [END] log lines.
    Returns best_score achieved in the episode.
    """
    rewards: List[float] = []
    history: list = []
    steps_taken = 0
    score = 0.0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env_reset(task_id)
        obs    = result["observation"]
        done   = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action = call_llm(client, obs, history)
            sr     = env_step(task_id, action)

            reward      = sr.get("reward", 0.0)
            done        = sr.get("done", False)
            info        = sr.get("info", {})
            obs         = sr.get("observation", obs)
            rewards.append(reward)
            steps_taken = step
            score       = info.get("best_score", reward)

            flags   = action.get("flagged_ids", [])
            summary = (
                f"flags={flags[:3]}"
                f"{'...' if len(flags) > 3 else ''}"
                f" disp={action.get('disposition', 'review')}"
                f" conf={action.get('confidence', 0):.2f}"
            )
            log_step(
                step=step,
                action=summary,
                reward=reward,
                done=done,
                error=None,
            )
            history.append({"role": "assistant", "content": json.dumps(action)})

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)
        if not rewards:
            rewards = [0.0]

    finally:
        success = score >= 0.5
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )

    return score


# ── Main ─────────────────────────────────────────────────────────────────────────
def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")
    all_scores: List[float] = []

    for task_id in TASKS:
        s = run_episode(client, task_id)
        all_scores.append(s)
        print(f"[DEBUG] {task_id} final score: {s:.3f}", flush=True)

    mean = sum(all_scores) / len(all_scores)
    print(f"[DEBUG] Overall mean score: {mean:.3f}", flush=True)


if __name__ == "__main__":
    main()
