"""
FinancialAnomalyEnv — the real RL environment.

Episode lifecycle:
  reset(task_id, seed=None) -> fresh episode, new transactions generated
  step(action)              -> grade action, return shaped reward + feedback
  state()                   -> current episode state
  close()                   -> cleanup (no-op for now)

Key RL properties:
  - Every reset() generates a NEW episode (different seed -> different data)
  - Agent gets MAX_STEPS=3 attempts per episode with feedback
  - Reward is shaped: includes improvement bonus and quality penalties
  - Ground truth is NEVER in the observation — only in self._ground_truth
"""

import uuid
import random
from server.models import (
    FinancialObservation,
    FinancialAction,
    StepResult,
    ResetResult,
    AccountProfile,
)
from server.generator import TransactionGenerator
from server.graders import (
    grade_duplicate_detection,
    grade_pattern_fraud,
    grade_laundering_chain,
)

MAX_STEPS = 3

GRADERS = {
    "duplicate_detection": grade_duplicate_detection,
    "pattern_fraud": grade_pattern_fraud,
    "laundering_chain": grade_laundering_chain,
}

TASK_IDS = list(GRADERS.keys())


class FinancialAnomalyEnv:

    def __init__(self):
        self._generator = TransactionGenerator()
        self._task_id = None
        self._episode_id = None
        self._transactions = []
        self._account_profiles = {}
        self._ground_truth = {}
        self._step_count = 0
        self._done = False
        self._best_score = 0.0
        self._last_reward = 0.0
        self._last_feedback = ""

    def reset(
        self, task_id: str = "duplicate_detection", seed: int = None
    ) -> ResetResult:
        """
        Generate a fresh episode. If seed is None, generate randomly.
        This means every reset() call produces a different episode.
        """
        if task_id not in TASK_IDS:
            raise ValueError(f"Unknown task: {task_id}")

        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        self._task_id = task_id
        self._episode_id = uuid.uuid4().hex[:12]
        self._step_count = 0
        self._done = False
        self._best_score = 0.0
        self._last_reward = 0.0
        self._last_feedback = ""

        # Generate fresh transactions + ground truth
        (
            self._transactions,
            self._account_profiles,
            self._ground_truth,
        ) = self._generator.generate(task_id, seed)

        obs = self._build_obs(feedback="")
        return ResetResult(observation=obs)

    def step(self, action: FinancialAction) -> StepResult:
        if self._done:
            return StepResult(
                observation=self._build_obs(self._last_feedback),
                reward=0.0,
                done=True,
                info={"error": "Episode finished. Call reset()."},
            )

        self._step_count += 1

        # Grade the action
        grader = GRADERS[self._task_id]
        score, breakdown, feedback = grader(
            action.flagged_ids,
            action.anomaly_types,
            action.report_text,
            self._ground_truth,
        )

        # Reward shaping
        improvement = max(0.0, score - self._best_score)
        shaped = score + 0.1 * improvement

        # Penalty: too-short report for laundering task
        if (
            self._task_id == "laundering_chain"
            and len(action.report_text.strip()) < 80
        ):
            shaped *= 0.75
            feedback += " | PENALTY: SAR report < 80 chars."

        # Penalty: confidence calibration
        if abs(action.confidence - score) > 0.5:
            shaped *= 0.95
            feedback += (
                f" | MINOR PENALTY: confidence={action.confidence:.2f}"
                f" vs actual={score:.2f}"
            )

        # Penalty: clearly wrong disposition
        if score > 0.75 and action.disposition == "approve":
            shaped *= 0.90
            feedback += " | PENALTY: flagged fraud but chose 'approve'."

        shaped = max(0.0, min(1.0, shaped))
        self._best_score = max(self._best_score, score)
        self._last_reward = shaped
        self._last_feedback = feedback
        self._done = score >= 0.99 or self._step_count >= MAX_STEPS

        return StepResult(
            observation=self._build_obs(feedback),
            reward=round(shaped, 4),
            done=self._done,
            info={
                "raw_score": round(score, 4),
                "shaped_reward": round(shaped, 4),
                "breakdown": breakdown,
                "step": self._step_count,
                "best_score": round(self._best_score, 4),
                "episode_id": self._episode_id,
            },
        )

    def state(self) -> dict:
        return {
            "task_id": self._task_id,
            "episode_id": self._episode_id,
            "step_count": self._step_count,
            "done": self._done,
            "last_reward": self._last_reward,
            "best_score": self._best_score,
            "last_feedback": self._last_feedback,
            "n_transactions": len(self._transactions),
            "n_accounts": len(self._account_profiles),
            "total_fraud": self._ground_truth.get("total_fraud", 0),
        }

    def close(self) -> None:
        """No-op cleanup — reserved for future resource release."""
        pass

    def _build_obs(self, feedback: str) -> FinancialObservation:
        instruction = TransactionGenerator.TASK_INSTRUCTIONS[self._task_id]
        txns = self._transactions
        total_amount = sum(t.amount for t in txns)
        n_accounts = len(set(t.account_id for t in txns))
        date_range = (
            [txns[0].timestamp[:10], txns[-1].timestamp[:10]] if txns else []
        )
        return FinancialObservation(
            transactions=txns,
            account_profiles=self._account_profiles,
            task_instruction=instruction,
            step_number=self._step_count,
            feedback=feedback,
            task_id=self._task_id,
            episode_id=self._episode_id,
            batch_stats={
                "n_transactions": len(txns),
                "n_accounts": n_accounts,
                "total_amount": round(total_amount, 2),
                "date_range": date_range,
            },
        )
