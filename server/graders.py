"""
All graders return (score: float, breakdown: dict, feedback: str).
Score is always in [0.0, 1.0].
Graders are DETERMINISTIC given the same inputs.
"""

from typing import List, Dict, Tuple


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, round(v, 4)))


def grade_duplicate_detection(
    flagged_ids: List[str],
    anomaly_types: List[str],
    report_text: str,
    ground_truth: dict,
) -> Tuple[float, dict, str]:
    """
    F1 score between predicted duplicate IDs and actual.
    - Precision: what fraction of flagged IDs are real duplicates
    - Recall: what fraction of real duplicates were found
    - F1 = harmonic mean
    - Report bonus: +0.05 if report_text > 40 chars and mentions
      "duplicate" (case insensitive). Capped so total <= 1.0.
    """
    # Truncate to min(len) to handle mismatches safely
    min_len = min(len(flagged_ids), len(anomaly_types)) if anomaly_types else len(flagged_ids)
    flagged_ids = flagged_ids[:min_len] if anomaly_types else flagged_ids

    actual = set(ground_truth["fraud_map"].keys())
    predicted = set(flagged_ids)
    tp = len(predicted & actual)
    fp = len(predicted - actual)
    fn = len(actual - predicted)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    report_bonus = (
        0.05
        if len(report_text.strip()) > 40 and "duplicate" in report_text.lower()
        else 0.0
    )
    score = _clamp(f1 + report_bonus)
    breakdown = {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "report_bonus": report_bonus,
        "actual_duplicates": len(actual),
    }
    feedback = (
        f"Found {tp}/{len(actual)} duplicates. "
        f"False positives: {fp}. "
        f"Precision={precision:.2f} Recall={recall:.2f} F1={f1:.2f}. "
        + (
            f"Missing: {list(actual - predicted)[:3]}"
            if fn > 0
            else "Perfect recall!"
        )
    )
    return score, breakdown, feedback


def grade_pattern_fraud(
    flagged_ids: List[str],
    anomaly_types: List[str],
    report_text: str,
    ground_truth: dict,
) -> Tuple[float, dict, str]:
    """
    Weighted accuracy:
      Correct ID + correct type  = 1.0 per transaction
      Correct ID + wrong type    = 0.5 per transaction (found but misclassified)
      False positive             = -0.2 penalty per extra flag
    Report bonus: +0.04 per distinct active pattern named in report_text
    (max +0.16). Final score clamped to [0.0, 1.0].
    """
    fraud_map = ground_truth["fraud_map"]  # {txn_id: true_label}
    # Normalize: truncate parallel lists to same length
    min_len = min(len(flagged_ids), len(anomaly_types))
    flagged_ids = flagged_ids[:min_len]
    anomaly_types = [a.lower().strip() for a in anomaly_types[:min_len]]

    predicted_map = dict(zip(flagged_ids, anomaly_types))
    actual_ids = set(fraud_map.keys())
    predicted_ids = set(predicted_map.keys())

    earned = 0.0
    correct_ids = 0
    type_correct = 0
    for tid, pred_type in predicted_map.items():
        if tid in fraud_map:
            correct_ids += 1
            if pred_type == fraud_map[tid]:
                earned += 1.0
                type_correct += 1
            else:
                earned += 0.5
        else:
            earned -= 0.2  # false positive penalty

    total_possible = len(actual_ids)
    raw_score = earned / total_possible if total_possible > 0 else 0.0

    # Report bonus
    active_patterns = set(fraud_map.values())
    report_lower = report_text.lower()
    bonus = sum(0.04 for p in active_patterns if p in report_lower)
    bonus = min(bonus, 0.16)

    score = _clamp(raw_score + bonus)
    missed = actual_ids - predicted_ids
    breakdown = {
        "correct_ids": correct_ids,
        "type_correct": type_correct,
        "false_positives": len(predicted_ids - actual_ids),
        "missed_transactions": len(missed),
        "raw_score": round(raw_score, 3),
        "report_bonus": round(bonus, 3),
        "active_patterns": list(active_patterns),
    }
    feedback = (
        f"Correctly identified {correct_ids}/{len(actual_ids)} fraud transactions. "
        f"Type accuracy: {type_correct}/{correct_ids if correct_ids else 1}. "
        f"False positives: {len(predicted_ids - actual_ids)}. "
        f"Active patterns this episode: {list(active_patterns)}. "
        + (f"Missed IDs: {list(missed)[:3]}" if missed else "")
    )
    return score, breakdown, feedback


def grade_laundering_chain(
    flagged_ids: List[str],
    anomaly_types: List[str],
    report_text: str,
    ground_truth: dict,
) -> Tuple[float, dict, str]:
    """
    Component-weighted scoring:

    Component 1 - Chain recall (0.45 weight):
      How many of the true chain transactions did the agent find?
      Only count transactions labeled 'structuring' or 'laundering'.
      score = (correct chain txns found) / total_chain_txns

    Component 2 - False positive penalty (0.20 weight):
      score = max(0, 1.0 - fp_count * 0.15)

    Component 3 - Account chain coverage (0.15 weight):
      Count how many of the 4 chain accounts appear in the agent's
      flagged transactions. score = matched_accounts / 4

    Component 4 - SAR report quality (0.20 weight):
      score_report checks (each worth 0.25):
        a) All 4 chain account IDs mentioned in report_text
        b) Word "structuring" OR "smurfing" in report_text
        c) Word "SAR" OR "suspicious activity" in report_text
        d) Report length >= 150 characters
    """
    fraud_map = ground_truth["fraud_map"]
    chain_accounts = ground_truth.get("chain_accounts", [])
    chain_ids = {
        tid
        for tid, lbl in fraud_map.items()
        if lbl in ("structuring", "laundering")
    }

    min_len = min(len(flagged_ids), len(anomaly_types))
    flagged_ids_n = flagged_ids[:min_len]
    anomaly_types_n = [a.lower().strip() for a in anomaly_types[:min_len]]
    predicted_chain = {
        tid
        for tid, lbl in zip(flagged_ids_n, anomaly_types_n)
        if lbl in ("structuring", "laundering")
    }
    all_predicted = set(flagged_ids)

    # Component 1: chain recall
    tp_chain = len(predicted_chain & chain_ids)
    c1 = tp_chain / len(chain_ids) if chain_ids else 0.0

    # Component 2: FP penalty
    fp = len(all_predicted - set(fraud_map.keys()))
    c2 = max(0.0, 1.0 - fp * 0.15)

    # Component 3: account coverage
    report_lower = report_text.lower()
    matched_accts = sum(
        1 for acc in chain_accounts if acc.lower() in report_lower
    )
    c3 = matched_accts / 4 if chain_accounts else 0.0

    # Component 4: SAR quality
    c4_checks = [
        all(acc.lower() in report_lower for acc in chain_accounts),
        any(w in report_lower for w in ("structuring", "smurfing")),
        any(w in report_lower for w in ("sar", "suspicious activity")),
        len(report_text.strip()) >= 150,
    ]
    c4 = sum(c4_checks) / 4

    score = _clamp(0.45 * c1 + 0.20 * c2 + 0.15 * c3 + 0.20 * c4)
    breakdown = {
        "chain_recall": round(c1, 3),
        "fp_penalty_score": round(c2, 3),
        "account_coverage": round(c3, 3),
        "sar_quality": round(c4, 3),
        "chain_tp": tp_chain,
        "chain_total": len(chain_ids),
        "false_positives": fp,
        "accounts_mentioned": matched_accts,
        "sar_checks": c4_checks,
    }
    feedback = (
        f"Chain recall: {tp_chain}/{len(chain_ids)} transactions. "
        f"False positives: {fp}. "
        f"Accounts in report: {matched_accts}/4. "
        f"SAR quality checks passed: {sum(c4_checks)}/4. "
        f"Chain accounts are: {chain_accounts}. "
        + (
            "Tip: include all chain account IDs in your report. "
            if matched_accts < 4
            else ""
        )
        + (
            "Tip: use the word 'structuring' or 'smurfing'. "
            if not c4_checks[1]
            else ""
        )
    )
    return score, breakdown, feedback
