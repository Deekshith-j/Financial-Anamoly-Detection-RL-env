"""
TransactionGenerator — creates a fresh, realistic transaction batch
every episode using only random seeds. NO static files.

Design:
  1. Create N accounts (N ~ Uniform(6, 12))
  2. Generate B baseline transactions per account (B ~ Gaussian(4, 1.5))
     distributed across a 30-day window, matching each account's
     behavioral profile (hours, merchants, amount distribution)
  3. Inject fraud patterns based on task_id:
     - duplicate_detection: inject DuplicateInjector only
     - pattern_fraud:       inject Velocity + RoundNumber +
                            UnusualMerchant + AfterHours (all active,
                            each with 0.7 probability of appearing)
     - laundering_chain:    inject LaunderingChainInjector + optionally
                            1-2 other patterns as noise (0.4 probability)
  4. Shuffle all transactions by timestamp
  5. Return (transactions, account_profiles, ground_truth)

ground_truth is a dict built from injector outputs:
  {
    "fraud_map":      {txn_id: anomaly_type, ...},  # ALL fraud txns
    "chain_accounts": [acc1, acc2, acc3, acc4],      # laundering only
    "total_fraud":    int,
    "total_clean":    int,
  }
Ground truth NEVER leaves the generator — it stays in env._ground_truth.
"""

import random
import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Dict

from server.models import TransactionRecord, AccountProfile
from server.account_simulator import generate_account_pool, MERCHANT_POOLS
from server.fraud_injector import (
    DuplicateInjector,
    VelocityAttackInjector,
    RoundNumberInjector,
    UnusualMerchantInjector,
    AfterHoursInjector,
    LaunderingChainInjector,
)


def _iso(dt: datetime) -> str:
    return dt.isoformat() + "Z"


def generate_baseline_transaction(
    account_id: str,
    profile: AccountProfile,
    seq: int,
    rng: random.Random,
    base_date: datetime,
) -> TransactionRecord:
    """
    Generate one realistic transaction matching the account's profile.
    Amount drawn from lognormal centered on account's avg_monthly_txn/20.
    Timestamp drawn from a distribution matching typical_hours.
    Merchant drawn from typical_merchants pool.
    """
    # Amount: lognormal so most are small, occasional large ones
    mu_amount = profile.avg_monthly_txn / 20
    amount = round(
        float(
            np.random.lognormal(
                mean=np.log(max(mu_amount, 1)),
                sigma=0.6,
            )
        ),
        2,
    )
    amount = max(0.50, min(amount, profile.avg_monthly_txn * 2))

    # Timestamp: respect typical_hours
    day_offset = rng.randint(0, 29)
    if profile.typical_hours == "business":
        hour = rng.randint(8, 18)
    elif profile.typical_hours == "evening":
        hour = rng.randint(17, 23)
    else:  # anytime
        hour = rng.randint(0, 23)
    minute = rng.randint(0, 59)
    dt = base_date + timedelta(days=day_offset, hours=hour, minutes=minute)

    # Merchant: pick from account's typical list
    merchant = rng.choice(profile.typical_merchants)

    # Infer category from merchant name
    category = "Retail"
    for cat, pool in MERCHANT_POOLS.items():
        if merchant in pool:
            category = cat.capitalize()
            break

    txn_type = rng.choices(
        ["debit", "debit", "debit", "transfer", "withdrawal"],
        weights=[50, 50, 50, 15, 10],
    )[0]

    return TransactionRecord(
        id=f"TXN_{seq:05d}",
        timestamp=_iso(dt),
        amount=amount,
        currency="USD",
        account_id=account_id,
        counterparty_id=merchant.replace(" ", "_") + "_" + str(rng.randint(1, 99)),
        merchant=merchant,
        merchant_category=category,
        country=profile.country,
        transaction_type=txn_type,
        channel=rng.choice(["online", "pos", "pos", "atm"]),
        description=f"Purchase at {merchant}",
    )


class TransactionGenerator:

    TASK_INSTRUCTIONS = {
        "duplicate_detection": (
            "You are a fraud analyst reviewing a transaction batch. "
            "Identify ALL duplicate or near-duplicate charges. "
            "A duplicate is: same account_id + same merchant + "
            "same amount (within $0.01) + within 24 hours of another transaction. "
            "Add each duplicate transaction ID (NOT the original) to flagged_ids. "
            "Set anomaly_types to ['duplicate'] for each flagged transaction. "
            "Write a brief incident report in report_text. "
            "Precision matters — false positives reduce your score."
        ),
        "pattern_fraud": (
            "You are a fraud analyst. This batch may contain multiple active "
            "fraud patterns. For each suspicious transaction, add its ID to "
            "flagged_ids and the matching pattern to anomaly_types (same index). "
            "Valid pattern labels: velocity_attack | round_number | "
            "unusual_merchant | after_hours. "
            "In report_text, describe every pattern you found and why. "
            "You receive feedback after each attempt — use it to improve. "
            "Not every pattern will be present in every episode."
        ),
        "laundering_chain": (
            "You are a BSA/AML compliance officer. This transaction batch "
            "contains a multi-hop money laundering chain using structuring "
            "(smurfing below $10,000 CTR threshold). "
            "Identify ALL transactions in the chain. Flag them in flagged_ids "
            "with anomaly_types 'structuring' (Phase 1) or 'laundering' (Phase 2+3). "
            "In report_text write a full SAR-style report: describe the account "
            "chain (A->B->C->D), transaction amounts and dates, structuring pattern, "
            "total funds moved, and recommend disposition (block/review). "
            "Minimum 150 words. Use your account_profiles to identify unusual "
            "cross-account flows."
        ),
    }

    def generate(
        self,
        task_id: str,
        seed: int,
    ) -> Tuple[List[TransactionRecord], Dict[str, AccountProfile], dict]:
        """
        Returns (transactions, account_profiles, ground_truth).
        ground_truth stays server-side, never sent to agent.
        """
        rng = random.Random(seed)
        np.random.seed(seed % (2**31))

        # 1. Generate account pool
        n_accounts = rng.randint(6, 12)
        accounts = generate_account_pool(n_accounts, rng)

        base_date = datetime(2024, 1, 1)
        seq_counter = [0]
        transactions = []

        # 2. Generate baseline transactions for each account
        for account_id, profile in accounts.items():
            n_txns = max(2, int(rng.gauss(4, 1.5)))
            for _ in range(n_txns):
                seq_counter[0] += 1
                txn = generate_baseline_transaction(
                    account_id, profile, seq_counter[0], rng, base_date
                )
                transactions.append(txn)

        # 3. Inject fraud patterns based on task
        ground_truth = {
            "fraud_map": {},
            "chain_accounts": [],
            "total_fraud": 0,
            "total_clean": 0,
        }

        if task_id == "duplicate_detection":
            inj = DuplicateInjector()
            ids, labels = inj.inject(transactions, accounts, rng, seq_counter)
            for tid, lbl in zip(ids, labels):
                ground_truth["fraud_map"][tid] = lbl

        elif task_id == "pattern_fraud":
            injectors = [
                VelocityAttackInjector(),
                RoundNumberInjector(),
                UnusualMerchantInjector(),
                AfterHoursInjector(),
            ]
            # Each pattern active with 0.7 probability (at least 2 active)
            active = [inj for inj in injectors if rng.random() < 0.7]
            if len(active) < 2:
                active = rng.sample(injectors, 2)
            for inj in active:
                ids, labels = inj.inject(
                    transactions, accounts, rng, seq_counter
                )
                for tid, lbl in zip(ids, labels):
                    ground_truth["fraud_map"][tid] = lbl

        elif task_id == "laundering_chain":
            inj = LaunderingChainInjector()
            result = inj.inject(transactions, accounts, rng, seq_counter)
            ids, labels, chain = result
            for tid, lbl in zip(ids, labels):
                ground_truth["fraud_map"][tid] = lbl
            ground_truth["chain_accounts"] = chain
            # Add 1-2 noise patterns (makes it harder)
            if rng.random() < 0.4:
                noise_inj = rng.choice(
                    [VelocityAttackInjector(), RoundNumberInjector()]
                )
                n_ids, n_labels = noise_inj.inject(
                    transactions, accounts, rng, seq_counter
                )
                for tid, lbl in zip(n_ids, n_labels):
                    ground_truth["fraud_map"][tid] = lbl

        # 4. Sort transactions by timestamp
        transactions.sort(key=lambda t: t.timestamp)

        # 5. Compute stats
        fraud_ids = set(ground_truth["fraud_map"].keys())
        ground_truth["total_fraud"] = len(fraud_ids)
        ground_truth["total_clean"] = len(transactions) - len(fraud_ids)

        return transactions, accounts, ground_truth
