"""
Injects stochastic fraud patterns into a clean transaction batch.
Returns (modified_transactions, ground_truth_labels).
Ground truth is ONLY stored server-side, never sent to agent.

Each pattern has:
  - inject(transactions, accounts, rng, seq_counter) -> (ids, labels) or (ids, labels, chain)
  - label_fn -> labels the injected transactions
"""

import random
import copy
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from server.models import TransactionRecord


def _iso(dt: datetime) -> str:
    return dt.isoformat() + "Z"


def _txn_id(prefix: str, seq: int) -> str:
    return f"{prefix}_{seq:05d}"


class DuplicateInjector:
    """
    Easy pattern. Picks N random legitimate transactions and creates
    near-copies: same account, same merchant, same amount, offset
    by a random delta between 30 minutes and 23 hours.
    n_duplicates drawn from Gaussian(mu=3, sigma=1), min 2, max 6.
    """
    name = "duplicate"

    def inject(
        self,
        transactions: List[TransactionRecord],
        accounts: dict,
        rng: random.Random,
        seq_counter: list,
    ) -> Tuple[List[str], List[str]]:
        n = max(2, min(6, int(rng.gauss(3, 1))))
        candidates = [t for t in transactions if t.transaction_type == "debit"]
        if len(candidates) < n:
            n = len(candidates)
        if n == 0:
            return [], []
        originals = rng.sample(candidates, n)
        injected_ids = []
        for orig in originals:
            seq_counter[0] += 1
            dup_id = _txn_id("TXN", seq_counter[0])
            delta_hours = rng.uniform(0.5, 23)
            orig_dt = datetime.fromisoformat(orig.timestamp.replace("Z", ""))
            new_dt = orig_dt + timedelta(hours=delta_hours)
            dup = TransactionRecord(
                id=dup_id,
                timestamp=_iso(new_dt),
                amount=orig.amount,
                currency=orig.currency,
                account_id=orig.account_id,
                counterparty_id=orig.counterparty_id,
                merchant=orig.merchant,
                merchant_category=orig.merchant_category,
                country=orig.country,
                transaction_type=orig.transaction_type,
                channel=orig.channel,
                description=orig.description + " [reprocessed]",
            )
            transactions.append(dup)
            injected_ids.append(dup_id)
        return injected_ids, ["duplicate"] * len(injected_ids)


class VelocityAttackInjector:
    """
    Medium pattern. Picks one account, injects M small transactions
    at the same merchant within a 5-minute window.
    M ~ Gaussian(4, 1), min 3, max 7. Amounts: uniform $1-$9.99.
    """
    name = "velocity_attack"

    def inject(
        self,
        transactions: List[TransactionRecord],
        accounts: dict,
        rng: random.Random,
        seq_counter: list,
    ) -> Tuple[List[str], List[str]]:
        account_id = rng.choice(list(accounts.keys()))
        m = max(3, min(7, int(rng.gauss(4, 1))))
        merchant = "RapidCharge_" + str(rng.randint(100, 999))
        base_dt = datetime(
            2024,
            rng.randint(1, 12),
            rng.randint(1, 28),
            rng.randint(0, 23),
            rng.randint(0, 55),
        )
        injected_ids = []
        for i in range(m):
            seq_counter[0] += 1
            t_id = _txn_id("TXN", seq_counter[0])
            dt = base_dt + timedelta(seconds=rng.randint(0, 300))
            transactions.append(
                TransactionRecord(
                    id=t_id,
                    timestamp=_iso(dt),
                    amount=round(rng.uniform(1.0, 9.99), 2),
                    currency="USD",
                    account_id=account_id,
                    counterparty_id=merchant,
                    merchant=merchant,
                    merchant_category="Online Services",
                    country="US",
                    transaction_type="debit",
                    channel="online",
                    description="micro-charge",
                )
            )
            injected_ids.append(t_id)
        return injected_ids, ["velocity_attack"] * m


class RoundNumberInjector:
    """
    Medium pattern. Injects K transactions of exactly $500, $1000,
    $5000, or $9900 from one account to "ATM_CASH" merchants.
    K ~ Gaussian(2, 0.7), min 2, max 4.
    """
    name = "round_number"

    def inject(
        self,
        transactions: List[TransactionRecord],
        accounts: dict,
        rng: random.Random,
        seq_counter: list,
    ) -> Tuple[List[str], List[str]]:
        account_id = rng.choice(list(accounts.keys()))
        k = max(2, min(4, int(rng.gauss(2, 0.7))))
        amounts = rng.choices([500.0, 1000.0, 2500.0, 5000.0, 9900.0], k=k)
        injected_ids = []
        base_dt = datetime(2024, rng.randint(1, 12), rng.randint(1, 28), 10, 0)
        for i, amt in enumerate(amounts):
            seq_counter[0] += 1
            t_id = _txn_id("TXN", seq_counter[0])
            dt = base_dt + timedelta(days=rng.randint(0, 5))
            transactions.append(
                TransactionRecord(
                    id=t_id,
                    timestamp=_iso(dt),
                    amount=amt,
                    currency="USD",
                    account_id=account_id,
                    counterparty_id="CASH_POINT_ATM",
                    merchant="CashPoint ATM",
                    merchant_category="ATM Withdrawal",
                    country="US",
                    transaction_type="withdrawal",
                    channel="atm",
                    description=f"Cash withdrawal ${amt:.0f}",
                )
            )
            injected_ids.append(t_id)
        return injected_ids, ["round_number"] * k


class UnusualMerchantInjector:
    """
    Medium pattern. Takes an account whose typical_merchants contain
    only normal retail/food/utilities and injects 1-2 transactions
    at a gambling or crypto merchant.
    """
    name = "unusual_merchant"

    def inject(
        self,
        transactions: List[TransactionRecord],
        accounts: dict,
        rng: random.Random,
        seq_counter: list,
    ) -> Tuple[List[str], List[str]]:
        from server.account_simulator import MERCHANT_POOLS

        # Find account with no gambling/crypto history
        suspicious_keywords = set(
            MERCHANT_POOLS["gambling"] + MERCHANT_POOLS["crypto"] + MERCHANT_POOLS["luxury"]
        )
        normal_accounts = [
            aid
            for aid, prof in accounts.items()
            if not any(m in suspicious_keywords for m in prof.typical_merchants)
        ]
        account_id = rng.choice(
            normal_accounts if normal_accounts else list(accounts.keys())
        )

        pool_name = rng.choice(["gambling", "crypto"])
        merchant = rng.choice(MERCHANT_POOLS[pool_name])
        injected_ids = []
        count = rng.randint(1, 2)
        for i in range(count):
            seq_counter[0] += 1
            t_id = _txn_id("TXN", seq_counter[0])
            dt = datetime(
                2024,
                rng.randint(1, 12),
                rng.randint(1, 28),
                rng.randint(0, 23),
                rng.randint(0, 59),
            )
            transactions.append(
                TransactionRecord(
                    id=t_id,
                    timestamp=_iso(dt),
                    amount=round(rng.uniform(50, 2000), 2),
                    currency="USD",
                    account_id=account_id,
                    counterparty_id=merchant.replace(" ", "_"),
                    merchant=merchant,
                    merchant_category=pool_name.capitalize(),
                    country=rng.choice(["US", "MT", "CY"]),
                    transaction_type="debit",
                    channel="online",
                    description=f"Payment to {merchant}",
                )
            )
            injected_ids.append(t_id)
        return injected_ids, ["unusual_merchant"] * len(injected_ids)


class AfterHoursInjector:
    """
    Medium pattern. Injects transactions between 01:00-04:59 AM
    for accounts whose typical_hours == "business" or "evening".
    """
    name = "after_hours"

    def inject(
        self,
        transactions: List[TransactionRecord],
        accounts: dict,
        rng: random.Random,
        seq_counter: list,
    ) -> Tuple[List[str], List[str]]:
        candidates = [
            aid
            for aid, p in accounts.items()
            if p.typical_hours in ("business", "evening")
        ]
        account_id = rng.choice(
            candidates if candidates else list(accounts.keys())
        )
        injected_ids = []
        count = rng.randint(1, 3)
        for _ in range(count):
            seq_counter[0] += 1
            t_id = _txn_id("TXN", seq_counter[0])
            hour = rng.randint(1, 4)
            minute = rng.randint(0, 59)
            dt = datetime(
                2024,
                rng.randint(1, 12),
                rng.randint(1, 28),
                hour,
                minute,
            )
            transactions.append(
                TransactionRecord(
                    id=t_id,
                    timestamp=_iso(dt),
                    amount=round(rng.uniform(100, 3000), 2),
                    currency="USD",
                    account_id=account_id,
                    counterparty_id="MerchantX_" + str(rng.randint(10, 99)),
                    merchant="NightOwl Electronics",
                    merchant_category="Electronics",
                    country="US",
                    transaction_type="debit",
                    channel="online",
                    description="Late night purchase",
                )
            )
            injected_ids.append(t_id)
        return injected_ids, ["after_hours"] * len(injected_ids)


class LaunderingChainInjector:
    """
    Hard pattern. Injects a full multi-hop structuring chain:

    Phase 1 - Placement (smurfing):
      Origin account makes N transfers just below $10,000 (CTR threshold)
      to an intermediate account. N ~ Uniform(3,5).
      Amounts: $8,500-$9,800 each, spaced 1-3 days apart.

    Phase 2 - Layering:
      Intermediate account consolidates and moves funds to a second
      intermediate account via 2-3 larger transfers with vague descriptions
      like "consulting fees", "invoice payment", "service retainer".

    Phase 3 - Integration:
      Final account moves money to an international wire transfer
      or luxury goods merchant. 1-2 transactions.

    Chain has 4 accounts: [origin, intermediate_1, intermediate_2, final].
    All chosen randomly from the account pool.
    Total injected transactions: 7-10.
    Returns (ids, labels, chain_accounts).
    """
    name = "laundering"

    def inject(
        self,
        transactions: List[TransactionRecord],
        accounts: dict,
        rng: random.Random,
        seq_counter: list,
    ) -> Tuple[List[str], List[str], List[str]]:
        account_ids = list(accounts.keys())
        if len(account_ids) < 4:
            # Pad with synthetic account IDs if needed (safety)
            return [], [], []

        chain = rng.sample(account_ids, 4)
        origin, inter1, inter2, final = chain

        injected_ids = []
        labels = []
        base_dt = datetime(2024, rng.randint(1, 6), 1)

        # Phase 1: smurfing from origin -> inter1
        n_smurf = rng.randint(3, 5)
        total_placed = 0.0
        for i in range(n_smurf):
            seq_counter[0] += 1
            t_id = _txn_id("TXN", seq_counter[0])
            amount = round(rng.uniform(8500, 9800), 2)
            total_placed += amount
            dt = base_dt + timedelta(days=rng.randint(0, 3) * (i + 1))
            transactions.append(
                TransactionRecord(
                    id=t_id,
                    timestamp=_iso(dt),
                    amount=amount,
                    currency="USD",
                    account_id=origin,
                    counterparty_id=inter1,
                    merchant="Wire Transfer",
                    merchant_category="Transfer",
                    country="US",
                    transaction_type="transfer",
                    channel="wire",
                    description=rng.choice(
                        [
                            "Personal transfer",
                            "Family support",
                            "Loan repayment",
                            "Gift transfer",
                        ]
                    ),
                )
            )
            injected_ids.append(t_id)
            labels.append("structuring")
            base_dt = dt

        # Phase 2: layering inter1 -> inter2
        n_layer = rng.randint(2, 3)
        layer_amount = total_placed / n_layer
        for i in range(n_layer):
            seq_counter[0] += 1
            t_id = _txn_id("TXN", seq_counter[0])
            amount = round(layer_amount * rng.uniform(0.9, 1.1), 2)
            dt = base_dt + timedelta(days=rng.randint(2, 5))
            transactions.append(
                TransactionRecord(
                    id=t_id,
                    timestamp=_iso(dt),
                    amount=amount,
                    currency="USD",
                    account_id=inter1,
                    counterparty_id=inter2,
                    merchant="B2B Payment",
                    merchant_category="Business Services",
                    country=rng.choice(["US", "DE", "NL", "SG"]),
                    transaction_type="transfer",
                    channel="wire",
                    description=rng.choice(
                        [
                            "Consulting fees",
                            "Invoice #" + str(rng.randint(1000, 9999)),
                            "Service retainer",
                            "Contract payment",
                            "Advisory fee",
                        ]
                    ),
                )
            )
            injected_ids.append(t_id)
            labels.append("laundering")
            base_dt = dt

        # Phase 3: integration inter2 -> final
        n_integrate = rng.randint(1, 2)
        for i in range(n_integrate):
            seq_counter[0] += 1
            t_id = _txn_id("TXN", seq_counter[0])
            amount = round(
                total_placed * rng.uniform(0.6, 0.9) / n_integrate, 2
            )
            dt = base_dt + timedelta(days=rng.randint(3, 7))
            transactions.append(
                TransactionRecord(
                    id=t_id,
                    timestamp=_iso(dt),
                    amount=amount,
                    currency="USD",
                    account_id=inter2,
                    counterparty_id=final,
                    merchant=rng.choice(
                        [
                            "INTL Wire Transfer",
                            "Offshore Holdings LLC",
                            "Grand Cayman Trust",
                            "Panama Assets Corp",
                        ]
                    ),
                    merchant_category="International Transfer",
                    country=rng.choice(["KY", "PA", "VG", "LU", "CH"]),
                    transaction_type="wire",
                    channel="wire",
                    description="Investment transfer",
                )
            )
            injected_ids.append(t_id)
            labels.append("laundering")
            base_dt = dt

        return injected_ids, labels, chain
