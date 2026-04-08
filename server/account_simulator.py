"""
Generates statistically realistic account behavioral profiles.
Each account has a personality — spending patterns, typical hours,
merchant preferences — drawn from distributions, not hardcoded.
"""

import random
import numpy as np
from server.models import AccountProfile

MERCHANT_POOLS = {
    "retail":     ["Amazon", "Walmart", "Target", "BestBuy", "HomeDepot", "Costco", "IKEA"],
    "food":       ["McDonalds", "Starbucks", "Chipotle", "DoorDash", "UberEats", "Subway"],
    "travel":     ["Delta Airlines", "United Hotels", "Airbnb", "Uber", "Lyft", "Hertz"],
    "utilities":  ["ConEd Electric", "AT&T", "Comcast", "Verizon", "Water Dept", "Gas Co"],
    "healthcare": ["CVS Pharmacy", "Walgreens", "LabCorp", "Kaiser", "Blue Cross"],
    "financial":  ["Chase ATM", "BofA Wire", "PayPal", "Venmo", "Zelle", "CashApp"],
    "gambling":   ["BetMGM", "DraftKings", "FanDuel", "Caesars Online", "PokerStars"],
    "crypto":     ["Coinbase", "Binance", "Kraken", "Gemini", "LocalBitcoins"],
    "luxury":     ["Tiffany", "Rolex", "Louis Vuitton", "Sothebys", "Ferrari Dealer"],
}

ACCOUNT_TYPES = ["checking", "savings", "business", "money_market"]


def generate_account_pool(n_accounts: int, rng: random.Random) -> dict:
    """
    Generate n_accounts with realistic behavioral profiles.
    Returns dict keyed by account_id.
    """
    profiles = {}
    for i in range(n_accounts):
        account_id = f"ACC_{1000 + i:04d}"
        acct_type  = rng.choice(ACCOUNT_TYPES)

        # Each account has a normal spending range
        if acct_type == "business":
            avg_monthly = rng.uniform(5000, 50000)
            typical_hrs = "business"
        elif acct_type == "money_market":
            avg_monthly = rng.uniform(1000, 8000)
            typical_hrs = "business"
        else:
            avg_monthly = rng.uniform(500, 5000)
            typical_hrs = rng.choice(["evening", "anytime", "business"])

        # Each account specializes in 2-4 merchant categories
        n_cats = rng.randint(2, 4)
        base_categories = rng.sample(
            ["retail", "food", "travel", "utilities", "healthcare", "financial"], n_cats
        )
        typical_merchants = []
        for cat in base_categories:
            # Sample min(2, pool_size) merchants to avoid errors on small pools
            pool = MERCHANT_POOLS[cat]
            n_sample = min(2, len(pool))
            typical_merchants.extend(rng.sample(pool, n_sample))

        # Ensure at least 2 typical merchants (safety guard)
        if len(typical_merchants) < 2:
            fallback_pool = MERCHANT_POOLS["retail"]
            while len(typical_merchants) < 2:
                m = rng.choice(fallback_pool)
                if m not in typical_merchants:
                    typical_merchants.append(m)

        profiles[account_id] = AccountProfile(
            account_id=account_id,
            account_type=acct_type,
            age_days=rng.randint(30, 3650),
            avg_monthly_txn=round(avg_monthly, 2),
            typical_merchants=typical_merchants,
            typical_hours=typical_hrs,
            country=rng.choice(["US", "US", "US", "US", "CA", "GB", "DE"]),
            risk_score=round(rng.uniform(0.0, 0.3), 3),  # most are low risk
        )
    return profiles
