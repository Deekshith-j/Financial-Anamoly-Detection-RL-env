from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum


class AnomalyType(str, Enum):
    DUPLICATE        = "duplicate"
    VELOCITY_ATTACK  = "velocity_attack"
    ROUND_NUMBER     = "round_number"
    UNUSUAL_MERCHANT = "unusual_merchant"
    AFTER_HOURS      = "after_hours"
    LAUNDERING       = "laundering"
    STRUCTURING      = "structuring"
    NONE             = "none"


class Disposition(str, Enum):
    BLOCK   = "block"
    REVIEW  = "review"
    APPROVE = "approve"


class TransactionRecord(BaseModel):
    id: str
    timestamp: str           # ISO 8601
    amount: float
    currency: str
    account_id: str
    counterparty_id: str
    merchant: str
    merchant_category: str
    country: str
    transaction_type: str    # debit | credit | transfer | withdrawal
    channel: str             # online | atm | pos | wire
    description: str
    # NOTE: NO label field — agent cannot see ground truth


class AccountProfile(BaseModel):
    account_id: str
    account_type: str        # checking | savings | business | money_market
    age_days: int            # how old the account is
    avg_monthly_txn: float   # normal spending baseline
    typical_merchants: List[str]
    typical_hours: str       # "business" | "evening" | "anytime"
    country: str
    risk_score: float        # 0.0-1.0 internal risk (NOT shown to agent)


class FinancialObservation(BaseModel):
    transactions: List[TransactionRecord]
    account_profiles: Dict[str, AccountProfile]
    task_instruction: str
    step_number: int
    feedback: str = ""
    task_id: str
    episode_id: str          # unique per reset(), e.g. uuid4 hex
    batch_stats: Dict        # aggregate stats: total_amount, n_accounts, etc.


class FinancialAction(BaseModel):
    flagged_ids: List[str] = Field(default_factory=list)
    anomaly_types: List[str] = Field(default_factory=list)
    report_text: str = ""
    disposition: str = "review"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class StepResult(BaseModel):
    observation: FinancialObservation
    reward: float
    done: bool
    info: Dict[str, Any]


class ResetResult(BaseModel):
    observation: FinancialObservation
