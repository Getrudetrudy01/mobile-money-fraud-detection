from typing import Literal

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    """A transaction to score. Only information available BEFORE approval."""
    step: int = Field(ge=1, description="Hour index since the start of the simulation")
    type: Literal["TRANSFER", "CASH_OUT"]
    amount: float = Field(gt=0)
    oldbalanceOrg: float = Field(ge=0, description="Sender balance before the transaction")
    oldbalanceDest: float = Field(ge=0, description="Receiver balance before the transaction")
    # Receiver history (would come from a feature store in production)
    dest_prior_txn_count: int | None = Field(default=None, ge=0)
    dest_prior_amount_sum: float | None = Field(default=None, ge=0)
    dest_hours_since_last: float | None = Field(default=None, ge=0)


class Factor(BaseModel):
    feature: str
    contribution: float


class Prediction(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float
    model_version: str
    top_factors: list[Factor]