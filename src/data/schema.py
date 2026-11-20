import pandera.pandas as pa

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# Memory-efficient types used when reading the raw CSV
RAW_DTYPES = {
    "step": "int16", "type": "category", "amount": "float32",
    "nameOrig": "string", "oldbalanceOrg": "float32", "newbalanceOrig": "float32",
    "nameDest": "string", "oldbalanceDest": "float32", "newbalanceDest": "float32",
    "isFraud": "int8", "isFlaggedFraud": "int8",
}

non_negative = pa.Check.ge(0)

raw_schema = pa.DataFrameSchema(
    {
        "step": pa.Column("int16", pa.Check.in_range(1, 743)),
        "type": pa.Column("category", pa.Check.isin(TRANSACTION_TYPES)),
        "amount": pa.Column("float32", non_negative),
        "nameOrig": pa.Column(checks=pa.Check.str_startswith("C"), nullable=False),
        "oldbalanceOrg": pa.Column("float32", non_negative),
        "newbalanceOrig": pa.Column("float32", non_negative),
        "nameDest": pa.Column(checks=pa.Check.str_matches(r"^[CM]"), nullable=False),
        "oldbalanceDest": pa.Column("float32", non_negative),
        "newbalanceDest": pa.Column("float32", non_negative),
        "isFraud": pa.Column("int8", pa.Check.isin([0, 1])),
        "isFlaggedFraud": pa.Column("int8", pa.Check.isin([0, 1])),
    },
    strict=True,  # fail if unexpected columns appear
)