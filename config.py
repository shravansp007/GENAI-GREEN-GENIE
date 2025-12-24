# config.py
from __future__ import annotations
import os
from pathlib import Path

# -----------------------------
# AWS / Bedrock
# -----------------------------
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "genai-green-genie-datasets")

# Claude model ID — make sure it matches what’s enabled in your AWS account
BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "anthropic.claude-3-sonnet-20240229-v1:0"
)

# -----------------------------
# Datasets (S3 keys & local fallbacks)
# -----------------------------
# Keys must match filenames you uploaded to S3
S3_KEYS = {
    "historical_prices": os.getenv("S3_KEY_HIST_PRICES", "historical_prices.csv"),
    "balance_sheets": os.getenv("S3_KEY_BALANCE", "balance_sheets.csv"),
    "esg_rankings": os.getenv("S3_KEY_ESG", "esg_rankings.csv"),
}

# Local fallback paths (used if S3 fetch fails)
# Tip: change BASE_DIR if your project lives elsewhere
BASE_DIR = Path(
    os.getenv("LOCAL_BASE_DIR", Path.home() / "OneDrive" / "Desktop" / "genai_green_genie")
)
LOCAL_DATA = {
    "historical_prices": str(BASE_DIR / "data" / "historical_prices.csv"),
    "balance_sheets": str(BASE_DIR / "data" / "balance_sheets.csv"),
    "esg_rankings": str(BASE_DIR / "data" / "esg_rankings.csv"),
}

# -----------------------------
# UI options
# -----------------------------
# Mapping of stock symbols to their sectors
SECTORS = {
     "Infrastructure",
     "Chemical",
     "Banking",
     "Automobile",
     "Financial Services",
     "Telecommunications",
     "Energy",
     "Consumer Goods",
     "Pharmaceuticals",
     "Mining",
     "Cement",
     "Information Technology",
     "Metals",
     "Media",
}

# Order as shown in the UI (left→right for st.radio)
RISK_LEVELS = ["Low", "Medium", "High"]

# -----------------------------
# Optional: quick config sanity check
# -----------------------------
def validate_config() -> None:
    missing = []
    if not AWS_REGION:
        missing.append("AWS_REGION")
    if not S3_BUCKET:
        missing.append("S3_BUCKET")
    for k, v in S3_KEYS.items():
        if not v:
            missing.append(f"S3_KEYS['{k}']")
    if missing:
        raise ValueError(f"Missing required config values: {', '.join(missing)}")
