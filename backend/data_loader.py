# data_loading.py
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Dict

import boto3
import botocore
import pandas as pd
from io import BytesIO
import config

# ---- Logging ----
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ---- S3 client (cached) ----
@lru_cache(maxsize=1)
def _s3_client():
    return boto3.client("s3", region_name=getattr(config, "AWS_REGION", None))

def _read_csv_bytes(data: bytes, **read_csv_kwargs) -> pd.DataFrame:
    """
    Safer CSV read: handles BOM, disables low_memory (better type inference),
    and allows caller overrides.
    """
    defaults = dict(encoding="utf-8-sig", low_memory=False)
    defaults.update(read_csv_kwargs or {})
    return pd.read_csv(BytesIO(data), **defaults)

def load_csv_from_s3(
    key: str,
    *,
    bucket: Optional[str] = None,
    local_fallback: Optional[Path | str] = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """
    Try S3 first; if it fails and a local fallback is provided, load that.
    """
    bucket = bucket or getattr(config, "S3_BUCKET", None)
    if not bucket:
        raise ValueError("S3 bucket not provided and config.S3_BUCKET is missing.")

    s3 = _s3_client()
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
        body = resp["Body"].read()
        df = _read_csv_bytes(body, **read_csv_kwargs)
        logger.info("✅ Loaded s3://%s/%s", bucket, key)
        return df

    except botocore.exceptions.ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        msg = e.response.get("Error", {}).get("Message", str(e))
        logger.warning("S3 ClientError (%s) for s3://%s/%s: %s", code, bucket, key, msg)
        if local_fallback:
            logger.info("↪ Falling back to local: %s", local_fallback)
            return pd.read_csv(local_fallback, **read_csv_kwargs)
        raise

    except Exception as e:
        logger.warning("S3 read failed for s3://%s/%s: %s", bucket, key, e)
        if local_fallback:
            logger.info("↪ Falling back to local: %s", local_fallback)
            return pd.read_csv(local_fallback, **read_csv_kwargs)
        raise

def load_data(
    *,
    local_dir: Path | str = None,
    read_csv_kwargs: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads prices, balance sheets, and ESG rankings as DataFrames.

    Expects these keys in config.S3_KEYS:
      - "historical_prices"
      - "balance_sheets"
      - "esg_rankings"

    For local fallbacks, looks under local_dir (if provided) for files named:
      - historical_prices.csv
      - balance_sheets.csv
      - esg_rankings.csv
    """
    if not hasattr(config, "S3_KEYS"):
        raise AttributeError("config.S3_KEYS is missing. Provide the S3 key mapping.")

    read_csv_kwargs = read_csv_kwargs or {}

    # Resolve local dir
    base = Path(local_dir) if local_dir else Path.cwd()
    local_prices = base / "data" / "historical_prices.csv"
    local_balance = base / "data" / "balance_sheets.csv"
    local_esg = base / "data" / "esg_rankings.csv"

    prices = load_csv_from_s3(
        config.S3_KEYS["historical_prices"],
        local_fallback=local_prices,
        **read_csv_kwargs,
    )
    balance = load_csv_from_s3(
        config.S3_KEYS["balance_sheets"],
        local_fallback=local_balance,
        **read_csv_kwargs,
    )
    esg = load_csv_from_s3(
        config.S3_KEYS["esg_rankings"],
        local_fallback=local_esg,
        **read_csv_kwargs,
    )

    return prices, balance, esg

if __name__ == "__main__":
    # Example: set a default local_dir to your OneDrive repo root if desired
    # local_root = r"C:\Users\shrav\OneDrive\Desktop\genai_green_genie"
    local_root = Path(__file__).resolve().parent  # or customize

    df_prices, df_balance, df_esg = load_data(
        local_dir=local_root,
        read_csv_kwargs=dict(dtype=None)  # you can pass dtype, nrows, etc.
    )

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    print("✅ Prices:\n", df_prices.head(), "\n")
    print("✅ Balance Sheets:\n", df_balance.head(), "\n")
    print("✅ ESG Rankings:\n", df_esg.head())
