from __future__ import annotations
import pandas as pd

def generate_recommendation(sector, risk, prices, balance, esg, *, top_n=5, random_state=42) -> pd.DataFrame:
    def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    if esg is None or esg.empty:
        return pd.DataFrame()

    df = esg.copy()

    # --- normalize columns ---
    company_col = pick_col(df, ["Company","Company Name","Stock","Ticker","Symbol","Name"])
    esg_col = pick_col(df, ["ESG Score","esg_score","esg","score"])
    sector_col = pick_col(df, ["Sector","sector","Industry","industry"])

    if company_col and company_col != "Company":
        df.rename(columns={company_col: "Company"}, inplace=True)
    elif "Company" not in df.columns:
        df.insert(0, "Company", df.index.astype(str))

    if esg_col and esg_col != "ESG Score":
        df.rename(columns={esg_col: "ESG Score"}, inplace=True)
    if "ESG Score" in df.columns:
        df["ESG Score"] = pd.to_numeric(df["ESG Score"], errors="coerce")

    if sector_col and sector_col != "Sector":
        df.rename(columns={sector_col: "Sector"}, inplace=True)
    if "Sector" not in df.columns:
        df["Sector"] = "Unknown"   # ✅ fallback so Sector always exists

    # --- filter by sector if selected ---
    if sector and str(sector).lower() != "all":
        df = df[df["Sector"].astype(str) == str(sector)]

    if df.empty:
        return pd.DataFrame()

    # --- risk logic ---
    r = (risk or "").lower()
    if r == "low" and "ESG Score" in df.columns:
        out = df.sort_values("ESG Score", ascending=False).head(top_n)
    elif r == "medium" and "ESG Score" in df.columns:
        sorted_df = df.sort_values("ESG Score", ascending=False)
        out = sorted_df.head(max(1, len(sorted_df)//2))
    elif r == "high":
        out = df.sample(n=min(top_n, len(df)), random_state=random_state)
    else:
        out = df.sample(n=min(top_n, len(df)), random_state=random_state)

    return out.reset_index(drop=True)
