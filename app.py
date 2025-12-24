
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px

from backend.aws_bedrock import query_bedrock
from backend.data_loader import load_data
from backend.recommender_lib import generate_recommendation
import config


# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(page_title="GenAI Green Genie", page_icon="🌱", layout="wide")

# Minimal, classy CSS polish
st.markdown(
    """
    <style>
      .app-title { font-size: 44px; font-weight: 800; letter-spacing: -0.3px; }
      .app-subtitle { font-size: 20px; opacity: 0.85; margin-top: -8px; }
      .section-title { font-size: 22px; font-weight: 700; margin: 24px 0 8px; }
      .soft-card { border-radius: 16px; padding: 16px; border: 1px solid rgba(255,255,255,0.08); background: rgba(255,255,255,0.02); }
      .spacer-8 { height: 8px; }
      .spacer-16 { height: 16px; }
      .spacer-24 { height: 24px; }
      .stButton > button { border-radius: 12px; padding: 10px 18px; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">GenAI Green Genie 💹🌱</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">AI-driven Personalized Investment Recommendations</div>', unsafe_allow_html=True)
st.markdown('<div class="spacer-16"></div>', unsafe_allow_html=True)


# -------------------------------
# Helpers
# -------------------------------
@st.cache_data(show_spinner=False)
def _load_data_cached():
    """Load prices, balance, esg once and cache."""
    return load_data()

def _pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _normalize_columns_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure canonical columns: Company, Sector, ESG Score (when available)."""
    df = df.copy()

    company_col = _pick_first_existing(df, ["Company", "Company Name", "Stock", "Ticker", "Symbol", "Name"])
    if company_col and company_col != "Company":
        df.rename(columns={company_col: "Company"}, inplace=True)
    elif "Company" not in df.columns:
        df.insert(0, "Company", df.index.astype(str))

    sector_col = _pick_first_existing(df, ["Sector", "sector", "Industry", "industry"])
    if sector_col and sector_col != "Sector":
        df.rename(columns={sector_col: "Sector"}, inplace=True)
    if "Sector" not in df.columns:
        df["Sector"] = "Unknown"

    esg_col = _pick_first_existing(df, ["ESG Score", "esg_score", "esg", "score"])
    if esg_col and esg_col != "ESG Score":
        df.rename(columns={esg_col: "ESG Score"}, inplace=True)
    if "ESG Score" in df.columns:
        df["ESG Score"] = pd.to_numeric(df["ESG Score"], errors="coerce")

    return df

def _build_explanation_prompt(user_input: str, sector: str, risk: str, recs_display: pd.DataFrame) -> str:
    companies = ", ".join(recs_display["Company"].astype(str).tolist()) if "Company" in recs_display.columns else "N/A"
    esg_present = "ESG Score" in recs_display.columns
    esg_hint = " (prioritizing higher ESG scores)" if esg_present and risk in {"Low", "Medium"} else ""

    return f"""
You are an investment assistant. Explain the recommendations clearly in non-technical language.

Investor notes: {user_input or "N/A"}
Sector: {sector}
Risk tolerance: {risk}{esg_hint}
Recommended Companies: {companies}

Keep it brief (120-180 words), avoid guarantees, and emphasize diversification and due diligence.
"""


# -------------------------------
# Data load + sector options
# -------------------------------
try:
    prices, balance, esg = _load_data_cached()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

SECTOR_CANDIDATES = ["Sector", "sector", "Industry", "industry"]
sector_col = next((c for c in SECTOR_CANDIDATES if c in esg.columns), None)
if sector_col:
    sector_options = ["All"] + sorted(esg[sector_col].dropna().astype(str).unique().tolist())
else:
    # Fallbacks from config
    if isinstance(getattr(config, "SECTORS", []), list) and config.SECTORS:
        sector_options = ["All"] + sorted(set(map(str, config.SECTORS)))
    elif isinstance(getattr(config, "SYMBOL_TO_SECTOR", {}), dict) and config.SYMBOL_TO_SECTOR:
        sector_options = ["All"] + sorted(set(config.SYMBOL_TO_SECTOR.values()))
    else:
        sector_options = ["All"]


# -------------------------------
# Controls (on main page)
# -------------------------------
st.markdown('<div class="section-title">Input</div>', unsafe_allow_html=True)
with st.container():
    c1, c2, c3 = st.columns([1.2, 1.2, 2.4])
    with c1:
        sector = st.selectbox("Select Investment Sector", options=sector_options)
    with c2:
        risk = st.radio("Select Risk Level", options=getattr(config, "RISK_LEVELS", ["Low", "Medium", "High"]), horizontal=True)
    with c3:
        user_input = st.text_input(
            "Describe your investment goals (optional)",
            placeholder="e.g., steady growth, long-term horizon, low drawdowns",
        )

st.markdown('<div class="spacer-8"></div>', unsafe_allow_html=True)
get_recs = st.button("Get Recommendations", type="primary")
st.markdown('<div class="spacer-16"></div>', unsafe_allow_html=True)


# -------------------------------
# Main Action
# -------------------------------
if get_recs:
    with st.spinner("⏳ Generating your recommendations..."):
        try:
            recs = generate_recommendation(sector, risk, prices, balance, esg)
        except Exception as e:
            st.error(f"Failed to generate recommendations: {e}")
            st.stop()

        if recs is None or recs.empty:
            st.warning("No recommendations found for the selected inputs. Try a different sector or risk level.")
            st.stop()

        # Normalize for display and hide index
        recs_display = _normalize_columns_for_display(recs)

        st.markdown('<div class="section-title">📊 Top Recommended Stocks</div>', unsafe_allow_html=True)

        # If ESG Score exists, show styled with index hidden
        if "ESG Score" in recs_display.columns:
            try:
                styled = (
                    recs_display
                    .style
                    .hide(axis="index")  # hide serial/index column
                    .background_gradient(cmap="viridis", subset=["ESG Score"])
                    .format({"ESG Score": "{:.0f}"})
                )
                st.dataframe(styled, width="stretch")
            except Exception:
                st.dataframe(recs_display, width="stretch", hide_index=True)
        else:
            st.dataframe(recs_display, width="stretch", hide_index=True)

        # Chart with a modern palette (Viridis) and no container warning
        if {"Company", "ESG Score"}.issubset(recs_display.columns):
            fig = px.bar(
                recs_display,
                x="Company",
                y="ESG Score",
                color="ESG Score",
                color_continuous_scale="viridis",
                text="ESG Score",
            )
            fig.update_layout(
                xaxis_title="",
                yaxis_title="ESG Score",
                margin=dict(l=10, r=10, t=10, b=10),
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, width="stretch")

        # AI explanation
        st.markdown('<div class="section-title">🧠 Why these picks?</div>', unsafe_allow_html=True)
        prompt = _build_explanation_prompt(user_input, sector, risk, recs_display)
        try:
            explanation = query_bedrock(prompt)  # should return a string
        except Exception as e:
            explanation = f"Error while calling Bedrock: {e}"
        st.write(explanation)
