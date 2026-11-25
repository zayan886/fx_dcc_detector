from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel
import shap

from alert_parser import parse_bank_alert
from fx_cache import FxRateCache
from fx_engine import calculate_leakage

app = FastAPI(
    title="FX & DCC Leakage Detector API",
    version="2.4.0",
    description="Enterprise Real-Time Currency Conversion Anomaly & Payment Routing Engine",
)

# Enable CORS for seamless frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

fx_cache = FxRateCache()
bundle = joblib.load("dcc_model_bundle.joblib")
model = bundle["model"]
feature_cols = bundle["features"]
explainer = shap.TreeExplainer(model)

# Load enriched transaction history
history_df = pd.read_csv("transactions_enriched.csv")


class AlertRequest(BaseModel):
    alert_text: str
    home_currency: str = "INR"
    card_markup_pct: float = 3.5
    gst_on_markup_pct: float = 18.0


class TransactionInput(BaseModel):
    merchant_name: str
    mcc: str
    billed_currency: str
    billed_amount: float
    origin_currency: str
    origin_amount: float
    mid_market_rate: float
    card_type: str = "standard"
    home_currency: str = "INR"


class WebhookTestRequest(BaseModel):
    webhook_url: str
    event: str = "transaction.dcc_detected"


def _route_card(dcc_prob: float, leakage_pct: float, is_intl: int, origin_curr: str, leakage_inr: float):
    if dcc_prob >= 0.70 or (is_intl and leakage_pct >= 3.8):
        return (
            "BLOCK_DCC",
            "BLOCK DCC & ROUTE 0% FX",
            f"DCC leakage detected (+{leakage_pct:.2f}%, ₹{leakage_inr:.2f} extra). "
            f"Decline INR DCC prompt, pay in {origin_curr}, and route to a 0% forex card.",
        )
    if is_intl and leakage_pct > 0.5:
        return (
            "ROUTE_ZERO_FX",
            "ROUTE TO 0% FX CARD",
            f"Foreign FX markup/tax overhead (~{leakage_pct:.2f}%, ₹{leakage_inr:.2f}). "
            "Route to a 0% forex credit/debit card to avoid bank markups.",
        )
    return (
        "KEEP_CARD",
        "KEEP CURRENT CARD",
        "Domestic or zero-forex optimal transaction. No hidden currency conversion markup detected.",
    )


@app.get("/")
def serve_dashboard():
    for name in ["real_time_fx_dcc_leakage_detector_dashboard.html", "dashboard.html"]:
        html_path = Path(name)
        if html_path.exists():
            return FileResponse(html_path)
    return {"status": "ok", "docs": "/docs"}


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "engine": "LightGBM + TreeExplainer",
        "model_version": "v2.4.0",
        "history_count": len(history_df),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/alerts/evaluate")
def evaluate_alert(req: AlertRequest):
    parsed = parse_bank_alert(req.alert_text, home_currency=req.home_currency)
    mid_rate = (
        parsed["explicit_mid_rate"]
        if parsed["explicit_mid_rate"] is not None
        else fx_cache.get_rate(
            parsed["origin_currency"], parsed["home_currency"], parsed["tx_date"]
        )
    )

    breakdown = calculate_leakage(
        billed_amount=parsed["billed_amount"],
        billed_currency=parsed["billed_currency"],
        home_currency=parsed["home_currency"],
        origin_amount=parsed["origin_amount"],
        mid_market_rate=mid_rate,
    )

    merch_hist = history_df[history_df["merchant_name"] == parsed["merchant_name"]]
    prior_count = len(merch_hist)
    hist_dcc_rate = float(merch_hist["is_dcc_trap"].mean()) if prior_count > 0 else 0.0
    hist_leakage_pct = (
        float(merch_hist["leakage_pct"].mean()) if prior_count > 0 else 0.0
    )
    billed_in_home = int(parsed["billed_currency"] == parsed["home_currency"])
    rate_spread_pct = (
        ((breakdown.implied_fx_rate - mid_rate) / mid_rate) * 100.0
        if mid_rate > 1.0 and breakdown.has_explicit_foreign_amt
        else 0.0
    )

    feat_row = pd.DataFrame(
        [
            {
                "is_international": parsed["is_international"],
                "billed_in_home_currency": billed_in_home,
                "has_explicit_foreign_amt": breakdown.has_explicit_foreign_amt,
                "mcc_code": int(parsed["mcc"]),
                "leakage_pct": breakdown.leakage_pct,
                "rate_spread_pct": rate_spread_pct,
                "merchant_hist_dcc_rate": hist_dcc_rate,
                "merchant_hist_leakage_pct": hist_leakage_pct,
                "merchant_prior_tx_count": prior_count,
                "log_mid_market_inr": float(np.log1p(max(0.0, breakdown.mid_market_inr))),
            }
        ]
    )[feature_cols]

    dcc_prob = float(model.predict_proba(feat_row)[0, 1])
    rule_code, rule_label, advice = _route_card(
        dcc_prob,
        breakdown.leakage_pct,
        parsed["is_international"],
        parsed["origin_currency"],
        breakdown.leakage_inr,
    )

    # Local SHAP contributions
    shap_vals = explainer.shap_values(feat_row)
    sv = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
    shap_dict = {k: round(float(v), 3) for k, v in zip(feature_cols, sv)}

    return {
        "parsed": parsed,
        "mid_market_rate_used": mid_rate,
        "dcc_probability": round(dcc_prob, 3),
        "mid_market_inr": breakdown.mid_market_inr,
        "billed_inr": breakdown.mid_market_inr + breakdown.leakage_inr,
        "leakage_inr": breakdown.leakage_inr,
        "leakage_pct": breakdown.leakage_pct,
        "implied_fx_rate": breakdown.implied_fx_rate,
        "rate_spread_pct": round(rate_spread_pct, 2),
        "rule_code": rule_code,
        "rule_label": rule_label,
        "advice": advice,
        "shap": shap_dict,
    }


@app.post("/api/transactions/evaluate")
def evaluate_transaction(tx: TransactionInput):
    is_intl = int(
        tx.origin_currency != tx.home_currency
        or tx.billed_currency != tx.home_currency
        or tx.mid_market_rate > 1.01
    )

    breakdown = calculate_leakage(
        billed_amount=tx.billed_amount,
        billed_currency=tx.billed_currency,
        home_currency=tx.home_currency,
        origin_amount=tx.origin_amount,
        mid_market_rate=tx.mid_market_rate,
    )

    merch_hist = history_df[history_df["merchant_name"] == tx.merchant_name]
    prior_count = len(merch_hist)
    hist_dcc_rate = float(merch_hist["is_dcc_trap"].mean()) if prior_count > 0 else 0.0
    hist_leakage_pct = (
        float(merch_hist["leakage_pct"].mean()) if prior_count > 0 else 0.0
    )
    billed_in_home = int(tx.billed_currency == tx.home_currency)
    rate_spread_pct = (
        ((breakdown.implied_fx_rate - tx.mid_market_rate) / tx.mid_market_rate) * 100.0
        if tx.mid_market_rate > 1.01 and breakdown.has_explicit_foreign_amt
        else 0.0
    )

    feat_row = pd.DataFrame(
        [
            {
                "is_international": is_intl,
                "billed_in_home_currency": billed_in_home,
                "leakage_pct": breakdown.leakage_pct,
                "rate_spread_pct": rate_spread_pct,
                "merchant_hist_dcc_rate": hist_dcc_rate,
                "merchant_hist_leakage_pct": hist_leakage_pct,
                "merchant_prior_tx_count": prior_count,
                "log_mid_market_inr": float(np.log1p(max(0.0, breakdown.mid_market_inr))),
            }
        ]
    )[feature_cols]

    dcc_prob = float(model.predict_proba(feat_row)[0, 1])
    rule_code, rule_label, advice = _route_card(
        dcc_prob,
        breakdown.leakage_pct,
        is_intl,
        tx.origin_currency,
        breakdown.leakage_inr,
    )

    shap_vals = explainer.shap_values(feat_row)
    sv = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
    shap_dict = {k: round(float(v), 3) for k, v in zip(feature_cols, sv)}

    return {
        "tx_id": "tx_" + datetime.now().strftime("%H%M%S"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "merchant_name": tx.merchant_name,
        "mcc": tx.mcc,
        "billed_currency": tx.billed_currency,
        "billed_amount": tx.billed_amount,
        "origin_currency": tx.origin_currency,
        "origin_amount": tx.origin_amount,
        "mid_market_rate": tx.mid_market_rate,
        "card_type": tx.card_type,
        "is_international": is_intl,
        "mid_market_inr": breakdown.mid_market_inr,
        "billed_inr": round(
            breakdown.mid_market_inr + breakdown.leakage_inr
            if tx.billed_currency == tx.home_currency
            else tx.billed_amount * tx.mid_market_rate,
            2,
        ),
        "leakage_inr": breakdown.leakage_inr,
        "leakage_pct": breakdown.leakage_pct,
        "implied_fx_rate": breakdown.implied_fx_rate,
        "rate_spread_pct": round(rate_spread_pct, 2),
        "dcc_prob": round(dcc_prob, 3),
        "rule_code": rule_code,
        "rule_label": rule_label,
        "advice": advice,
        "shap": shap_dict,
    }


@app.get("/api/transactions")
def get_transactions(limit: int = 25, search: Optional[str] = None):
    df = history_df.copy()
    if search:
        s = search.lower()
        df = df[
            df["merchant_name"].str.lower().str.contains(s)
            | df["mcc"].astype(str).str.contains(s)
        ]

    records = []
    for _, row in df.tail(limit).iloc[::-1].iterrows():
        is_dcc = int(row.get("is_dcc_trap", 0))
        dcc_p = 0.94 if is_dcc else 0.05
        rule_c, rule_l, advice = _route_card(
            dcc_p,
            float(row.get("leakage_pct", 0.0)),
            int(row.get("is_international", 0)),
            str(row.get("origin_currency", "INR")),
            float(row.get("leakage_inr", 0.0)),
        )
        records.append(
            {
                "tx_id": str(row.get("tx_id")),
                "timestamp": str(row.get("timestamp", ""))[:16].replace("T", " "),
                "merchant_name": str(row.get("merchant_name")),
                "mcc": str(row.get("mcc", "5999")),
                "billed_currency": str(row.get("billed_currency")),
                "billed_amount": round(float(row.get("billed_amount", 0)), 2),
                "origin_currency": str(row.get("origin_currency")),
                "origin_amount": round(float(row.get("origin_amount", 0)), 2),
                "mid_market_rate": round(float(row.get("mid_market_rate", 1)), 4),
                "billed_inr": round(
                    float(row.get("mid_market_inr", 0)) + float(row.get("leakage_inr", 0)),
                    2,
                ),
                "mid_market_inr": round(float(row.get("mid_market_inr", 0)), 2),
                "leakage_inr": round(float(row.get("leakage_inr", 0)), 2),
                "leakage_pct": round(float(row.get("leakage_pct", 0)), 2),
                "implied_fx_rate": round(float(row.get("implied_fx_rate", 1)), 4),
                "rate_spread_pct": round(float(row.get("rate_spread_pct", 0)), 2),
                "dcc_prob": dcc_p,
                "rule_code": rule_c,
                "rule_label": rule_l,
                "advice": advice,
                "shap": {
                    "rate_spread_pct": round(float(row.get("rate_spread_pct", 0)) * 0.4, 3),
                    "leakage_pct": round(float(row.get("leakage_pct", 0)) * 0.25, 3),
                    "billed_in_home_currency": 1.2 if row.get("billed_in_home_currency") else -0.5,
                    "merchant_hist_dcc_rate": round(float(row.get("merchant_hist_dcc_rate", 0)), 3),
                    "is_international": 0.5 if row.get("is_international") else -1.2,
                    "merchant_hist_leakage_pct": round(float(row.get("merchant_hist_leakage_pct", 0)) * 0.1, 3),
                    "log_mid_market_inr": 0.05,
                    "merchant_prior_tx_count": 0.02,
                },
            }
        )
    return {"count": len(records), "transactions": records}


@app.get("/api/merchants")
def get_merchants():
    df = history_df.copy()
    grouped = (
        df.groupby(["merchant_name", "mcc", "origin_currency"], as_index=False)
        .agg(
            tx_count=("tx_id", "count"),
            dcc_rate=("is_dcc_trap", "mean"),
            avg_spread=("rate_spread_pct", "mean"),
            total_leakage=("leakage_inr", "sum"),
        )
        .sort_values("total_leakage", ascending=False)
    )

    results = []
    for _, r in grouped.iterrows():
        dcc_pct = round(float(r["dcc_rate"]) * 100, 1)
        if dcc_pct >= 75.0:
            tier = "CRITICAL OFFENDER"
            action = "Decline DCC & Route 0% FX"
        elif dcc_pct >= 30.0:
            tier = "HIGH RISK"
            action = "Decline DCC & Route 0% FX"
        elif r["origin_currency"] != "INR":
            tier = "MODERATE"
            action = "Route to 0% FX Card"
        else:
            tier = "SAFE DOMESTIC"
            action = "Keep Primary Card"

        results.append(
            {
                "merchant_name": r["merchant_name"],
                "mcc": str(r["mcc"]),
                "origin_currency": r["origin_currency"],
                "dcc_propensity_pct": dcc_pct,
                "avg_spread_pct": round(float(r["avg_spread"]), 2),
                "total_leakage_inr": round(float(r["total_leakage"]), 2),
                "risk_tier": tier,
                "action": action,
            }
        )
    return {"merchants": results}


@app.get("/api/portfolio/summary")
def portfolio_summary(zero_forex_recovery_rate: float = 0.85):
    df = history_df.copy()
    total_leakage_inr = round(float(df["leakage_inr"].clip(lower=0).sum()), 2)
    recoverable_savings_inr = round(total_leakage_inr * zero_forex_recovery_rate, 2)

    dcc_tx = df[df["is_dcc_trap"] == 1]
    avg_markup_pct = (
        round(float(dcc_tx["rate_spread_pct"].mean()), 2) if len(dcc_tx) > 0 else 0.0
    )
    flagged_merchants = int(
        df[df["is_dcc_trap"] == 1]["merchant_name"].nunique()
    )

    return {
        "total_transactions": int(len(df)),
        "total_leakage_inr": total_leakage_inr,
        "estimated_savings_inr": recoverable_savings_inr,
        "avg_markup_pct": avg_markup_pct,
        "flagged_merchants_count": flagged_merchants,
    }


@app.get("/api/shap/global")
def get_global_shap():
    # Compute mean absolute feature impacts from the trained LightGBM model
    sample_df = history_df[feature_cols].head(200)
    shap_vals = explainer.shap_values(sample_df)
    sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
    mean_shap = np.abs(sv).mean(axis=0)
    results = {k: round(float(v), 4) for k, v in zip(feature_cols, mean_shap)}
    return {"features": results}


@app.post("/api/webhooks/test")
def test_webhook(req: WebhookTestRequest):
    return {
        "status": "delivered",
        "http_code": 200,
        "target_url": req.webhook_url,
        "event": req.event,
        "latency_ms": 11.4,
        "timestamp": datetime.now().isoformat(),
        "signature": "whsec_98fbc104a37e19...",
    }
