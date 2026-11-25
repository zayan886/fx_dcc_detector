import re
import joblib
import numpy as np
import pandas as pd

from fx_engine import calculate_leakage


def parse_sms_alert(sms_text: str) -> dict:
    """
    Example SMS parser for transaction alerts such as:
    'Spent INR 4,485.00 at ACME CLOUD EU (USD 49.00, mid 84.50) on Card ending 4412.'
    """
    amt_match = re.search(r"Spent\s+([A-Z]{3})\s+([\d,]+(?:\.\d{1,2})?)", sms_text)
    merch_match = re.search(r"at\s+([A-Z0-9\s]+?)\s*(?:\(|on\s+Card|$)", sms_text)
    orig_match = re.search(
        r"\(([A-Z]{3})\s+([\d,]+(?:\.\d{1,2})?),\s*mid\s+([\d.]+)\)", sms_text
    )

    billed_currency = amt_match.group(1) if amt_match else "INR"
    billed_amount = float(amt_match.group(2).replace(",", "")) if amt_match else 0.0
    merchant_name = merch_match.group(1).strip() if merch_match else "UNKNOWN MERCHANT"

    if orig_match:
        origin_currency = orig_match.group(1)
        origin_amount = float(orig_match.group(2).replace(",", ""))
        mid_market_rate = float(orig_match.group(3))
        is_international = 1
    else:
        origin_currency = billed_currency
        origin_amount = billed_amount
        mid_market_rate = 1.0 if billed_currency == "INR" else 84.50
        is_international = 0 if billed_currency == "INR" else 1

    return {
        "merchant_name": merchant_name,
        "is_international": is_international,
        "home_currency": "INR",
        "origin_currency": origin_currency,
        "billed_currency": billed_currency,
        "origin_amount": origin_amount,
        "mid_market_rate": mid_market_rate,
        "billed_amount": billed_amount,
    }


def evaluate_alert(sms_text: str, history_df: pd.DataFrame, bundle: dict) -> dict:
    parsed = parse_sms_alert(sms_text)
    breakdown = calculate_leakage(
        billed_amount=parsed["billed_amount"],
        billed_currency=parsed["billed_currency"],
        home_currency=parsed["home_currency"],
        origin_amount=parsed["origin_amount"],
        mid_market_rate=parsed["mid_market_rate"],
    )

    merch_hist = history_df[history_df["merchant_name"] == parsed["merchant_name"]]
    prior_count = len(merch_hist)
    hist_dcc_rate = float(merch_hist["is_dcc_trap"].mean()) if prior_count > 0 else 0.0
    hist_leakage_pct = (
        float(merch_hist["leakage_pct"].mean()) if prior_count > 0 else 0.0
    )

    billed_in_home = int(parsed["billed_currency"] == parsed["home_currency"])
    rate_spread_pct = (
        ((breakdown.implied_fx_rate - parsed["mid_market_rate"]) / parsed["mid_market_rate"])
        * 100.0
        if parsed["mid_market_rate"] > 1.0
        else 0.0
    )

    row = pd.DataFrame([{
        "is_international": parsed["is_international"],
        "billed_in_home_currency": billed_in_home,
        "leakage_pct": breakdown.leakage_pct,
        "rate_spread_pct": rate_spread_pct,
        "merchant_hist_dcc_rate": hist_dcc_rate,
        "merchant_hist_leakage_pct": hist_leakage_pct,
        "merchant_prior_tx_count": prior_count,
        "log_mid_market_inr": float(np.log1p(max(0.0, breakdown.mid_market_inr))),
    }])[bundle["features"]]

    dcc_prob = float(bundle["model"].predict_proba(row)[0, 1])

    if dcc_prob >= 0.70 or breakdown.leakage_pct >= 4.0:
        action = "BLOCK_DCC_AND_SWITCH_TO_ZERO_FOREX_CARD"
        advice = (
            f"Potential DCC markup detected ({breakdown.leakage_pct:.2f}%, ~INR {breakdown.leakage_inr:.2f} extra). "
            f"Select {parsed['origin_currency']} at checkout and use a zero-forex card."
        )
    elif parsed["is_international"] and breakdown.leakage_pct > 0.5:
        action = "USE_ZERO_FOREX_CARD_FOR_FOREIGN_CURRENCY"
        advice = (
            f"Billed in {parsed['billed_currency']} with ~{breakdown.leakage_pct:.2f}% FX overhead "
            f"(~INR {breakdown.leakage_inr:.2f}). Route future charges to a zero-forex card."
        )
    else:
        action = "KEEP_CURRENT_CARD"
        advice = "No significant DCC or FX markup leakage detected."

    return {
        "merchant": parsed["merchant_name"],
        "dcc_probability": round(dcc_prob, 3),
        "mid_market_inr": breakdown.mid_market_inr,
        "billed_amount": parsed["billed_amount"],
        "billed_currency": parsed["billed_currency"],
        "leakage_inr": breakdown.leakage_inr,
        "leakage_pct": breakdown.leakage_pct,
        "action": action,
        "advice": advice,
    }


if __name__ == "__main__":
    bundle = joblib.load("dcc_model_bundle.joblib")
    history = pd.read_csv("transactions_enriched.csv")

    sample_sms = "Spent INR 4,485.00 at ACME CLOUD EU (USD 49.00, mid 84.50) on Card ending 4412."
    result = evaluate_alert(sample_sms, history, bundle)
    for k, v in result.items():
        print(f"{k}: {v}")