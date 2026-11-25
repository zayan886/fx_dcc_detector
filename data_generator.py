from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def generate_transactions(n_rows: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    merchants = [
        # (name, mcc, is_international, default_foreign_curr, dcc_propensity, base_foreign_amt)
        ("ACME CLOUD EU", "5734", 1, "USD", 0.85, 49.0),
        ("STEAM STORE INTL", "5816", 1, "USD", 0.75, 24.0),
        ("HOTEL BOOKING GLOBAL", "7011", 1, "EUR", 0.80, 140.0),
        ("AIRLINE INTL DIRECT", "4511", 1, "USD", 0.65, 320.0),
        ("DEV API SERVICE", "5734", 1, "USD", 0.25, 29.0),
        ("MUSEUM PASS ABROAD", "7991", 1, "GBP", 0.70, 35.0),
        ("LOCAL SUPERMARKET", "5411", 0, "INR", 0.00, 950.0),
        ("METRO TRANSIT", "4111", 0, "INR", 0.00, 180.0),
        ("CITY PHARMACY", "5912", 0, "INR", 0.00, 620.0),
    ]

    mid_rates = {
        "USD": 84.50,
        "EUR": 91.20,
        "GBP": 107.40,
        "AED": 23.01,
        "INR": 1.00,
    }

    start_time = datetime(2026, 9, 1, 9, 0, 0)
    rows = []

    for i in range(n_rows):
        name, mcc, is_intl, f_curr, dcc_prop, base_amt = merchants[
            rng.integers(0, len(merchants))
        ]
        ts = start_time + timedelta(hours=int(rng.integers(1, 2000)))
        mid_rate = mid_rates[f_curr] * float(rng.normal(1.0, 0.004))

        if not is_intl:
            billed_amt = round(float(base_amt * rng.uniform(0.4, 2.2)), 2)
            rows.append({
                "tx_id": f"tx_{i+1:05d}",
                "timestamp": ts.isoformat(),
                "merchant_name": name,
                "mcc": mcc,
                "is_international": 0,
                "home_currency": "INR",
                "origin_currency": "INR",
                "billed_currency": "INR",
                "origin_amount": billed_amt,
                "mid_market_rate": 1.0,
                "card_markup_pct": 0.0,
                "gst_on_markup_pct": 0.0,
                "billed_amount": billed_amt,
                "is_dcc_trap": 0,
            })
            continue

        origin_amt = round(float(base_amt * rng.uniform(0.5, 2.5)), 2)
        is_dcc = int(rng.random() < dcc_prop)

        if is_dcc:
            # DCC: merchant/gateway bills in INR at a marked-up rate (4% to 9.5% higher)
            dcc_markup = float(rng.uniform(0.04, 0.095))
            billed_currency = "INR"
            card_markup_pct = 0.0
            gst_pct = 0.0
            billed_amt = round(origin_amt * mid_rate * (1.0 + dcc_markup), 2)
        else:
            # Billed in native foreign currency; bank applies standard markup + GST (or 0% card)
            billed_currency = f_curr
            zero_forex = rng.random() < 0.25
            card_markup_pct = 0.0 if zero_forex else 3.5
            gst_pct = 0.0 if zero_forex else 18.0
            fee_multiplier = 1.0 + (card_markup_pct / 100.0) * (1.0 + gst_pct / 100.0)
            billed_amt = round(origin_amt * fee_multiplier, 2)

        rows.append({
            "tx_id": f"tx_{i+1:05d}",
            "timestamp": ts.isoformat(),
            "merchant_name": name,
            "mcc": mcc,
            "is_international": 1,
            "home_currency": "INR",
            "origin_currency": f_curr,
            "billed_currency": billed_currency,
            "origin_amount": origin_amt,
            "mid_market_rate": round(mid_rate, 4),
            "card_markup_pct": card_markup_pct,
            "gst_on_markup_pct": gst_pct,
            "billed_amount": billed_amt,
            "is_dcc_trap": is_dcc,
        })

    df = pd.DataFrame(rows)
    df.to_csv("transactions.csv", index=False)
    return df


if __name__ == "__main__":
    df = generate_transactions()
    print(f"Generated {len(df)} transactions. DCC trap rate: {df['is_dcc_trap'].mean():.2%}")