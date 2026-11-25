from datetime import datetime
import re
from typing import Optional

MERCHANT_HINTS = {
    "ACME CLOUD": ("5734", "USD", 1),
    "STEAM": ("5816", "USD", 1),
    "HOTEL": ("7011", "EUR", 1),
    "BOOKING": ("7011", "EUR", 1),
    "AIRLINE": ("4511", "USD", 1),
    "DEV API": ("5734", "USD", 1),
    "MUSEUM": ("7991", "GBP", 1),
    "GROCERY": ("5411", "INR", 0),
    "SUPERMARKET": ("5411", "INR", 0),
    "METRO": ("4111", "INR", 0),
    "PHARMACY": ("5912", "INR", 0),
}


def _normalize_date(raw_date: Optional[str]) -> str:
    if not raw_date:
        return datetime.now().date().isoformat()
    for fmt in ("%d-%m-%y", "%d/%m/%y", "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw_date, fmt).date().isoformat()
        except ValueError:
            continue
    return datetime.now().date().isoformat()


def parse_bank_alert(text: str, home_currency: str = "INR") -> dict:
    clean = " ".join(text.strip().split())

    # 1. Primary billed amount & currency
    amt_match = re.search(
        r"(?:Spent\s+)?([A-Z]{3})\s+([\d,]+(?:\.\d{1,2})?)|([\d,]+(?:\.\d{1,2})?)\s+([A-Z]{3})",
        clean,
        re.IGNORECASE,
    )
    if amt_match:
        if amt_match.group(1) and amt_match.group(2):
            billed_currency = amt_match.group(1).upper()
            billed_amount = float(amt_match.group(2).replace(",", ""))
        else:
            billed_amount = float(amt_match.group(3).replace(",", ""))
            billed_currency = amt_match.group(4).upper()
    else:
        billed_currency = home_currency
        billed_amount = 0.0

    # 2. Merchant name
    merch_match = re.search(
        r"\bat\s+([A-Z0-9][A-Z0-9\s\.\-&*]+?)(?=\s+(?:on\s+\d|via\s+|\(|on\s+Card|$))",
        clean,
        re.IGNORECASE,
    )
    merchant_name = merch_match.group(1).strip().upper() if merch_match else "UNKNOWN MERCHANT"

    # 3. Card last-4 (optional)
    card_match = re.search(r"(?:Card\s*(?:ending|XX|No\.?)?\s*)(\d{4})", clean, re.IGNORECASE)
    card_last4 = card_match.group(1) if card_match else "0000"

    # 4. Date (optional)
    date_match = re.search(r"\b(\d{2}[/-]\d{2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b", clean)
    tx_date = _normalize_date(date_match.group(1) if date_match else None)

    # 5. Explicit foreign currency + optional mid-market rate inside parentheses
    fx_paren = re.search(
        r"\(([A-Z]{3})\s+([\d,]+(?:\.\d{1,2})?)(?:,\s*mid\s+([\d.]+))?\)",
        clean,
        re.IGNORECASE,
    )

    # Infer MCC and international status from merchant keywords
    inferred_mcc = "5999"
    inferred_curr = "USD" if billed_currency == home_currency else billed_currency
    inferred_intl = 0 if billed_currency == home_currency else 1

    for key, (mcc, curr, intl) in MERCHANT_HINTS.items():
        if key in merchant_name:
            inferred_mcc = mcc
            inferred_curr = curr
            inferred_intl = intl
            break

    if fx_paren:
        origin_currency = fx_paren.group(1).upper()
        origin_amount = float(fx_paren.group(2).replace(",", ""))
        explicit_mid_rate = float(fx_paren.group(3)) if fx_paren.group(3) else None
        has_explicit_foreign_amt = 1
        is_international = 1
    elif billed_currency != home_currency:
        origin_currency = billed_currency
        origin_amount = billed_amount
        explicit_mid_rate = None
        has_explicit_foreign_amt = 1
        is_international = 1
    else:
        # Foreign amount NOT in alert: keep origin_amount=None so model relies on history/MCC/currency
        origin_currency = inferred_curr
        origin_amount = None
        explicit_mid_rate = None
        has_explicit_foreign_amt = 0
        is_international = inferred_intl

    return {
        "tx_date": tx_date,
        "card_last4": card_last4,
        "merchant_name": merchant_name,
        "mcc": inferred_mcc,
        "is_international": is_international,
        "home_currency": home_currency,
        "origin_currency": origin_currency,
        "origin_amount": origin_amount,
        "explicit_mid_rate": explicit_mid_rate,
        "has_explicit_foreign_amt": has_explicit_foreign_amt,
        "billed_currency": billed_currency,
        "billed_amount": billed_amount,
    }