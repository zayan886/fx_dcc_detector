from dataclasses import dataclass
from typing import Optional


@dataclass
class LeakageBreakdown:
    mid_market_inr: float
    leakage_inr: float
    leakage_pct: float
    implied_fx_rate: float
    has_explicit_foreign_amt: int


def calculate_leakage(
    billed_amount: float,
    billed_currency: str,
    home_currency: str,
    origin_amount: Optional[float],
    mid_market_rate: float,
) -> LeakageBreakdown:
    has_foreign = 1 if (origin_amount is not None and origin_amount > 0) else 0

    if billed_currency == home_currency and has_foreign and mid_market_rate > 1.0:
        mid_market_inr = float(origin_amount) * mid_market_rate
        leakage_inr = billed_amount - mid_market_inr
        implied_fx_rate = billed_amount / float(origin_amount)
    elif billed_currency == home_currency:
        mid_market_inr = billed_amount
        leakage_inr = 0.0
        implied_fx_rate = 1.0
    else:
        qty = float(origin_amount) if has_foreign else billed_amount
        mid_market_inr = qty * mid_market_rate
        billed_in_home = billed_amount * mid_market_rate
        leakage_inr = max(0.0, billed_in_home - mid_market_inr)
        implied_fx_rate = mid_market_rate

    leakage_pct = (leakage_inr / mid_market_inr * 100.0) if mid_market_inr > 0 else 0.0
    return LeakageBreakdown(
        mid_market_inr=round(mid_market_inr, 2),
        leakage_inr=round(leakage_inr, 2),
        leakage_pct=round(leakage_pct, 2),
        implied_fx_rate=round(implied_fx_rate, 4),
        has_explicit_foreign_amt=has_foreign,
    )