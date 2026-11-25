from datetime import date, datetime, timedelta
import sqlite3
from typing import Optional
import requests

FRANKFURTER_BASE = "https://api.frankfurter.dev/v1"

FALLBACK_RATES_TO_INR = {
    "USD": 84.50,
    "EUR": 91.20,
    "GBP": 107.40,
    "AED": 23.01,
    "SGD": 63.10,
    "JPY": 0.55,
    "INR": 1.00,
}


class FxRateCache:
    def __init__(self, db_path: str = "fx_rates_cache.sqlite"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fx_rates (
                    rate_date TEXT NOT NULL,
                    base_curr TEXT NOT NULL,
                    quote_curr TEXT NOT NULL,
                    rate REAL NOT NULL,
                    source TEXT NOT NULL,
                    PRIMARY KEY (rate_date, base_curr, quote_curr)
                )
                """
            )

    def _get_cached(self, rate_date: str, base_curr: str, quote_curr: str) -> Optional[float]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT rate FROM fx_rates WHERE rate_date = ? AND base_curr = ? AND quote_curr = ?",
                (rate_date, base_curr, quote_curr),
            )
            row = cur.fetchone()
            return float(row[0]) if row else None

    def _save_cached(
        self, rate_date: str, base_curr: str, quote_curr: str, rate: float, source: str
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO fx_rates (rate_date, base_curr, quote_curr, rate, source)
                VALUES (?, ?, ?, ?, ?)
                """,
                (rate_date, base_curr, quote_curr, rate, source),
            )

    @staticmethod
    def _normalize_date(ts_or_date: Optional[str]) -> str:
        if not ts_or_date:
            return date.today().isoformat()
        try:
            return datetime.fromisoformat(ts_or_date.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            return ts_or_date[:10]

    def get_rate(
        self, base_curr: str, quote_curr: str, ts_or_date: Optional[str] = None
    ) -> float:
        """
        Returns 1 base_curr = X quote_curr using cache -> Frankfurter API -> fallback.
        """
        base = base_curr.upper()
        quote = quote_curr.upper()
        if base == quote:
            return 1.0

        target_str = self._normalize_date(ts_or_date)
        cached = self._get_cached(target_str, base, quote)
        if cached is not None:
            return cached

        target_dt = date.fromisoformat(target_str)
        # Walk back up to 5 days for weekends/bank holidays
        for offset in range(6):
            query_dt = target_dt - timedelta(days=offset)
            endpoint = (
                f"{FRANKFURTER_BASE}/latest"
                if query_dt >= date.today()
                else f"{FRANKFURTER_BASE}/{query_dt.isoformat()}"
            )
            try:
                resp = requests.get(
                    endpoint, params={"base": base, "symbols": quote}, timeout=5
                )
                if resp.status_code == 200:
                    payload = resp.json()
                    rates = payload.get("rates", {})
                    if quote in rates:
                        rate = float(rates[quote])
                        self._save_cached(target_str, base, quote, rate, "frankfurter")
                        return rate
            except requests.RequestException:
                continue

        # Fallback if offline or currency unsupported by ECB/Frankfurter
        if base in FALLBACK_RATES_TO_INR and quote == "INR":
            rate = FALLBACK_RATES_TO_INR[base]
            self._save_cached(target_str, base, quote, rate, "fallback")
            return rate
        if base == "INR" and quote in FALLBACK_RATES_TO_INR:
            rate = 1.0 / FALLBACK_RATES_TO_INR[quote]
            self._save_cached(target_str, base, quote, rate, "fallback")
            return rate

        return 1.0