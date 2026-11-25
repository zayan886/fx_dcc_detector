# 💳 Real-Time FX & DCC Leakage Detector

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/Model-LightGBM%20%2B%20SHAP-2E8B57?style=flat-square)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

An automated FinTech risk & card-routing engine that detects **Dynamic Currency Conversion (DCC)** traps and hidden foreign exchange markups on debit and credit card transactions in real time—calculating exact INR leakage, explaining risk predictions with **SHAP**, and recommending optimal **Zero-Forex card routing**.

---

## 🎯 The Real-World Problem

When cardholders pay on international websites, app stores, or foreign POS terminals, payment gateways routinely default to **Dynamic Currency Conversion (DCC)**—billing in the user's home currency (`INR`) at an inflated exchange rate (**4%–9% above mid-market**) on top of standard bank forex markups (**3.5% + 18% GST**).

Most bank SMS and email alerts do not disclose the mid-market rate or even the original foreign currency amount, causing cardholders to silently lose **5%–8% per international transaction**.

---

## ✨ Key Features

- **Two-Layer Detection Engine**
  - **Deterministic FX Calculator (`fx_engine.py`):** Computes exact mid-market INR value, rupee leakage, percentage overhead, and implied FX rate spread.
  - **LightGBM + SHAP Classifier (`train_model.py`):** Predicts DCC trap probability even when bank alerts omit the foreign currency amount, using chronological merchant history, MCC codes, and billing currency signals.
- **Live & Historical FX Cache (`fx_cache.py`):** Fetches live and historical exchange rates via the **Frankfurter API** with weekend/holiday fallback and a local **SQLite cache** keyed by `(date, base, quote)`.
- **Multi-Format Bank Alert Parser (`alert_parser.py`):** Parses structured and incomplete SMS/email transaction alerts, infers MCC/international context, and routes signals to the model.
- **Automated Card-Routing & Portfolio API (`api.py`):** FastAPI service exposing real-time alert scoring, cumulative INR saved, top DCC-offending merchants, and actionable card-routing rules alongside an interactive web dashboard.

---

## 🏗️ System Architecture

```text
SMS / Email / Webhook Alert
          │
          ▼
  [ alert_parser.py ] ──► Extracts billed amount, currency, merchant, date, optional FX hint
          │
          ▼
    [ fx_cache.py ]   ──► SQLite Cache ──► Frankfurter Live/Historical FX API
          │
          ▼
   [ fx_engine.py ]   ──► Computes mid-market INR, exact leakage (INR & %), implied FX rate
          │
          ▼
[ LightGBM + SHAP ]   ──► Scores DCC probability + explains top feature drivers
          │
          ▼
  [ Routing Engine ]  ──► BLOCK_DCC / ROUTE_TO_ZERO_FOREX_CARD / KEEP_CURRENT_CARD
