import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import classification_report, precision_recall_curve, auc, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from fx_engine import calculate_leakage

FEATURE_COLS = [
    "is_international",
    "billed_in_home_currency",
    "leakage_pct",
    "rate_spread_pct",
    "merchant_hist_dcc_rate",
    "merchant_hist_leakage_pct",
    "merchant_prior_tx_count",
    "log_mid_market_inr",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True).copy()

    breakdowns = [
        calculate_leakage(
            billed_amount=row["billed_amount"],
            billed_currency=row["billed_currency"],
            home_currency=row["home_currency"],
            origin_amount=row["origin_amount"],
            mid_market_rate=row["mid_market_rate"],
        )
        for _, row in df.iterrows()
    ]

    df["mid_market_inr"] = [b.mid_market_inr for b in breakdowns]
    df["leakage_inr"] = [b.leakage_inr for b in breakdowns]
    df["leakage_pct"] = [b.leakage_pct for b in breakdowns]
    df["implied_fx_rate"] = [b.implied_fx_rate for b in breakdowns]

    df["billed_in_home_currency"] = (
        df["billed_currency"] == df["home_currency"]
    ).astype(int)

    df["rate_spread_pct"] = np.where(
        df["mid_market_rate"] > 1.0,
        ((df["implied_fx_rate"] - df["mid_market_rate"]) / df["mid_market_rate"]) * 100.0,
        0.0,
    )

    df["merchant_prior_tx_count"] = df.groupby("merchant_name").cumcount()
    df["merchant_hist_dcc_rate"] = (
        df.groupby("merchant_name")["is_dcc_trap"]
        .transform(lambda s: s.shift(1).expanding().mean())
        .fillna(0.0)
    )
    df["merchant_hist_leakage_pct"] = (
        df.groupby("merchant_name")["leakage_pct"]
        .transform(lambda s: s.shift(1).expanding().mean())
        .fillna(0.0)
    )
    df["log_mid_market_inr"] = np.log1p(df["mid_market_inr"].clip(lower=0.0))
    df["has_explicit_foreign_amt"] = [b.has_explicit_foreign_amt for b in breakdowns]
    df["mcc_code"] = pd.to_numeric(df["mcc"], errors="coerce").fillna(5999).astype(int)
    return df


def main():
    raw_df = pd.read_csv("transactions.csv")
    feat_df = add_features(raw_df)

    X = feat_df[FEATURE_COLS]
    y = feat_df["is_dcc_trap"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = lgb.LGBMClassifier(
        n_estimators=120,
        learning_rate=0.05,
        num_leaves=15,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}\n")
    print(classification_report(y_test, y_pred, digits=3))

    # SHAP feature importance on test set
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    mean_abs_shap = pd.Series(
        np.abs(sv).mean(axis=0), index=FEATURE_COLS
    ).sort_values(ascending=False)
    print("Top Features by Mean |SHAP|:")
    print(mean_abs_shap.round(4))

    joblib.dump({"model": model, "features": FEATURE_COLS}, "dcc_model_bundle.joblib")
    feat_df.to_csv("transactions_enriched.csv", index=False)
    print("\nSaved dcc_model_bundle.joblib and transactions_enriched.csv")


if __name__ == "__main__":
    main()