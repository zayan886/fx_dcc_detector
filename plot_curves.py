import shutil
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from train_model import FEATURE_COLS, add_features


def generate_roc_pr_curves():
    # 1. Load data & run feature engineering
    raw_df = pd.read_csv("transactions.csv")
    feat_df = add_features(raw_df)

    X = feat_df[FEATURE_COLS]
    y = feat_df["is_dcc_trap"]

    # 2. Train/Test split matching model training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Load trained model bundle
    bundle = joblib.load("dcc_model_bundle.joblib")
    model = bundle["model"]

    # 4. Probabilities for the positive class (is_dcc_trap = 1)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Compute ROC curve metrics
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Compute Precision-Recall curve metrics
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    avg_prec = average_precision_score(y_test, y_prob)

    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"PR-AUC (auc):  {pr_auc:.4f}")
    print(f"Avg Precision: {avg_prec:.4f}")

    # Set up dark styled figures matching dashboard aesthetic
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # --- Plot 1: ROC Curve ---
    ax1.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"LightGBM (ROC-AUC = {roc_auc:.4f})")
    ax1.plot([0, 1], [0, 1], color="#9ca3af", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.50)")
    ax1.fill_between(fpr, tpr, alpha=0.15, color="#2563eb")
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.05])
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="600")
    ax1.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12, fontweight="600")
    ax1.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold", pad=12)
    ax1.legend(loc="lower right", fontsize=11, frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # --- Plot 2: PR Curve ---
    baseline = y_test.sum() / len(y_test)
    ax2.plot(recall, precision, color="#10b981", lw=2.5, label=f"LightGBM (PR-AUC = {pr_auc:.4f})")
    ax2.plot([0, 1], [baseline, baseline], color="#ef4444", lw=1.5, linestyle="--", label=f"No-Skill Baseline ({baseline:.3f})")
    ax2.fill_between(recall, precision, alpha=0.15, color="#10b981")
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.05])
    ax2.set_xlabel("Recall", fontsize=12, fontweight="600")
    ax2.set_ylabel("Precision", fontsize=12, fontweight="600")
    ax2.set_title("Precision-Recall (PR) Curve", fontsize=14, fontweight="bold", pad=12)
    ax2.legend(loc="lower left", fontsize=11, frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("FX & DCC Leakage Detector: Performance Curves", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    # Save combined plot
    combined_path = Path("roc_pr_curves.png")
    fig.savefig(combined_path, bbox_inches="tight")
    plt.close(fig)

    # Individual ROC plot
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6), dpi=300)
    ax_roc.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"ROC-AUC = {roc_auc:.4f}")
    ax_roc.plot([0, 1], [0, 1], color="#9ca3af", lw=1.5, linestyle="--", label="Random Chance (0.50)")
    ax_roc.fill_between(fpr, tpr, alpha=0.15, color="#2563eb")
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.05])
    ax_roc.set_xlabel("False Positive Rate", fontsize=12, fontweight="600")
    ax_roc.set_ylabel("True Positive Rate", fontsize=12, fontweight="600")
    ax_roc.set_title("ROC Curve (DCC Trap Detection)", fontsize=14, fontweight="bold", pad=12)
    ax_roc.legend(loc="lower right", fontsize=11)
    ax_roc.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig_roc.savefig("roc_auc_curve.png", bbox_inches="tight")
    plt.close(fig_roc)

    # Individual PR plot
    fig_pr, ax_pr = plt.subplots(figsize=(7, 6), dpi=300)
    ax_pr.plot(recall, precision, color="#10b981", lw=2.5, label=f"PR-AUC = {pr_auc:.4f}")
    ax_pr.plot([0, 1], [baseline, baseline], color="#ef4444", lw=1.5, linestyle="--", label=f"Baseline ({baseline:.3f})")
    ax_pr.fill_between(recall, precision, alpha=0.15, color="#10b981")
    ax_pr.set_xlim([-0.02, 1.02])
    ax_pr.set_ylim([-0.02, 1.05])
    ax_pr.set_xlabel("Recall", fontsize=12, fontweight="600")
    ax_pr.set_ylabel("Precision", fontsize=12, fontweight="600")
    ax_pr.set_title("Precision-Recall Curve (DCC Trap Detection)", fontsize=14, fontweight="bold", pad=12)
    ax_pr.legend(loc="lower left", fontsize=11)
    ax_pr.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig_pr.savefig("pr_auc_curve.png", bbox_inches="tight")
    plt.close(fig_pr)

    print("Saved roc_pr_curves.png, roc_auc_curve.png, and pr_auc_curve.png")

    # Copy to artifact directory for presentation if artifact directory exists
    artifact_dir = Path(r"C:\Users\TUF F16\.gemini\antigravity\brain\b7d82b2d-cdd5-4058-883e-671b22f1afd3")
    if artifact_dir.exists():
        shutil.copy("roc_pr_curves.png", artifact_dir / "roc_pr_curves.png")
        shutil.copy("roc_auc_curve.png", artifact_dir / "roc_auc_curve.png")
        shutil.copy("pr_auc_curve.png", artifact_dir / "pr_auc_curve.png")
        print(f"Copied plots to {artifact_dir}")


if __name__ == "__main__":
    generate_roc_pr_curves()
