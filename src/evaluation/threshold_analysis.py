import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "semantic"
    / "semantic_full_seed42"
    / "validation_predictions.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "semantic" / "semantic_full_seed42"

THRESHOLD_FILE = OUTPUT_DIR / "threshold_analysis.csv"
SUMMARY_FILE = OUTPUT_DIR / "threshold_summary.json"


def main():
    print("Loading:")
    print(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    y_true = df["label"].to_numpy(dtype=int)
    probabilities = df["p_vulnerable"].to_numpy(dtype=float)

    print()
    print("Samples:", len(df))
    print("Vulnerable:", int(y_true.sum()))
    print("Non-vulnerable:", int(len(y_true) - y_true.sum()))

    pr_auc = average_precision_score(y_true, probabilities)
    roc_auc = roc_auc_score(y_true, probabilities)

    print()
    print(f"PR-AUC:  {pr_auc:.6f}")
    print(f"ROC-AUC: {roc_auc:.6f}")

    print()
    print("All probability summary:")
    print(df["p_vulnerable"].describe())

    print()
    print("Vulnerable probability summary:")
    print(df.loc[df["label"] == 1, "p_vulnerable"].describe())

    print()
    print("Non-vulnerable probability summary:")
    print(df.loc[df["label"] == 0, "p_vulnerable"].describe())

    results = []
    thresholds = np.arange(0.001, 1.000, 0.001)

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)
        mcc = matthews_corrcoef(y_true, predictions)

        results.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "mcc": float(mcc),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )

    results_df = pd.DataFrame(results)

    best_f1 = results_df.loc[results_df["f1"].idxmax()]
    best_mcc = results_df.loc[results_df["mcc"].idxmax()]

    print()
    print("BEST F1 THRESHOLD")
    print(best_f1.to_string())

    print()
    print("BEST MCC THRESHOLD")
    print(best_mcc.to_string())

    recall_targets = {}
    for target in [0.50, 0.60, 0.70, 0.80, 0.90]:
        candidates = results_df[results_df["recall"] >= target]
        if len(candidates) == 0:
            continue

        best = candidates.loc[candidates["precision"].idxmax()]
        recall_targets[str(target)] = {
            "threshold": float(best["threshold"]),
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "f1": float(best["f1"]),
            "mcc": float(best["mcc"]),
        }

    results_df.to_csv(THRESHOLD_FILE, index=False)

    summary = {
        "samples": int(len(df)),
        "vulnerable": int(y_true.sum()),
        "non_vulnerable": int(len(y_true) - y_true.sum()),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "best_f1": {
            "threshold": float(best_f1["threshold"]),
            "precision": float(best_f1["precision"]),
            "recall": float(best_f1["recall"]),
            "f1": float(best_f1["f1"]),
            "mcc": float(best_f1["mcc"]),
        },
        "best_mcc": {
            "threshold": float(best_mcc["threshold"]),
            "precision": float(best_mcc["precision"]),
            "recall": float(best_mcc["recall"]),
            "f1": float(best_mcc["f1"]),
            "mcc": float(best_mcc["mcc"]),
        },
        "recall_targets": recall_targets,
    }

    with SUMMARY_FILE.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("Saved:")
    print(THRESHOLD_FILE)
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()
