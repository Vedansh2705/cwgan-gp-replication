# evaluation/generate_plots.py
# =============================================================
# Phase 12: Saved CSV results se saare plots generate karna
# =============================================================
# Ye script training dobara nahi karta - sirf already-saved
# CSV files (results/ folder) ko padh kar graphs banata hai.

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

DATASETS = {
    "IRIS": 2.00,
    "WINE": 2.02,
    "HEART": 1.25,
    "HABERMAN": 2.78,
    "LIVER": 1.23,
}

METHOD_ORDER = ["SMOTE", "ADASYN", "Borderline-SMOTE", "ROS",
                "GAN", "CGAN", "WGAN-GP", "CWGAN-GP"]


def load_dataset_csv(dataset_name):
    """CSV padhta hai: Method, Classifier, Metric, Mean, Std columns."""
    path = os.path.join(RESULTS_DIR, f"full_comparison_8methods_{dataset_name}.csv")
    if not os.path.exists(path):
        print(f"  [SKIP] {path} nahi mila")
        return None
    return pd.read_csv(path)


def compute_mean_rank_per_method(df, metric):
    """Ek dataset ke liye, har method ka mean rank (sab classifiers ke across)."""
    sub = df[df["Metric"] == metric]
    classifiers = sub["Classifier"].unique()

    ranks = {m: [] for m in METHOD_ORDER}
    for clf in classifiers:
        clf_rows = sub[sub["Classifier"] == clf].set_index("Method")["Mean"]
        sorted_methods = clf_rows.sort_values().index.tolist()
        for rank_val, method in enumerate(sorted_methods, start=1):
            if method in ranks:
                ranks[method].append(rank_val)

    return {m: np.mean(v) if v else 0 for m, v in ranks.items()}


# =============================================================
# PLOT 1: Per-dataset bar chart (mean rank, all methods)
# =============================================================
def plot_per_dataset_ranks():
    for dataset_name in DATASETS:
        df = load_dataset_csv(dataset_name)
        if df is None:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, metric in zip(axes, ["f_measure", "g_mean", "auc"]):
            ranks = compute_mean_rank_per_method(df, metric)
            methods = list(ranks.keys())
            values = list(ranks.values())

            colors = ['#d62728' if m == 'CWGAN-GP' else '#1f77b4' for m in methods]
            ax.bar(methods, values, color=colors)
            ax.set_title(f"{metric.upper()} - Mean Rank")
            ax.set_ylabel("Mean Rank (higher = better)")
            ax.tick_params(axis='x', rotation=60)
            ax.set_ylim(0, 8)
            ax.grid(axis='y', alpha=0.3)

        fig.suptitle(f"{dataset_name} (IR={DATASETS[dataset_name]}) — Method Comparison", fontsize=14)
        fig.tight_layout()
        out_path = os.path.join(PLOTS_DIR, f"ranks_{dataset_name}.png")
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")


# =============================================================
# PLOT 2: Cross-dataset summary - CWGAN-GP rank vs IR
# =============================================================
def plot_cross_dataset_summary():
    records = []
    for dataset_name, ir in DATASETS.items():
        df = load_dataset_csv(dataset_name)
        if df is None:
            continue
        for metric in ["f_measure", "g_mean", "auc"]:
            ranks = compute_mean_rank_per_method(df, metric)
            records.append({
                "dataset": dataset_name, "IR": ir, "metric": metric,
                "CWGAN-GP_rank": ranks.get("CWGAN-GP", 0)
            })

    summary_df = pd.DataFrame(records)
    summary_df = summary_df.sort_values("IR")

    fig, ax = plt.subplots(figsize=(9, 6))
    for metric in ["f_measure", "g_mean", "auc"]:
        sub = summary_df[summary_df["metric"] == metric]
        ax.plot(sub["IR"], sub["CWGAN-GP_rank"], marker='o', label=metric.upper(), linewidth=2)

    for _, row in summary_df[summary_df["metric"] == "f_measure"].iterrows():
        ax.annotate(row["dataset"], (row["IR"], row["CWGAN-GP_rank"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax.set_xlabel("Imbalance Ratio (IR)")
    ax.set_ylabel("CWGAN-GP Mean Rank (max=8, higher=better)")
    ax.set_title("CWGAN-GP Performance vs Dataset Imbalance Ratio")
    ax.axhline(y=4.5, color='gray', linestyle='--', alpha=0.5, label="Mid-point (4.5/8)")
    ax.legend()
    ax.grid(alpha=0.3)

    out_path = os.path.join(PLOTS_DIR, "cross_dataset_summary.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # CSV bhi save karo reference ke liye
    summary_df.to_csv(os.path.join(RESULTS_DIR, "cross_dataset_summary.csv"), index=False)


# =============================================================
# PLOT 3: Mean metric values (not rank) - grouped bar per dataset
# =============================================================
def plot_metric_values():
    for dataset_name in DATASETS:
        df = load_dataset_csv(dataset_name)
        if df is None:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        f1_df = df[df["Metric"] == "f_measure"]
        pivot = f1_df.pivot_table(index="Classifier", columns="Method", values="Mean")
        pivot = pivot.reindex(columns=[m for m in METHOD_ORDER if m in pivot.columns])

        pivot.plot(kind='bar', ax=ax, width=0.85)
        ax.set_title(f"{dataset_name} — F-measure by Classifier and Method")
        ax.set_ylabel("F-measure")
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

        out_path = os.path.join(PLOTS_DIR, f"fmeasure_detail_{dataset_name}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    print("Per-dataset rank plots generate ho rahe hain...")
    plot_per_dataset_ranks()

    print("\nCross-dataset summary plot generate ho raha hai...")
    plot_cross_dataset_summary()

    print("\nPer-classifier F-measure detail plots generate ho rahe hain...")
    plot_metric_values()

    print(f"\nSab plots yahan save hue: {PLOTS_DIR}")