"""
Open Question Experiment:
Does hair removal improve classification more for hair-heavy images?

This script compares the baseline and extended pipelines by splitting
the test data into two groups based on hair coverage:
    - low hair
    - high hair

The split is done using the median hair coverage to ensure roughly
balanced group sizes, allowing a fair comparison across subgroups.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score
)

# Load data
features = pd.read_csv("data/features_extended.csv")
pred_base = pd.read_csv("results/predictions_baseline.csv")
pred_ext = pd.read_csv("results/predictions_extended.csv")

# Merge hair coverage with predictions
hair = features[["img_id", "patient_id", "hair_coverage"]]
base = pred_base.merge(hair, on=["img_id", "patient_id"])
ext = pred_ext.merge(hair, on=["img_id", "patient_id"])

# Define hair groups
median_hair = base["hair_coverage"].median()
base["hair_group"] = base["hair_coverage"].apply(lambda x: "low" if x <= median_hair else "high")
ext["hair_group"] = ext["hair_coverage"].apply(lambda x: "low" if x <= median_hair else "high")


def compute_metrics(df):
    """
    Compute evaluation metrics for a given subset of data.

    Parameters:
        df (DataFrame): subset of predictions

    Returns:
        dict: dictionary with evaluation metrics
    """
    y_true = df["true_label"]
    y_pred = df["predicted_label"]
    y_prob = df["probability_cancer"]

    results = {
        "n": len(df),
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    # AUC requires both lesion classes to be present
    if y_true.nunique() > 1:
        results["auc"] = roc_auc_score(y_true, y_prob)
    else:
        results["auc"] = np.nan

    return results

# Run experiment
rows = []

# Evaluate each model within each hair group
for group in ["low", "high"]:
    for name, df in [("baseline", base), ("extended", ext)]:
        subset = df[df["hair_group"] == group]

        metrics = compute_metrics(subset)
        metrics["group"] = group
        metrics["model"] = name

        rows.append(metrics)

# Convert results to DataFrame
summary = pd.DataFrame(rows)
summary = summary[["group", "model", "n", "accuracy", "recall", "precision", "f1", "auc"]]

# Save results
summary.to_csv("results/open_question_hair.csv", index=False)


# create AUC graph 
pivot = summary.pivot(index="group", columns="model", values="auc")

pivot.plot(kind="bar")

plt.ylabel("AUC")
plt.title("Model Performance by Hair Coverage")
plt.xticks(rotation=0)
plt.legend(title="Model")

plt.tight_layout()
plt.savefig("results/figures/open_question_hair_plot.png", dpi=300)
plt.show()