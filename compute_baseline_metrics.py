import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

df = pd.read_csv("results/predictions_baseline.csv")

y_true = df["true_label"]
y_pred = df["predicted_label"]

accuracy = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print("=== BASELINE RESULTS ===")
print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)

print("\nConfusion matrix:")
print("TP:", tp)
print("FP:", fp)
print("TN:", tn)
print("FN:", fn)

results = pd.DataFrame({
    "Metric": ["Accuracy", "Recall", "Precision", "TP", "FP", "TN", "FN"],
    "Value": [accuracy, recall, precision, tp, fp, tn, fn]
})

results.to_csv("results/baseline_metrics_summary.csv",
               index=False)
