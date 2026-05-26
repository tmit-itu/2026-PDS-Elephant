import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

df = pd.read_csv("results/predictions_extended.csv")

#true + prediction labels
y_true = df["true_label"]
y_pred = df["predicted_label"]

#metrics
accuracy = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)

#confusion matrix
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

#results
print("=== EXTENDED RESULTS ===")
print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)

print("\nConfusion matrix:")
print("TP:", tp)
print("FP:", fp)
print("TN:", tn)
print("FN:", fn)

#summary csv
extended_summary = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Recall",
        "Precision",
        "TP",
        "FP",
        "TN",
        "FN"
    ],
    "Value": [
        0.697560975609756,
        0.7066666666666667,
        0.7327188940092166,
        159.0,
        58.0,
        127.0,
        66.0
    ]
})

extended_summary.to_csv("results/summary_extended.csv", index=False)
