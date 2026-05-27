import pandas as pd
from src.classifier import (
    get_features,
    cross_validate_tree_depth,
    train_model,
    evaluate_model
)

data = pd.read_csv("results/features_baseline.csv")

features = get_features(data)

best_depth, cv_results = cross_validate_tree_depth(
    data=data,
    features=features,
    max_depth=10
)

cv_results.to_csv("results/cross_validation_baseline.csv", index=False)

model = train_model(data, features, max_depth=best_depth)

evaluate_model(
    data=data,
    features=features,
    model=model,
    result_dir="results",
    output_name="predictions_baseline.csv"
)

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix
)

df = pd.read_csv("results/predictions_baseline.csv")

y_true = df["true_label"]
y_pred = df["predicted_label"]

accuracy = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print("\nBaseline Results:=")
print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("TP:", tp, "FP:", fp, "TN:", tn, "FN:", fn)
 
results = pd.DataFrame({
    "Metric": ["Accuracy", "Recall", "Precision", "TP", "FP", "TN", "FN"],
    "Value": [accuracy, recall, precision, tp, fp, tn, fn]
})

results.to_csv("results/baseline_metrics_summary.csv", index=False)
