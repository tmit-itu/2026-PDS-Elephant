import pandas as pd
from src.classifier import (
    get_features,
    cross_validate_tree_depth,
    train_model,
    evaluate_model
)
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

data = pd.read_csv("data/features_baseline.csv")

features = get_features(data)

best_depth, cv_results = cross_validate_tree_depth(
    data=data,
    features=features,
    max_depth=10
)

cv_results.to_csv("results/cross_validation_baseline.csv", index=False)

model = train_model(data, features, max_depth=best_depth,
                    model_path="results/models/tree_baseline.pkl")


evaluate_model(
    data=data,
    features=features,
    model=model,
    result_dir="results",
    output_name="predictions_baseline.csv"
)

df = pd.read_csv("results/predictions_baseline.csv")

y_true = df["true_label"]
y_pred = df["predicted_label"]

accuracy = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print("\nBaseline Results:=")
print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("F1 Score", f1)
print("TP:", tp, "FP:", fp, "TN:", tn, "FN:", fn)
 
results = pd.DataFrame({
    "Metric": ["Accuracy", "Recall", "Precision", "F1 Score", "TP", "FP", "TN", "FN"],
    "Value": [accuracy, recall, precision, f1, tp, fp, tn, fn]
})

results.to_csv("results/baseline_metrics_summary.csv", index=False)

cm = confusion_matrix(y_true, y_pred)

display_matrix = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Benign",
        "Skin Cancer"
    ]
)

fig, ax = plt.subplots(figsize=(6, 6))

display_matrix.plot(
    cmap="Reds",
    ax=ax,
    colorbar=True
)

plt.title("Baseline Model Confusion Matrix")
plt.tight_layout()
# Save image
plt.savefig(
    "results/baseline_confusion_matrix.png",
    dpi=300
)
plt.show()
print(
    "Confusion matrix saved to "
    "results/baseline_confusion_matrix.png"
)