import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix
)

from src.classifier import (
    get_features,
    cross_validate_tree_depth,
    train_model,
    evaluate_model
)

# Load data
data = pd.read_csv("results/features_extended.csv")

# Get feature columns
features = get_features(data)

# Find best tree depth
best_depth, cv_results = cross_validate_tree_depth(
    data=data,
    features=features,
    max_depth=10
)

# Save cross-validation results
cv_results.to_csv(
    "results/cross_validation_extended.csv",
    index=False
)

# Train model
model = train_model(
    data=data,
    features=features,
    max_depth=best_depth
)

# Generate predictions CSV
evaluate_model(
    data=data,
    features=features,
    model=model,
    result_dir="results",
    output_name="predictions_extended.csv"
)

# -------------------------
# Create summary CSV
# -------------------------

# Load predictions
df = pd.read_csv("results/predictions_extended.csv")

y_true = df["true_label"]
y_pred = df["predicted_label"]

# Metrics
accuracy = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred
).ravel()

# Save summary
summary = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Recall",
        "Precision",
        "F1 Score",
        "TP",
        "FP",
        "TN",
        "FN"
    ],
    "Value": [
        accuracy,
        recall,
        precision,
        f1,
        tp,
        fp,
        tn,
        fn
    ]
})

summary.to_csv(
    "results/extended_metrics_summary.csv",
    index=False
)

print("Summary saved to results/extended_metrics_summary.csv")
