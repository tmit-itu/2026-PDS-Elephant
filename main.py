import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
)
from src.classifier import get_features, evaluate_model

# ----- configuration -----
# Set load_model to True to load the saved model from disk
# and produce predictions without retraining.
load_model = True

# Which model to load: "baseline" or "extended"
mode = "extended"
# -------------------------

features_csv = f"data/features_{mode}.csv"
model_path = f"results/models/tree_{mode}.pkl"

print(f"Mode: {mode}")
print(f"load_model: {load_model}")
print(f"Reading features from: {features_csv}")
print(f"Loading model from: {model_path}")

# Load features
data = pd.read_csv(features_csv)
features = get_features(data)

# Load the trained model
model = joblib.load(model_path)

# Evaluate and write predictions
evaluate_model(
    data=data,
    features=features,
    model=model,
    result_dir="results/predictions/",
    output_name=f"predictions_{mode}_loaded.csv",
)

#print a quick metrics summary
df = pd.read_csv(f"results/predictions/predictions_{mode}_loaded.csv")
y_true = df["true_label"]
y_pred = df["predicted_label"]

acc = accuracy_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print(f"\n{mode.title()} model results (loaded from disk):")
print(f"  Accuracy:  {acc:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  F1:        {f1:.4f}")
print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")