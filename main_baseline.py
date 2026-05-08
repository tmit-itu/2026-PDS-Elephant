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