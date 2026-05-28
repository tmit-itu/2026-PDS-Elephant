import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import numpy as np


def get_features(data):
    """
    Select feature columns from the dataset.

    Excludes identifiers and target variable, keeping only numerical features.

    Parameters:
        data (DataFrame): dataset with features and labels

    Returns:
        list: names of feature columns
    """

    ignore_columns = ["img_id", "patient_id", "cancer"]
    return [col for col in data.columns if col not in ignore_columns]


def split_by_patient(data, features, test_size=0.2):
    """
    Split data into train and test sets based on patient IDs.

    Ensures that images from the same patient do not appear in both
    training and testing sets, preventing data leakage.

    Parameters:
        data (DataFrame): full dataset
        features (list): feature column names
        test_size (float): proportion of patients used for testing

    Returns:
        data, X, y, train_idx, test_idx
    """

    # Prepare data and labels
    data = data.dropna().copy()
    data["cancer"] = data["cancer"].astype(int)

    X = data[features]
    y = data["cancer"]
    patients = data["patient_id"]

    # Split unique patients into train/test
    unique_patients = patients.unique()
    train_patients, test_patients = train_test_split(
        unique_patients,
        test_size=test_size,
        random_state=42
    )

    # Create boolean indices for splitting
    train_idx = patients.isin(train_patients)
    test_idx = patients.isin(test_patients)

    return data, X, y, train_idx, test_idx


def cross_validate_tree_depth(data, features, max_depth=10):
    """
    Select optimal Decision Tree depth using GroupKFold cross-validation.

    The evaluation is performed on the training set with patient-level
    grouping to avoid data leakage.

    Parameters:
        data (DataFrame): dataset
        features (list): feature column names
        max_depth (int): maximum tree depth to test

    Returns:
        best_depth (int): selected tree depth
        results_df (DataFrame): cross-validation results
    """

    data, X, y, train_idx, _ = split_by_patient(data, features)

    X_train = X[train_idx]
    y_train = y[train_idx]
    groups_train = data.loc[train_idx, "patient_id"]

    cv = GroupKFold(n_splits=5)
    results = []

    # Evaluate tree depth from 1 to max_depth
    for depth in range(1, max_depth + 1):
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)

        auc_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            groups=groups_train,
            scoring="roc_auc"
        )

        results.append({
            "max_depth": depth,
            "mean_auc": np.mean(auc_scores),
            "std_auc": np.std(auc_scores)
        })

    results_df = pd.DataFrame(results)

    # Select depth with highest mean AUC
    best_row = results_df.loc[results_df["mean_auc"].idxmax()]
    best_depth = int(best_row["max_depth"])

    print("\nCross-validation results:")
    print(results_df)

    print(f"\nBest tree depth: {best_depth}")
    print(f"Best mean AUC: {best_row['mean_auc']:.4f} ± {best_row['std_auc']:.4f}")

    return best_depth, results_df


def train_model(data, features, max_depth=4, model_path=None):
    """
    Train a Decision Tree classifier using the training dataset.

    Optionally saves the trained model to disk.

    Parameters:
        data (DataFrame): dataset
        features (list): feature column names
        max_depth (int): tree depth
        model_path (str, optional): path to save model

    Returns:
        trained model
    """

    data, X, y, train_idx, _ = split_by_patient(data, features)
    X_train = X[train_idx]
    y_train = y[train_idx]

    # Train model
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

# Save model if path provided
    if model_path is not None:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Saved model to {model_path}")

    return model


def evaluate_model(data, features, model, result_dir, output_name):
    """
    Evaluate trained model on the test set and save predictions.

    Computes standard evaluation metrics and stores predictions
    for further analysis.

    Parameters:
        data (DataFrame): dataset
        features (list): feature column names
        model: trained classifier
        result_dir (str): directory to save outputs
        output_name (str): name of prediction CSV

    Returns:
        dict: evaluation metrics
    """

    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    data, X, y, _, test_idx = split_by_patient(data, features)

    X_test = X[test_idx]
    y_test = y[test_idx]

    # Generate predictions and probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\nModel evaluation:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"F1 score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save predictions to CSV
    results = pd.DataFrame({
        "img_id": data.loc[test_idx, "img_id"].values,
        "patient_id": data.loc[test_idx, "patient_id"].values,
        "true_label": y_test.values,
        "predicted_label": y_pred,
        "probability_cancer": y_prob
    })

    results.to_csv(result_dir / output_name, index=False)

    print(f"\nSaved predictions to: {result_dir / output_name}")

    return {
        "accuracy": acc,
        "recall": rec,
        "precision": prec,
        "f1": f1,
        "auc": auc
    }