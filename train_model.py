"""
DeepFold - Model Training & Evaluation Pipeline
Trains balanced multi-class classification models (Multinomial Logistic Regression & Small MLP)
to predict Texas Hold'em showdown outcomes (Win, Loss, Tie).
Evaluates with Stratified 80/20 train/test split, reporting confusion matrices and metrics.
Saves production pipeline to models/deepfold_model.joblib and metadata to models/model_metadata.json.
"""

import os
import json
import time
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from features import extract_features_from_df, FEATURE_NAMES


def print_confusion_matrix_pretty(cm: np.ndarray, labels: list):
    """Prints a clean ASCII confusion matrix table."""
    header = f"{'Actual \\ Pred':<15}" + "".join([f"{l:>10}" for l in labels])
    print(header)
    print("-" * len(header))
    for i, actual in enumerate(labels):
        row = f"{actual:<15}" + "".join([f"{cm[i, j]:>10,}" for j in range(len(labels))])
        print(row)
    print()


def train_and_evaluate(
    data_path: str = "data/poker_simulation_100k.csv",
    model_dir: str = "models",
    preferred_model: str = "mlp"
):
    os.makedirs(model_dir, exist_ok=True)
    
    print("=" * 60)
    print("DEEPFOLD MODEL TRAINING & EVALUATION PIPELINE")
    print("=" * 60)
    
    # 1. Load simulation data
    print(f"[*] Loading dataset from '{data_path}'...")
    df = pd.read_csv(data_path)
    print(f"[+] Loaded {len(df):,} showdown rows.")
    
    # 2. Extract features
    print("[*] Extracting domain-informed feature vectors...")
    t0 = time.time()
    X = extract_features_from_df(df)
    t1 = time.time()
    print(f"[+] Extracted {X.shape[1]} features for {len(X):,} rows in {t1 - t0:.2f}s.")
    print(f"    Feature names: {FEATURE_NAMES}")
    
    y = df["outcome"]
    classes = ["Loss", "Tie", "Win"]
    
    # 3. Stratified 80/20 train/test split
    print("\n[*] Splitting data (80% Train / 20% Test, stratified by class)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[+] Training set: {len(X_train):,} rows | Test set: {len(X_test):,} rows")
    
    # 4. Model 1: Multinomial Logistic Regression (explainable baseline with class_weight='balanced')
    print("\n" + "-" * 60)
    print("MODEL 1: Multinomial Logistic Regression (class_weight='balanced')")
    print("-" * 60)
    
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ])
    
    t_start = time.time()
    lr_pipeline.fit(X_train, y_train)
    t_train = time.time() - t_start
    print(f"[+] Logistic Regression trained in {t_train:.2f}s.")
    
    y_pred_lr = lr_pipeline.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    macro_f1_lr = f1_score(y_test, y_pred_lr, average="macro")
    
    print(f"\nLogistic Regression Test Accuracy: {acc_lr * 100:.2f}% | Macro F1: {macro_f1_lr:.4f}")
    print("\nClassification Report (Logistic Regression):")
    print(classification_report(y_test, y_pred_lr, labels=classes, target_names=classes, digits=4))
    
    cm_lr = confusion_matrix(y_test, y_pred_lr, labels=classes)
    print("Confusion Matrix (Logistic Regression):")
    print_confusion_matrix_pretty(cm_lr, classes)
    
    # 5. Model 2: Small MLP (Multi-Layer Perceptron: 64 -> 32 hidden units with balanced sample weights)
    print("-" * 60)
    print("MODEL 2: Small MLP Classifier (64x32 hidden layers, balanced weights)")
    print("-" * 60)
    
    # Compute balanced sample weights for training MLP
    sample_weights_train = compute_sample_weight("balanced", y_train)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=200,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=42
    )
    
    t_start = time.time()
    mlp.fit(X_train_scaled, y_train, sample_weight=sample_weights_train)
    t_train_mlp = time.time() - t_start
    print(f"[+] MLP Classifier trained in {t_train_mlp:.2f}s.")
    
    y_pred_mlp = mlp.predict(X_test_scaled)
    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    macro_f1_mlp = f1_score(y_test, y_pred_mlp, average="macro")
    
    print(f"\nMLP Classifier Test Accuracy: {acc_mlp * 100:.2f}% | Macro F1: {macro_f1_mlp:.4f}")
    print("\nClassification Report (MLP Classifier):")
    print(classification_report(y_test, y_pred_mlp, labels=classes, target_names=classes, digits=4))
    
    cm_mlp = confusion_matrix(y_test, y_pred_mlp, labels=classes)
    print("Confusion Matrix (MLP Classifier):")
    print_confusion_matrix_pretty(cm_mlp, classes)
    
    # Create complete MLP pipeline
    mlp_pipeline = Pipeline([
        ("scaler", scaler),
        ("clf", mlp)
    ])
    
    # 6. Select and serialize best model
    if preferred_model == "lr" or (macro_f1_lr > macro_f1_mlp and preferred_model != "mlp"):
        selected_name = "Logistic Regression"
        selected_pipeline = lr_pipeline
        selected_acc = acc_lr
        selected_f1 = macro_f1_lr
        selected_report = classification_report(y_test, y_pred_lr, labels=classes, target_names=classes, output_dict=True)
    else:
        selected_name = "MLP Classifier (64x32)"
        selected_pipeline = mlp_pipeline
        selected_acc = acc_mlp
        selected_f1 = macro_f1_mlp
        selected_report = classification_report(y_test, y_pred_mlp, labels=classes, target_names=classes, output_dict=True)
        
    model_save_path = os.path.join(model_dir, "deepfold_model.joblib")
    joblib.dump(selected_pipeline, model_save_path)
    print("=" * 60)
    print(f"[+] Selected Model: {selected_name}")
    print(f"[+] Serialized model pipeline saved to '{model_save_path}'")
    
    # Also save metadata for reproducibility and inspection
    metadata = {
        "model_name": selected_name,
        "feature_names": FEATURE_NAMES,
        "classes": list(selected_pipeline.classes_),
        "test_accuracy": float(selected_acc),
        "test_macro_f1": float(selected_f1),
        "classification_report": selected_report,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    metadata_path = os.path.join(model_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[+] Model metadata saved to '{metadata_path}'")
    print("=" * 60 + "\n")
    
    return selected_pipeline, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepFold Model Training")
    parser.add_argument("--data", type=str, default="data/poker_simulation_100k.csv")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "lr"])
    args = parser.parse_args()
    
    train_and_evaluate(data_path=args.data, model_dir=args.model_dir, preferred_model=args.model)
