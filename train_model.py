"""
Sign Language Model Training Script
Trains a Random Forest classifier on landmark features extracted from dataset/*.csv.
Applies wrist-relative and scale normalization via utils.py to ensure 100% parity
with real-time inference.
"""

import os
import glob
import json
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

from utils import normalize_landmark_array, FEATURE_COUNT


def train_and_evaluate():
    print("=" * 65)
    print(" SIGN LANGUAGE MODEL TRAINING PIPELINE")
    print("=" * 65)

    # 1. Locate dataset files
    dataset_dir = "dataset"
    csv_files = sorted(glob.glob(os.path.join(dataset_dir, "*.csv")))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{dataset_dir}/'. "
            "Please collect data using collect_data.py first."
        )

    print(f"\n[1/5] Loading datasets from '{dataset_dir}/':")
    data_frames = []
    for f in csv_files:
        df = pd.read_csv(f, header=None)
        label = df.iloc[0, 0]
        print(f"  - {os.path.basename(f)}: Label '{label}', Samples: {len(df)}, Columns: {df.shape[1]}")
        data_frames.append(df)

    dataset = pd.concat(data_frames, ignore_index=True)
    print(f"  Total samples loaded: {len(dataset)}")

    # 2. Extract labels and raw features
    y = dataset.iloc[:, 0].astype(str).values
    X_raw = dataset.iloc[:, 1:].values.astype(np.float32)

    if X_raw.shape[1] != FEATURE_COUNT:
        raise ValueError(
            f"Expected {FEATURE_COUNT} landmark coordinates (21 landmarks x 3 axes), "
            f"but found {X_raw.shape[1]} columns."
        )

    # 3. Apply Wrist-Relative and Scale Normalization
    print("\n[2/5] Applying landmark normalization (wrist-relative & scale invariant)...")
    X_normalized = normalize_landmark_array(X_raw)
    print("  Feature extraction normalized: 63 coordinates transformed to invariant scale [-1, 1].")

    # 4. Stratified Train / Test Split
    print("\n[3/5] Splitting dataset (80% Train, 20% Test) with stratification...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"  Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 5. Train Random Forest Classifier
    print("\n[4/5] Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 6. Comprehensive Evaluation
    print("\n[5/5] Evaluating Model Performance...")
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    classes = sorted(list(set(y)))

    print("\n" + "-" * 40)
    print(f" OVERALL TEST ACCURACY: {test_accuracy * 100:.2f}%")
    print("-" * 40)

    print("\nClassification Report (Precision, Recall, F1-Score):")
    print(classification_report(y_test, y_pred, target_names=classes, digits=4))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in classes], columns=[f"Pred_{c}" for c in classes])
    print(cm_df)

    # 5-fold cross-validation
    print("\nRunning 5-Fold Stratified Cross-Validation on entire normalized dataset...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_normalized, y, cv=cv, scoring="accuracy")
    print(f"Cross-Validation Scores: {[round(s, 4) for s in cv_scores]}")
    print(f"Mean CV Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

    # 7. Save Model & Metadata
    os.makedirs("models", exist_ok=True)
    primary_model_path = os.path.join("models", "sign_language_model.pkl")
    fallback_model_path = "sign_language_model.pkl"

    joblib.dump(model, primary_model_path)
    joblib.dump(model, fallback_model_path)
    print(f"\nModel saved successfully to:")
    print(f"  - {primary_model_path}")
    print(f"  - {fallback_model_path} (root fallback)")

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "classes": list(model.classes_),
        "num_classes": len(model.classes_),
        "feature_count": int(model.n_features_in_),
        "normalization": "wrist_relative_scale_invariant",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_accuracy": float(test_accuracy),
        "cross_val_mean": float(cv_scores.mean()),
        "cross_val_std": float(cv_scores.std()),
        "model_type": "RandomForestClassifier",
        "n_estimators": 100
    }

    metadata_path = os.path.join("models", "model_metadata.json")
    with open(metadata_path, "w") as mf:
        json.dump(metadata, mf, indent=4)
    print(f"Training metadata saved to {metadata_path}")
    print("=" * 65)

    return model, metadata


if __name__ == "__main__":
    train_and_evaluate()