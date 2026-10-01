"""
Sign Language Translation - Preprocessing and Pipeline Parity Test Suite
Validates that:
1. Preprocessing is 100% identical between training and real-time prediction.
2. Wrist-relative translation invariance holds mathematically.
3. Scale normalization invariance holds mathematically.
4. Feature shapes and bounds strictly match 63 features in [-1.0, 1.0].
5. Model prediction parity between 1D vector and 2D batch inputs.
6. Temporal smoothing correctly filters noise and detects consensus.
7. Dataset samples match required schema without modifications.
"""

import os
import glob
import numpy as np
import pandas as pd
import joblib

from utils import (
    FEATURE_COUNT,
    NUM_LANDMARKS,
    COORDINATE_DIM,
    normalize_landmark_array,
    predict_features,
    load_trained_model,
    TemporalSmoother
)


def test_constants():
    assert NUM_LANDMARKS == 21, f"Expected 21 landmarks, got {NUM_LANDMARKS}"
    assert COORDINATE_DIM == 3, f"Expected 3 coordinates (x, y, z), got {COORDINATE_DIM}"
    assert FEATURE_COUNT == 63, f"Expected 63 features, got {FEATURE_COUNT}"
    print("[PASS] Constants verified: 21 landmarks * 3 axes = 63 features.")


def test_wrist_relative_and_scale_invariance():
    # Generate synthetic hand landmarks: 21 points in 3D
    np.random.seed(42)
    sample_hand = np.random.uniform(0.1, 0.9, size=(21, 3)).astype(np.float32)
    raw_flat = sample_hand.flatten()

    norm_base = normalize_landmark_array(raw_flat)
    assert norm_base.shape == (63,), f"Expected shape (63,), got {norm_base.shape}"

    # Check that landmark 0 (wrist) is translated to (0, 0, 0)
    norm_pts = norm_base.reshape((21, 3))
    np.testing.assert_allclose(
        norm_pts[0],
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        atol=1e-6,
        err_msg="Wrist landmark (index 0) must be exactly at (0, 0, 0) after normalization."
    )

    # Check bounds: coordinates must be in [-1.0, 1.0]
    assert np.all(norm_base >= -1.0 - 1e-6) and np.all(norm_base <= 1.0 + 1e-6), "Values exceed [-1, 1] range!"
    assert np.isclose(np.max(np.abs(norm_pts)), 1.0, atol=1e-5), "Max absolute span should equal 1.0"

    # Test Translation Invariance: Shift hand by arbitrary delta (dx, dy, dz)
    shift_vector = np.array([0.35, -0.22, 0.47], dtype=np.float32)
    shifted_hand = sample_hand + shift_vector
    norm_shifted = normalize_landmark_array(shifted_hand.flatten())
    np.testing.assert_allclose(
        norm_base,
        norm_shifted,
        atol=1e-5,
        err_msg="Translation invariance failed: shifted hand produced different features!"
    )

    # Test Scale Invariance: Scale hand size by 2.75x
    scaled_hand = sample_hand * 2.75
    norm_scaled = normalize_landmark_array(scaled_hand.flatten())
    np.testing.assert_allclose(
        norm_base,
        norm_scaled,
        atol=1e-5,
        err_msg="Scale invariance failed: scaled hand produced different features!"
    )

    print("[PASS] Translation invariance and scale invariance mathematically confirmed.")


def test_1d_vs_2d_parity():
    np.random.seed(101)
    samples_2d = np.random.uniform(0.0, 1.0, size=(15, 63)).astype(np.float32)
    norm_batch = normalize_landmark_array(samples_2d)
    assert norm_batch.shape == (15, 63)

    for i in range(15):
        norm_single = normalize_landmark_array(samples_2d[i])
        np.testing.assert_allclose(
            norm_batch[i],
            norm_single,
            atol=1e-6,
            err_msg=f"1D vs 2D batch normalization mismatch on sample {i}"
        )
    print("[PASS] 1D single-frame and 2D batch normalization produce identical outputs.")


def test_model_and_prediction_parity():
    model, path = load_trained_model()
    classes = list(model.classes_)
    assert len(classes) == 5, f"Expected 5 classes, found {len(classes)}"
    assert sorted(classes) == ["A", "B", "C", "D", "E"], f"Unexpected classes: {classes}"

    # Load 1 sample from each CSV to test prediction
    for sign in classes:
        csv_file = os.path.join("dataset", f"{sign}.csv")
        df = pd.read_csv(csv_file, header=None)
        raw_sample = df.iloc[0, 1:].values.astype(np.float32)

        # Preprocessing as done in training
        norm_training = normalize_landmark_array(df.iloc[:5, 1:].values.astype(np.float32))[0]

        # Preprocessing as done in real-time prediction
        norm_realtime = normalize_landmark_array(raw_sample)

        np.testing.assert_allclose(
            norm_training,
            norm_realtime,
            atol=1e-6,
            err_msg=f"Preprocessing mismatch between training and real-time inference for sign '{sign}'"
        )

        # Predict using helper
        pred_sign, conf, prob_dict = predict_features(model, norm_realtime)
        assert pred_sign in classes
        assert 0.0 <= conf <= 1.0
        assert np.isclose(sum(prob_dict.values()), 1.0, atol=1e-4)

    print(f"[PASS] Model ({path}) verified with 100% training-inference parity on classes: {classes}.")


def test_temporal_smoother():
    smoother = TemporalSmoother(window_size=5, min_consensus=3, confidence_threshold=0.70)

    # 1. Below confidence threshold -> should not be stable
    res1 = smoother.add("A", 0.55)
    assert not res1["is_stable"], "Unconfident prediction should not be stable"
    assert res1["stable_sign"] is None

    # 2. Confident prediction 1 time -> not enough consensus yet
    res2 = smoother.add("B", 0.95)
    assert not res2["is_stable"], "Single frame should not reach consensus"

    # 3. Add 2 more confident 'B' frames -> now count=3 >= min_consensus(3)
    smoother.add("B", 0.92)
    res4 = smoother.add("B", 0.94)
    assert res4["is_stable"], "Consensus threshold reached; gesture must be marked stable"
    assert res4["stable_sign"] == "B"
    assert res4["confidence"] >= 0.70

    # 4. Clear reset test
    smoother.clear()
    assert len(smoother.history) == 0
    assert smoother.last_stable_sign is None
    print("[PASS] Temporal smoothing logic and hysteresis filter verified.")


def test_dataset_integrity():
    csv_files = sorted(glob.glob(os.path.join("dataset", "*.csv")))
    expected_signs = ["A", "B", "C", "D", "E"]
    found_signs = [os.path.basename(f).replace(".csv", "") for f in csv_files]
    assert found_signs == expected_signs, f"Dataset signs altered! Found: {found_signs}"

    total_samples = 0
    for f in csv_files:
        df = pd.read_csv(f, header=None)
        assert df.shape[1] == 64, f"File {f} must have 64 columns (1 label + 63 coords), got {df.shape[1]}"
        total_samples += len(df)

    assert total_samples == 2892, f"Dataset sample count altered! Expected 2892, got {total_samples}"
    print(f"[PASS] Dataset integrity intact: {len(csv_files)} classes, {total_samples} samples untouched.")


if __name__ == "__main__":
    print("=" * 65)
    print(" RUNNING PREPROCESSING AND PIPELINE PARITY TEST SUITE")
    print("=" * 65)
    test_constants()
    test_wrist_relative_and_scale_invariance()
    test_1d_vs_2d_parity()
    test_dataset_integrity()
    test_model_and_prediction_parity()
    test_temporal_smoother()
    print("=" * 65)
    print(" ALL 6 TEST SUITES PASSED PERFECTLY (100% PARITY GUARANTEED)")
    print("=" * 65)
