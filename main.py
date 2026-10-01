"""
Sign Language to Text and Speech - Diagnostic and Quick Launcher
Performs environment check, camera validation, and launches hand tracking test.
"""

import os
import cv2
import mediapipe as mp
from utils import load_trained_model, extract_landmarks


def run_diagnostic():
    print("=" * 65)
    print(" SIGN LANGUAGE SYSTEM DIAGNOSTIC & LAUNCHER")
    print("=" * 65)

    model = None
    try:
        model, model_path = load_trained_model()
        print(f"[OK] Trained model found: {model_path}")
        print(f"     Classes: {list(model.classes_)}")
    except Exception as e:
        print(f"[WARNING] Model check: {e}")
        print("          Run 'python train_model.py' to generate model.")

    # Check dataset
    csv_count = len([f for f in os.listdir("dataset") if f.endswith(".csv")]) if os.path.exists("dataset") else 0
    print(f"[OK] Dataset files: {csv_count} CSV classes found.")

    print("\nAvailable Run Commands:")
    print("  1. Web App (Streamlit) : python -m streamlit run app.py")
    print("  2. Desktop Live App    : python predict.py")
    print("  3. Collect New Data    : python collect_data.py")
    print("  4. Retrain Model       : python train_model.py")
    print("  5. Run Parity Tests    : python test_preprocessing_parity.py")
    print("-" * 65)
    print("Starting Camera & Hand Landmark Test (Press 'Q' in camera window to exit)...")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera (index 0) could not be opened. Diagnostic aborted.")
        return

    from utils import predict_single_hand

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )
                if model is not None:
                    try:
                        pred_sign, conf, _, _ = predict_single_hand(model, hand_landmarks)
                        color = (0, 255, 0) if conf >= 0.70 else (0, 165, 255)
                        cv2.putText(
                            frame,
                            f"Live Sign: {pred_sign} ({conf * 100:.1f}%)",
                            (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            color,
                            2
                        )
                    except Exception:
                        pass

        cv2.putText(
            frame,
            "Hand Tracking Test (Press 'Q' to Exit)",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("Sign Language System - Diagnostic", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Diagnostic closed.")


if __name__ == "__main__":
    run_diagnostic()