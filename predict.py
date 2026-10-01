"""
Real-Time Sign Language Prediction and Text/Speech Desktop System
Uses OpenCV, MediaPipe, Random Forest classifier, and Temporal Smoothing.
Provides real-time webcam inference with keyboard-controlled text/sentence formation
and speech output.
"""

import cv2
import mediapipe as mp
import numpy as np

from utils import (
    predict_single_hand,
    load_trained_model,
    TemporalSmoother,
    speak_text_local
)


def run_prediction():
    print("=" * 65)
    print(" REAL-TIME SIGN LANGUAGE RECOGNITION SYSTEM (DESKTOP)")
    print("=" * 65)

    # 1. Load Model
    try:
        model, model_path = load_trained_model()
        print(f"Loaded trained model from: {model_path}")
        print(f"Supported classes: {list(model.classes_)}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Setup MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    # 3. Setup Temporal Smoother
    smoother = TemporalSmoother(window_size=7, min_consensus=4, confidence_threshold=0.70)

    # 4. Open Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera (index 0).")
        return

    recognized_text = ""
    print("\nControls:")
    print("  [A]     : Add Current Stable Sign to Text")
    print("  [SPACE] : Add Space (Separate Words)")
    print("  [D]     : Delete Last Character (Backspace)")
    print("  [C]     : Clear All Text")
    print("  [S]     : 🔊 Speak Recognized Text")
    print("  [Q]     : Quit Application")
    print("-" * 65)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab camera frame.")
            break

        # Mirror for natural interaction
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        raw_pred = None
        confidence = 0.0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw skeleton
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Extract normalized 63 features and predict via unified pipeline
                raw_pred, confidence, _, _ = predict_single_hand(model, hand_landmarks)

        # Update Temporal Smoother
        smooth_result = smoother.add(raw_pred, confidence)
        stable_sign = smooth_result["stable_sign"]
        is_stable = smooth_result["is_stable"]
        status_msg = smooth_result["status"]

        # ----------------------------------------------------
        # Draw Professional HUD
        # ----------------------------------------------------
        # Top banner for prediction
        cv2.rectangle(frame, (0, 0), (w, 110), (25, 25, 25), -1)

        if raw_pred is not None and confidence >= 0.70:
            pred_text = f"Sign: {raw_pred} ({confidence * 100:.1f}%)"
            color = (0, 255, 0) if is_stable else (0, 255, 255)
        elif raw_pred is not None:
            pred_text = f"Uncertain ({confidence * 100:.1f}%)"
            color = (0, 165, 255)
        else:
            pred_text = "No Hand Detected"
            color = (150, 150, 150)

        cv2.putText(frame, pred_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(frame, f"Filter Status: {status_msg}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        if stable_sign:
            cv2.putText(
                frame,
                f"STABLE: [{stable_sign}] (Press 'A' to commit)",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 120),
                2
            )

        # Bottom banner for Recognized Text
        cv2.rectangle(frame, (0, h - 80), (w, h), (15, 15, 15), -1)
        display_sentence = recognized_text if recognized_text else "[Text will appear here]"
        cv2.putText(frame, f"Text: {display_sentence}", (20, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        cv2.putText(
            frame,
            "Keys: [A] Commit Sign  |  [SPACE] Space  |  [D] Backspace  |  [C] Clear  |  [S] Speak  |  [Q] Quit",
            (20, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1
        )

        cv2.imshow("Real-Time Sign Language System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('a') and stable_sign:
            recognized_text += stable_sign
            print(f"Committed: '{stable_sign}' -> Full text: '{recognized_text}'")
        elif key == 32:  # Space
            recognized_text += " "
            print(f"Added space -> Full text: '{recognized_text}'")
        elif key == ord('d') or key == 8:  # Backspace / 'd'
            if recognized_text:
                recognized_text = recognized_text[:-1]
                print(f"Deleted character -> Full text: '{recognized_text}'")
        elif key == ord('c'):  # Clear
            recognized_text = ""
            print("Cleared recognized text.")
        elif key == ord('s'):  # Speak
            if recognized_text.strip():
                print(f"Speaking: '{recognized_text}'")
                speak_text_local(recognized_text)
            else:
                print("No text to speak.")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Application closed.")


if __name__ == "__main__":
    run_prediction()
