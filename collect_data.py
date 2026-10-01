"""
Sign Language Data Collection Tool
Collects 21 hand landmark coordinates (63 features) using MediaPipe Hands.
Saves raw coordinates to dataset/{LABEL}.csv for subsequent training.
"""

import os
import csv
import cv2
import mediapipe as mp
from utils import extract_raw_landmarks


def run_data_collection():
    print("=" * 60)
    print(" SIGN LANGUAGE DATA COLLECTION TOOL")
    print("=" * 60)

    label = input("Enter the sign label to collect (e.g., A, B, C, F): ").strip().upper()
    if not label:
        print("Error: Label cannot be empty.")
        return

    os.makedirs("dataset", exist_ok=True)
    file_path = os.path.join("dataset", f"{label}.csv")

    existing_count = 0
    file_mode = "a"

    if os.path.exists(file_path):
        with open(file_path, "r", newline="") as f:
            existing_count = sum(1 for _ in f)
        print(f"File '{file_path}' already exists with {existing_count} samples.")
        choice = input("Do you want to (A)ppend to it or (O)verwrite? [A/O, default A]: ").strip().lower()
        if choice == "o":
            file_mode = "w"
            existing_count = 0
            print("File will be overwritten.")
        else:
            file_mode = "a"
            print("New samples will be appended.")
    else:
        file_mode = "w"

    # Setup MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera (index 0). Please check camera connection.")
        return

    print("\nControls:")
    print("  [SPACE] or [S] : Toggle Recording (Start / Pause)")
    print("  [Q]            : Quit and Save")
    print("-" * 60)

    is_recording = False
    session_collected = 0

    with open(file_path, file_mode, newline="") as f:
        writer = csv.writer(f)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from camera.")
                break

            # Mirror frame for intuitive interaction
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            hand_detected = False

            if results.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw visual skeleton
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # Record sample if active
                    if is_recording:
                        raw_coords = extract_raw_landmarks(hand_landmarks)
                        writer.writerow([label] + raw_coords)
                        session_collected += 1

            # On-screen HUD Overlay
            status_text = "RECORDING" if is_recording else "PAUSED"
            status_color = (0, 0, 255) if is_recording else (0, 255, 255)

            # Top background banner
            cv2.rectangle(frame, (0, 0), (w, 80), (30, 30, 30), -1)

            cv2.putText(
                frame,
                f"Sign: [{label}]  |  Status: {status_text}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2
            )

            cv2.putText(
                frame,
                f"Session: {session_collected}  |  Total: {existing_count + session_collected}  |  Hand: {'YES' if hand_detected else 'NO'}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1
            )

            # Bottom help text
            cv2.putText(
                frame,
                "Press 'S' or [SPACE] to Toggle Recording  |  Press 'Q' to Quit",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1
            )

            cv2.imshow("Sign Language Data Collector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s') or key == 32:  # 's' or Space
                is_recording = not is_recording

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    total_final = existing_count + session_collected
    print("\n" + "=" * 60)
    print(f"Data collection ended for sign '{label}'.")
    print(f"New samples collected in this session: {session_collected}")
    print(f"Total samples now in {file_path}: {total_final}")
    print("=" * 60)


if __name__ == "__main__":
    run_data_collection()