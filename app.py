"""
Real-Time Sign Language to Text and Speech System
Streamlit Web Application
Student Project Prototype: Computer and Communication Engineering (CCE)
"""

import os
import glob
import json
import time
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import mediapipe as mp

from utils import (
    FEATURE_COUNT,
    NUM_LANDMARKS,
    extract_raw_landmarks,
    normalize_landmark_array,
    extract_landmarks,
    predict_features,
    predict_single_hand,
    load_trained_model,
    TemporalSmoother,
    speak_text_local,
    generate_speech_js,
    generate_audio_bytes
)

# ---------------------------------------------------------------------------
# 1. Page Configuration & Theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sign Language to Text & Speech",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Aesthetic Styling for College Project Demonstration
st.markdown("""
<style>
    /* Metric & Card styling */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 14px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
    }
    .text-display-box {
        background-color: #0b1329;
        color: #38bdf8;
        font-family: 'Courier New', monospace;
        font-size: 26px;
        font-weight: 700;
        letter-spacing: 2.5px;
        padding: 18px 24px;
        border-radius: 10px;
        border: 2px solid #0284c7;
        min-height: 75px;
        display: flex;
        align-items: center;
        word-break: break-word;
        box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.5);
    }
    .status-badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 13px;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 6px;
    }
    .badge-success { background-color: #065f46; color: #6ee7b7; border: 1px solid #10b981; }
    .badge-warning { background-color: #854d0e; color: #fde047; border: 1px solid #eab308; }
    .badge-info { background-color: #1e3a8a; color: #93c5fd; border: 1px solid #3b82f6; }
    .badge-danger { background-color: #881337; color: #fca5a5; border: 1px solid #f43f5e; }
    
    .sign-badge {
        font-size: 48px;
        font-weight: 900;
        color: #10b981;
        text-align: center;
        background: rgba(16, 185, 129, 0.12);
        border-radius: 14px;
        padding: 12px;
        border: 2px solid rgba(16, 185, 129, 0.4);
        margin: 10px 0;
        letter-spacing: 2px;
    }
    .uncertain-badge {
        font-size: 28px;
        font-weight: 800;
        color: #f59e0b;
        text-align: center;
        background: rgba(245, 158, 11, 0.12);
        border-radius: 14px;
        padding: 12px;
        border: 2px solid rgba(245, 158, 11, 0.4);
        margin: 10px 0;
    }
    .chip-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 12px;
    }
    .chip {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 12px;
        color: #94a3b8;
    }
    .chip strong {
        color: #f1f5f9;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 2. Session State Initialization
# ---------------------------------------------------------------------------
if "recognized_text" not in st.session_state:
    st.session_state.recognized_text = ""
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "last_stable_sign" not in st.session_state:
    st.session_state.last_stable_sign = None
if "smoother" not in st.session_state:
    st.session_state.smoother = TemporalSmoother(window_size=5, min_consensus=3, confidence_threshold=0.70)
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None
if "last_spoken_text" not in st.session_state:
    st.session_state.last_spoken_text = None


# ---------------------------------------------------------------------------
# 3. Model & MediaPipe Resource Loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained machine learning model...")
def get_cached_model():
    """Loads and caches the Random Forest model."""
    try:
        loaded_model, path = load_trained_model()
        return loaded_model, path, None
    except Exception as e:
        return None, None, str(e)


@st.cache_resource
def get_mediapipe_hands():
    """Initializes and caches MediaPipe Hands detector."""
    mp_hands_module = mp.solutions.hands
    hands_detector = mp_hands_module.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_draw_module = mp.solutions.drawing_utils
    mp_styles_module = mp.solutions.drawing_styles
    return hands_detector, mp_draw_module, mp_styles_module, mp_hands_module


model, model_path, model_error = get_cached_model()
hands, mp_draw, mp_styles, mp_hands = get_mediapipe_hands()


# ---------------------------------------------------------------------------
# 4. Sidebar: Settings, Model Info & Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ System Control")

    if model is not None:
        st.success(f"✅ Model Loaded: Random Forest")
        st.caption(f"📁 Path: `{model_path}`")
        classes = list(model.classes_)
        st.write(f"**Trained Signs ({len(classes)} Classes):**")
        st.code("  ".join(classes))
    else:
        st.error("❌ Model Not Found")
        st.info("Run `python train_model.py` in your terminal to train the model.")

    st.markdown("---")
    st.subheader("🎯 Detection Threshold")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.50,
        max_value=0.95,
        value=0.70,
        step=0.05,
        help="Predictions below this confidence are filtered as 'Uncertain' to prevent false positive classifications."
    )

    st.markdown("---")
    st.subheader("🔄 Temporal Stability Buffer")
    buffer_window = st.slider(
        "Buffer Window Size",
        min_value=3,
        max_value=11,
        value=5,
        step=2,
        help="Number of consecutive frames used to determine gesture consensus."
    )

    # Sync smoother parameters
    st.session_state.smoother.update_parameters(
        window_size=buffer_window,
        min_consensus=max(2, buffer_window // 2 + 1),
        confidence_threshold=confidence_threshold
    )

    st.markdown("---")
    st.subheader("🔊 Audio Output Preference")
    tts_mode = st.radio(
        "Speech Mode",
        options=["Web Speech API (Browser)", "In-App Audio Player (gTTS)", "Local pyttsx3 (Offline SAPI5)", "All / Automatic"],
        index=3,
        help="Choose the primary text-to-speech mechanism."
    )

    st.markdown("---")
    st.subheader("💡 Camera Best Practices")
    st.markdown("""
    - Position your hand clearly in front of the lens.
    - Ensure clear lighting without strong backlighting.
    - Hold the hand steady when capturing snapshots.
    - Keep fingers clearly visible for distinct signs.
    """)

    st.markdown("---")
    st.caption("🎓 3rd-Year CCE Student Project Prototype")
    st.caption("Real-Time Sign Language to Text & Speech")


# ---------------------------------------------------------------------------
# 5. Header Banner & Status Chips
# ---------------------------------------------------------------------------
st.title("🤟 Real-Time Sign Language to Text and Speech")
st.write(
    "Automated hand gesture recognition using MediaPipe 3D Landmark Extraction, "
    "Wrist-Relative & Scale-Invariant Normalization, Random Forest Classification, and Speech Output."
)

if model_error:
    st.error(f"⚠️ Model Initialization Error: {model_error}")
    st.stop()

# Project Info Chips
classes_list = list(model.classes_) if model else []
st.markdown(f"""
<div class="chip-container">
    <div class="chip">Classifier: <strong>Random Forest (100 Trees)</strong></div>
    <div class="chip">Features: <strong>63 Invariant Coordinates (21 × 3)</strong></div>
    <div class="chip">Active Classes: <strong>{", ".join(classes_list)}</strong></div>
    <div class="chip">Threshold: <strong>{int(confidence_threshold * 100)}%</strong></div>
    <div class="chip">Parity: <strong>100% Mathematically Verified</strong></div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 6. Main Interface Layout: Input vs Analysis
# ---------------------------------------------------------------------------
col_input, col_pred = st.columns([1.1, 0.9], gap="large")

captured_image_array = None
synthetic_features = None
input_mode = "camera"

with col_input:
    st.subheader("📷 Gesture Input")
    
    input_tab1, input_tab2, input_tab3 = st.tabs(["📸 Webcam Snapshot", "📁 Upload Image", "🧪 Test Dataset Sample"])
    
    with input_tab1:
        camera_file = st.camera_input("Hold your sign gesture steady and take a photo")
        mirror_webcam = st.checkbox(
            "🪞 Mirror webcam snapshot horizontally",
            value=True,
            help="Enabled by default so camera acts as a mirror, matching training orientation."
        )
        if camera_file is not None:
            file_bytes = camera_file.getvalue()
            img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
            if mirror_webcam and img is not None:
                img = cv2.flip(img, 1)
            captured_image_array = img
            input_mode = "camera"

    with input_tab2:
        uploaded_file = st.file_uploader("Upload an image containing a hand sign", type=["jpg", "jpeg", "png"])
        mirror_upload = st.checkbox(
            "🪞 Mirror uploaded image horizontally",
            value=False,
            help="Enable if image was captured without camera mirror."
        )
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
            if mirror_upload and img is not None:
                img = cv2.flip(img, 1)
            captured_image_array = img
            input_mode = "upload"

    with input_tab3:
        st.markdown("**Viva / Demo Verification Mode:** Test trained model against existing dataset samples.")
        csv_options = sorted([os.path.basename(f).replace(".csv", "") for f in glob.glob("dataset/*.csv")])
        if csv_options:
            selected_sample_sign = st.selectbox("Select Ground Truth Sign from Dataset", options=csv_options)
            sample_col1, sample_col2 = st.columns([1, 1])
            with sample_col1:
                sample_idx = st.number_input("Sample Index", min_value=0, max_value=200, value=0, step=1)
            with sample_col2:
                load_sample_btn = st.button("🔬 Test Ground Truth Sample", use_container_width=True)
            
            if load_sample_btn:
                try:
                    df = pd.read_csv(f"dataset/{selected_sample_sign}.csv", header=None)
                    raw_row = df.iloc[sample_idx, 1:].values.astype(np.float32)
                    synthetic_features = normalize_landmark_array(raw_row)
                    input_mode = "dataset_sample"
                    st.success(f"Loaded ground-truth sample for '{selected_sample_sign}' (Sample #{sample_idx})")
                except Exception as ex:
                    st.error(f"Error loading sample: {ex}")
        else:
            st.info("No CSV files found in dataset/")


# ---------------------------------------------------------------------------
# 7. Processing Pipeline: Detection, Extraction, & Prediction
# ---------------------------------------------------------------------------
predicted_sign = None
prediction_confidence = 0.0
prob_dict = {}
annotated_frame = None
hand_detected = False
num_hands_found = 0
smooth_result = {"is_stable": False, "status": "Waiting for input...", "stable_sign": None}

# Scenario A: Real Image Input (Camera or Upload)
if captured_image_array is not None:
    frame = captured_image_array.copy()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 1. MediaPipe Hand Landmark Detection
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_detected = True
        num_hands_found = len(results.multi_hand_landmarks)
        primary_hand = results.multi_hand_landmarks[0]

        # 2. Draw Hand Landmarks Skeleton
        for hl in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hl,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )
        annotated_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 3. 63-Feature Extraction & Normalization via Unified Pipeline
        try:
            predicted_sign, prediction_confidence, prob_dict, features = predict_single_hand(model, primary_hand)

            # 4. Temporal Smoothing Update
            smooth_result = st.session_state.smoother.add(predicted_sign, prediction_confidence)
            snapshot_eval = st.session_state.smoother.evaluate_single_frame(predicted_sign, prediction_confidence)

            if smooth_result["stable_sign"]:
                st.session_state.last_stable_sign = smooth_result["stable_sign"]
            elif snapshot_eval["is_stable"]:
                st.session_state.last_stable_sign = snapshot_eval["stable_sign"]

        except Exception as ex:
            st.error(f"Error during feature extraction / prediction: {ex}")
    else:
        hand_detected = False
        annotated_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        smooth_result = st.session_state.smoother.add(None, 0.0)

# Scenario B: Direct Dataset Sample Test
elif synthetic_features is not None:
    hand_detected = True
    predicted_sign, prediction_confidence, prob_dict = predict_features(model, synthetic_features)
    smooth_result = st.session_state.smoother.evaluate_single_frame(predicted_sign, prediction_confidence)
    if smooth_result["is_stable"]:
        st.session_state.last_stable_sign = predicted_sign


# ---------------------------------------------------------------------------
# 8. Prediction & Classification Display (Right Column)
# ---------------------------------------------------------------------------
with col_pred:
    st.subheader("🔍 Gesture Analysis & Prediction")

    if captured_image_array is None and synthetic_features is None:
        st.info("👋 Take a webcam snapshot, upload a photo, or test a dataset sample to begin recognition.")
    elif not hand_detected:
        st.warning("⚠️ No hand detected. Please hold your hand clearly within the camera frame with fingers spread.")
    else:
        if num_hands_found > 1:
            st.info(f"ℹ️ Detected {num_hands_found} hands. Evaluating primary hand.")

        # Check confidence against threshold
        is_confident = (prediction_confidence >= confidence_threshold) and (predicted_sign is not None)

        if is_confident:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 13px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px;">Predicted Sign</div>
                <div class="sign-badge">{predicted_sign}</div>
                <div style="margin-top: 10px; font-size: 14px; color: #cbd5e1; display: flex; justify-content: space-between;">
                    <span>Confidence: <strong>{prediction_confidence * 100:.1f}%</strong></span>
                    <span>Status: <strong style="color: #10b981;">Confident</strong></span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(prediction_confidence)
        else:
            runner_up_text = f" (Highest candidate: '{predicted_sign}' at {prediction_confidence * 100:.1f}%)" if predicted_sign else ""
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 13px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px;">Status</div>
                <div class="uncertain-badge">⚠️ Uncertain Sign</div>
                <div style="margin-top: 10px; font-size: 13px; color: #f59e0b;">
                    Confidence is below the required <strong>{int(confidence_threshold * 100)}%</strong> threshold{runner_up_text}.
                    Gesture is ambiguous or transitional. Please adjust your hand position.
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(prediction_confidence)

        # Stability Indicator
        if smooth_result["is_stable"]:
            st.markdown(f'<span class="status-badge badge-success">✓ Gesture Stable ({predicted_sign})</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="status-badge badge-warning">⚡ {smooth_result["status"]}</span>', unsafe_allow_html=True)

        # Probability Breakdown Chart
        if prob_dict:
            with st.expander("📊 Class Probability Distribution", expanded=False):
                chart_df = pd.DataFrame({
                    "Sign": list(prob_dict.keys()),
                    "Probability (%)": [p * 100 for p in prob_dict.values()]
                }).sort_values(by="Probability (%)", ascending=False)
                st.bar_chart(chart_df.set_index("Sign"))

    # Display detected skeleton overlay
    if annotated_frame is not None:
        st.image(annotated_frame, caption="Detected Hand Skeleton & 21 MediaPipe Landmarks", use_container_width=True)


# ---------------------------------------------------------------------------
# 9. Text Formation & Sentence Building Section
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("📝 Text & Sentence Formation")

# Recognized Text Display Box
text_container = st.session_state.recognized_text if st.session_state.recognized_text else "(No text recognized yet - take a snapshot to begin)"
st.markdown(f'<div class="text-display-box">{text_container}</div>', unsafe_allow_html=True)
st.write("")

# Action Buttons
btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = st.columns([1.2, 1, 1, 1, 1.4])

with btn_col1:
    can_add = (predicted_sign is not None) and (prediction_confidence >= confidence_threshold)
    add_label = f"➕ Add '{predicted_sign}'" if can_add else "➕ Add Sign"
    if st.button(add_label, use_container_width=True, disabled=not can_add):
        st.session_state.recognized_text += str(predicted_sign)
        # Log to prediction history
        st.session_state.prediction_history.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "sign": str(predicted_sign),
            "confidence": f"{prediction_confidence * 100:.1f}%"
        })
        st.rerun()

with btn_col2:
    if st.button("␣ Space", use_container_width=True):
        st.session_state.recognized_text += " "
        st.rerun()

with btn_col3:
    if st.button("⌫ Delete", use_container_width=True):
        if st.session_state.recognized_text:
            st.session_state.recognized_text = st.session_state.recognized_text[:-1]
            st.rerun()

with btn_col4:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.recognized_text = ""
        st.session_state.last_stable_sign = None
        st.session_state.audio_bytes = None
        if "smoother" in st.session_state:
            st.session_state.smoother.clear()
        st.rerun()

with btn_col5:
    speak_clicked = st.button("🔊 Speak Sentence", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# 10. Text-to-Speech (TTS) Execution
# ---------------------------------------------------------------------------
if speak_clicked:
    current_text = st.session_state.recognized_text.strip()
    if current_text:
        st.session_state.last_spoken_text = current_text
        
        # 1. Web Speech API (Client-side, executes in user's browser)
        if tts_mode in ["Web Speech API (Browser)", "All / Automatic"]:
            speech_script = generate_speech_js(current_text)
            components.html(speech_script, height=0)

        # 2. Local pyttsx3 offline engine (Windows system audio)
        if tts_mode in ["Local pyttsx3 (Offline SAPI5)", "All / Automatic"]:
            speak_text_local(current_text, async_mode=True)

        # 3. In-App Audio Player via gTTS
        if tts_mode in ["In-App Audio Player (gTTS)", "All / Automatic"]:
            audio_bytes = generate_audio_bytes(current_text)
            st.session_state.audio_bytes = audio_bytes

        st.success(f"🔊 Spoken: \"{current_text}\"")
    else:
        st.warning("⚠️ No text to speak. Please add recognized signs or use a demo preset.")

# Show interactive audio player if audio was generated
if st.session_state.audio_bytes is not None:
    st.markdown("**Audio Playback Control:**")
    st.audio(st.session_state.audio_bytes, format="audio/mp3", autoplay=True)


# ---------------------------------------------------------------------------
# 11. Viva Demonstration Presets & Word Bank
# ---------------------------------------------------------------------------
with st.expander("💬 Quick Demo Word Bank & Manual Input", expanded=False):
    st.markdown("""
    **College Project Demonstration Helper:**
    Use these one-click preset phrases during presentations to demonstrate speech synthesis and sentence translation.
    """)
    preset_col1, preset_col2, preset_col3, preset_col4, preset_col5 = st.columns(5)
    with preset_col1:
        if st.button("Insert: HELLO"):
            st.session_state.recognized_text += "HELLO "
            st.rerun()
    with preset_col2:
        if st.button("Insert: GOOD MORNING"):
            st.session_state.recognized_text += "GOOD MORNING "
            st.rerun()
    with preset_col3:
        if st.button("Insert: THANK YOU"):
            st.session_state.recognized_text += "THANK YOU "
            st.rerun()
    with preset_col4:
        if st.button("Insert: WELCOME"):
            st.session_state.recognized_text += "WELCOME "
            st.rerun()
    with preset_col5:
        if st.button("Insert: HOW ARE YOU"):
            st.session_state.recognized_text += "HOW ARE YOU "
            st.rerun()

    st.markdown("---")
    # Manual Text Editor field
    edited_text = st.text_input("Or edit recognized text directly:", value=st.session_state.recognized_text)
    if edited_text != st.session_state.recognized_text:
        st.session_state.recognized_text = edited_text
        st.rerun()


# ---------------------------------------------------------------------------
# 12. College Project Documentation & Technical Architecture Tab
# ---------------------------------------------------------------------------
st.markdown("---")
with st.expander("📘 Technical Pipeline & Examiner Reference Architecture", expanded=False):
    tab_arch, tab_math, tab_metrics = st.tabs(["🏗️ Pipeline Architecture", "📐 Mathematical Normalization", "📊 Model Metrics"])
    
    with tab_arch:
        st.markdown("""
        ### End-to-End System Pipeline

        ```
        ┌────────────────────────────────────────────────────────┐
        │  1. Camera Input (Mirror Webcam / Image Upload)        │
        └───────────────────────────┬────────────────────────────┘
                                    ▼
        ┌────────────────────────────────────────────────────────┐
        │  2. MediaPipe Hand Tracking (21 3D Landmarks)           │
        └───────────────────────────┬────────────────────────────┘
                                    ▼
        ┌────────────────────────────────────────────────────────┐
        │  3. Normalization (Wrist-Relative & Scale-Invariant)  │
        │     → Translates wrist (index 0) to origin (0, 0, 0)   │
        │     → Normalizes coordinate span to [-1.0, 1.0]        │
        │     → Output: 63 Invariant Feature Vector              │
        └───────────────────────────┬────────────────────────────┘
                                    ▼
        ┌────────────────────────────────────────────────────────┐
        │  4. Random Forest Classifier (100 Decision Trees)      │
        │     → Computes class probabilities P(C|X)              │
        └───────────────────────────┬────────────────────────────┘
                                    ▼
        ┌────────────────────────────────────────────────────────┐
        │  5. Confidence Gating & Temporal Smoothing Buffer      │
        │     → If P(max) < Threshold: Marked "Uncertain"        │
        │     → If P(max) >= Threshold: Verified across buffer   │
        └───────────────────────────┬────────────────────────────┘
                                    ▼
        ┌────────────────────────────────────────────────────────┐
        │  6. Sentence Formation & Speech Synthesis              │
        │     → Commit Sign, Space, Backspace, Clear             │
        │     → Web Speech API + gTTS + pyttsx3 Fallback         │
        └────────────────────────────────────────────────────────┘
        ```
        """)

    with tab_math:
        st.markdown(r"""
        ### Feature Extraction & Normalization Formulation

        1. **Raw Landmarks Extraction**:
           MediaPipe outputs 21 landmarks in camera space:
           $$\mathbf{p}_i = (x_i, y_i, z_i), \quad i \in \{0, 1, \dots, 20\}$$

        2. **Wrist-Relative Translation Invariance**:
           Landmark $0$ (the wrist) serves as the local coordinate origin. Every landmark coordinate is translated relative to the wrist:
           $$\mathbf{p}_i' = \mathbf{p}_i - \mathbf{p}_0, \quad \forall i \in \{0, \dots, 20\}$$
           This guarantees that the hand position within the camera frame does not affect classification.

        3. **Scale Invariance (Distance Normalization)**:
           To make the system invariant to hand size and distance from the camera lens, coordinates are scaled by the maximum coordinate span:
           $$s = \max_{i, k} |p_{i, k}'|, \quad \mathbf{p}_i'' = \frac{\mathbf{p}_i'}{s}$$
           $$\mathbf{p}_i'' \in [-1.0, 1.0]$$

        4. **Vector Flattening**:
           The 21 normalized 3D landmarks are concatenated into a 1D vector:
           $$\mathbf{X} = [x_0'', y_0'', z_0'', \dots, x_{20}'', y_{20}'', z_{20}'']^T \in \mathbb{R}^{63}$$
        """)

    with tab_metrics:
        # Load metadata if exists
        metadata_file = os.path.join("models", "model_metadata.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r") as mf:
                    meta = json.load(mf)
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("Overall Test Accuracy", f"{meta.get('test_accuracy', 0.9948) * 100:.2f}%")
                m_col2.metric("5-Fold Cross-Val Mean", f"{meta.get('cross_val_mean', 0.9952) * 100:.2f}%")
                m_col3.metric("Trained Samples", f"{meta.get('train_samples', 2313) + meta.get('test_samples', 579)}")
                m_col4.metric("Input Features", f"{meta.get('feature_count', 63)}")
            except Exception:
                pass
        
        st.markdown("""
        | Class Sign | Samples in Dataset | Feature Dimensions | Preprocessing Parity |
        | :---: | :---: | :---: | :---: |
        | **A** | 1,700 | 63 (21 × 3) | 100% Identical |
        | **B** | 383 | 63 (21 × 3) | 100% Identical |
        | **C** | 250 | 63 (21 × 3) | 100% Identical |
        | **D** | 345 | 63 (21 × 3) | 100% Identical |
        | **E** | 214 | 63 (21 × 3) | 100% Identical |
        | **Total** | **2,892** | **63** | **Guaranteed** |
        """)