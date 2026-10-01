"""
Sign Language to Text and Speech - Utilities Module
Contains shared feature extraction, landmark normalization, temporal smoothing,
model loading, and text-to-speech helpers.
"""

import os
import threading
from collections import deque, Counter
import numpy as np
import joblib


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_LANDMARKS = 21
COORDINATE_DIM = 3
FEATURE_COUNT = NUM_LANDMARKS * COORDINATE_DIM  # 63 features


# ---------------------------------------------------------------------------
# Feature Extraction and Normalization
# ---------------------------------------------------------------------------

def extract_raw_landmarks(hand_landmarks):
    """
    Extracts 21 landmark (x, y, z) coordinates in raw MediaPipe format.
    Returns:
        list of 63 floats: [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
    """
    raw_coords = []
    for lm in hand_landmarks.landmark:
        raw_coords.extend([float(lm.x), float(lm.y), float(lm.z)])
    return raw_coords


def normalize_landmark_array(raw_array):
    """
    Applies wrist-relative translation invariance and bounding scale invariance
    to landmark coordinates.

    Used identically across training (train_model.py), desktop inference (predict.py),
    and web application inference (app.py).

    Args:
        raw_array: numpy array or list of shape (N, 63) or (63,) containing raw coordinates.

    Returns:
        numpy array of the same shape with normalized coordinates in [-1, 1], dtype float32.
    """
    is_1d = False
    arr = np.asarray(raw_array, dtype=np.float32)
    if arr.ndim == 1:
        is_1d = True
        arr = arr.reshape(1, -1)

    if arr.shape[1] != FEATURE_COUNT:
        raise ValueError(
            f"Expected {FEATURE_COUNT} features per sample (21 landmarks x 3 axes), "
            f"got {arr.shape[1]}"
        )

    # Reshape to (N, 21, 3)
    pts = arr.reshape((-1, NUM_LANDMARKS, COORDINATE_DIM)).copy()

    # Step 1: Wrist-relative translation (landmark 0 is wrist)
    wrist = pts[:, 0:1, :].copy()
    pts -= wrist

    # Step 2: Scale normalization by maximum coordinate span per sample
    max_val = np.max(np.abs(pts), axis=(1, 2), keepdims=True)
    scale = np.where(max_val > 1e-6, max_val, 1.0)
    pts /= scale

    normalized = pts.reshape((-1, FEATURE_COUNT))
    return normalized[0] if is_1d else normalized


def extract_landmarks(hand_landmarks):
    """
    Extracts and normalizes 21 landmark (x, y, z) coordinates from a MediaPipe hand object.
    
    Returns:
        numpy 1D array of shape (63,) ready for model prediction.
    """
    raw_coords = extract_raw_landmarks(hand_landmarks)
    return normalize_landmark_array(raw_coords)


def predict_features(model, normalized_features):
    """
    Predicts class probabilities and top class from a normalized 63-feature vector.

    Args:
        model: Trained scikit-learn classifier with predict_proba and classes_
        normalized_features: numpy array of shape (63,) or (1, 63)

    Returns:
        tuple: (predicted_sign: str, confidence: float, prob_dict: dict)
    """
    arr = np.asarray(normalized_features, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != FEATURE_COUNT:
        raise ValueError(f"Expected {FEATURE_COUNT} features, got {arr.shape[1]}")

    probs = model.predict_proba(arr)[0]
    classes = model.classes_
    prob_dict = {str(cls): float(p) for cls, p in zip(classes, probs)}
    top_idx = int(np.argmax(probs))
    return str(classes[top_idx]), float(probs[top_idx]), prob_dict


def predict_single_hand(model, hand_landmarks):
    """
    Unified end-to-end prediction helper for a single detected hand.
    Guarantees 100% preprocessing parity with train_model.py.

    Args:
        model: Trained scikit-learn classifier
        hand_landmarks: MediaPipe NormalizedLandmarkList

    Returns:
        tuple: (predicted_sign: str, confidence: float, prob_dict: dict, features: np.ndarray)
    """
    features = extract_landmarks(hand_landmarks)
    predicted_sign, confidence, prob_dict = predict_features(model, features)
    return predicted_sign, confidence, prob_dict, features


# ---------------------------------------------------------------------------
# Model Loader
# ---------------------------------------------------------------------------

def load_trained_model(model_paths=None):
    """
    Loads the trained Random Forest classifier.
    Checks models/sign_language_model.pkl and fallback root sign_language_model.pkl.

    Returns:
        tuple: (model, resolved_path)
    """
    if model_paths is None:
        model_paths = [
            os.path.join("models", "sign_language_model.pkl"),
            "sign_language_model.pkl"
        ]

    for path in model_paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                return model, path
            except Exception as e:
                raise RuntimeError(f"Error loading model from {path}: {str(e)}")

    searched = ", ".join(model_paths)
    raise FileNotFoundError(
        f"Model file not found. Checked locations: {searched}. "
        "Please run train_model.py first to train and generate the model."
    )


# ---------------------------------------------------------------------------
# Temporal Smoothing & Stability Buffer
# ---------------------------------------------------------------------------

class TemporalSmoother:
    """
    Sliding window buffer that stabilizes real-time predictions across consecutive frames,
    filtering out single-frame misclassifications, hand jitter, and transitional gestures.
    """

    def __init__(self, window_size=7, min_consensus=4, confidence_threshold=0.70):
        self.window_size = window_size
        self.min_consensus = min_consensus
        self.confidence_threshold = confidence_threshold
        self.history = deque(maxlen=window_size)
        self.last_stable_sign = None
        self.stable_frame_count = 0

    def update_parameters(self, window_size=None, min_consensus=None, confidence_threshold=None):
        """Dynamically updates window size, consensus requirement, and threshold."""
        if window_size is not None and window_size != self.window_size:
            self.window_size = window_size
            old_history = list(self.history)
            self.history = deque(old_history, maxlen=window_size)
        if min_consensus is not None:
            self.min_consensus = min_consensus
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold

    def add(self, prediction, confidence):
        """
        Adds a single-frame prediction to the buffer.
        
        Args:
            prediction: predicted class label (str or None)
            confidence: prediction probability float (0.0 to 1.0)
            
        Returns:
            dict: {
                'stable_sign': str or None,
                'is_stable': bool,
                'confidence': float,
                'consensus_ratio': float,
                'status': str
            }
        """
        if prediction is None or confidence < self.confidence_threshold:
            self.history.append((None, confidence))
        else:
            self.history.append((prediction, confidence))

        valid_preds = [p for p, _ in self.history if p is not None]

        if not valid_preds:
            self.stable_frame_count = 0
            return {
                "stable_sign": None,
                "is_stable": False,
                "confidence": 0.0,
                "consensus_ratio": 0.0,
                "status": "No confident hand sign"
            }

        counts = Counter(valid_preds)
        most_common_sign, count = counts.most_common(1)[0]
        consensus_ratio = count / len(self.history)

        # Average confidence for the most common sign
        sign_confidences = [c for p, c in self.history if p == most_common_sign]
        avg_confidence = float(np.mean(sign_confidences)) if sign_confidences else 0.0

        is_stable = (count >= self.min_consensus) and (avg_confidence >= self.confidence_threshold)

        if is_stable:
            if most_common_sign == self.last_stable_sign:
                self.stable_frame_count += 1
            else:
                self.stable_frame_count = 1
                self.last_stable_sign = most_common_sign
            status = f"Stable sign '{most_common_sign}' ({self.stable_frame_count} frames)"
        else:
            self.stable_frame_count = 0
            status = "Stabilizing gesture..."

        return {
            "stable_sign": most_common_sign if is_stable else None,
            "is_stable": is_stable,
            "confidence": avg_confidence,
            "consensus_ratio": consensus_ratio,
            "status": status
        }

    def evaluate_single_frame(self, prediction, confidence):
        """
        Instantaneous evaluation for single snapshots (such as st.camera_input or image uploads),
        bypassing multi-frame consensus requirements while strictly enforcing confidence thresholds.
        """
        if prediction is None or confidence < self.confidence_threshold:
            return {
                "stable_sign": None,
                "is_stable": False,
                "confidence": float(confidence) if confidence is not None else 0.0,
                "consensus_ratio": 0.0,
                "status": f"Uncertain ({float(confidence)*100:.1f}%)" if prediction else "No hand detected"
            }
        return {
            "stable_sign": prediction,
            "is_stable": True,
            "confidence": float(confidence),
            "consensus_ratio": 1.0,
            "status": f"Confident sign '{prediction}' ({float(confidence)*100:.1f}%)"
        }

    def clear(self):
        """Resets the history buffer."""
        self.history.clear()
        self.last_stable_sign = None
        self.stable_frame_count = 0


# ---------------------------------------------------------------------------
# Text-To-Speech (TTS) Helpers
# ---------------------------------------------------------------------------

def _speak_worker(text):
    """Internal worker that runs pyttsx3 safely with COM initialization."""
    try:
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except Exception:
            pass
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 145)
        engine.setProperty("volume", 1.0)
        engine.say(text.strip())
        engine.runAndWait()
        try:
            engine.stop()
        except Exception:
            pass
    except Exception as e:
        print(f"[TTS Warning] Local pyttsx3 error: {e}")


def speak_text_local(text, async_mode=True):
    """
    Speaks text locally using pyttsx3 (works offline on Windows).
    Uses a background daemon thread by default to prevent UI freezing.
    
    Returns:
        tuple: (success: bool, message: str)
    """
    if not text or not text.strip():
        return False, "No text to speak"

    try:
        if async_mode:
            thread = threading.Thread(target=_speak_worker, args=(text,), daemon=True)
            thread.start()
            return True, "Spoken successfully via pyttsx3 (async)"
        else:
            _speak_worker(text)
            return True, "Spoken successfully via pyttsx3"
    except Exception as e:
        return False, f"Local TTS error: {str(e)}"


def generate_speech_js(text):
    """
    Generates a browser-side Web Speech API invocation script.
    Allows Streamlit to speak directly through the client browser's speakers
    (critical for Streamlit Cloud deployment where server has no audio output).
    """
    clean_text = text.replace('"', '\\"').replace("'", "\\'").replace("\n", " ").strip()
    return f"""
    <script>
    (function() {{
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();
            var utterance = new SpeechSynthesisUtterance("{clean_text}");
            utterance.rate = 0.95;
            utterance.pitch = 1.0;
            utterance.lang = 'en-US';
            window.speechSynthesis.speak(utterance);
        }} else {{
            console.warn('Speech synthesis not supported in this browser.');
        }}
    }})();
    </script>
    """


def generate_audio_bytes(text):
    """
    Generates in-memory MP3 audio bytes using gTTS.
    Returns:
        bytes or None
    """
    if not text or not text.strip():
        return None
    try:
        from gtts import gTTS
        import io
        tts = gTTS(text=text.strip(), lang="en")
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception:
        return None
