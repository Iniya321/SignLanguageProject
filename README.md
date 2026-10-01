# 🤟 Real-Time Sign Language to Text and Speech System

> **A Computer Vision & Machine Learning Prototype for Assistive Communication**  
> *Developed as a 3rd-Year Computer and Communication Engineering (CCE) Capstone Project*

---

## 📌 Problem Statement

Communication barriers between deaf or hard-of-hearing individuals who use sign language and people unfamiliar with sign language pose significant challenges in education, healthcare, and daily social interactions. Many existing solutions either require expensive wearable sensor gloves or rely on rigid cloud-only APIs with high latency. There is a strong need for an accessible, vision-based, real-time sign language interpreter that runs on standard consumer webcams, translates hand gestures into readable text, and articulates words aloud via speech synthesis.

---

## 🎯 Objective

To develop a robust, end-to-end prototype capable of:
1. Detecting hand gestures in real-time using computer vision (**MediaPipe Hands**).
2. Extracting and normalizing 21 landmark 3D coordinates (63 features) with **wrist-relative translation** and **scale invariance**.
3. Classifying hand gestures using a trained machine learning model (**Random Forest Classifier**) with probability confidence estimation.
4. Stabilizing predictions through a **temporal smoothing buffer** to eliminate jitter and transitional gesture noise.
5. Allowing interactive **word and sentence formation** (Add Sign, Space, Backspace, Clear).
6. Converting recognized text into natural voice output using **Text-to-Speech (TTS)**.
7. Providing a browser-compatible, cloud-ready **Streamlit web interface** as well as a local desktop OpenCV interface.

---

## 🚀 Key Features

- **21 3D Hand Landmark Tracking**: Tracks wrist, thumb, index, middle, ring, and pinky joints with sub-millimeter precision.
- **Position & Distance Invariance**: Normalizes landmarks relative to the wrist and scales by hand bounding span, ensuring accurate recognition regardless of where the hand is positioned on screen or its distance from the camera.
- **Confidence Threshold Filtering**: Any prediction below a configurable threshold (default 70%) is flagged as *"Uncertain Sign"*, preventing false positives during hand movement.
- **Temporal Stability Check**: Uses a sliding window majority-voting buffer to verify that a gesture is held steadily across frames before committing.
- **Interactive Sentence Builder**: Supports incremental letter composition, space separation between words, character deletion, and text clearing.
- **Dual-Mode Text-to-Speech (TTS)**:
  - **Client-Side Browser Speech**: Utilizes the HTML5 Web Speech API (`window.speechSynthesis`) so speech synthesis works seamlessly when deployed on Streamlit Cloud.
  - **Local Windows Engine**: Integrates `pyttsx3` for offline local voice synthesis on Windows.
- **Self-Contained Data Collection & Training**: Built-in CLI tools to record additional hand sign samples and retrain the classifier in seconds.

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Programming Language** | Python 3.12 | Core logic and backend pipeline |
| **Computer Vision** | OpenCV (`cv2`) & MediaPipe Hands | Video frame capture, landmark detection, skeleton drawing |
| **Data Processing** | NumPy & Pandas | 3D coordinate manipulation, matrix math, CSV dataset management |
| **Machine Learning** | Scikit-learn (`RandomForestClassifier`) | 100-tree ensemble classification with `predict_proba()` |
| **Model Persistence** | Joblib | Serialization and loading of the trained model artifact |
| **Web Interface** | Streamlit | Browser-accessible GUI with webcam input and interactive widgets |
| **Text-to-Speech** | Web Speech API / `pyttsx3` / `gTTS` | Multimodal voice synthesis for recognized text |

---

## 🏗️ System Architecture

```
                  ┌───────────────────────────────┐
                  │       User Hand Gesture       │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │ Camera Input (Webcam / Image) │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │   MediaPipe Hand Detection    │
                  │   (21 Landmarks: x, y, z)     │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │     Feature Normalization     │
                  │ - Wrist Translation Offset    │
                  │ - Coordinate Scale Invariance │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │   Random Forest Classifier    │
                  │   (100 Decision Trees)        │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │   Confidence Thresholding     │
                  │   (e.g., Confidence >= 70%)   │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │   Temporal Smoothing Buffer   │
                  │   (Sliding Window Majority)   │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │   Interactive Text Formation  │
                  │   (Letters → Words → Sentence)│
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │      Text-to-Speech (TTS)     │
                  │   (Audible Spoken Output)     │
                  └───────────────────────────────┘
```

---

## 🔍 How Feature Normalization Works

### The Core Cause of Earlier Prediction Errors:
In naive implementations, raw MediaPipe landmark coordinates are passed directly to the model. Because $x$ and $y$ represent absolute screen coordinates ($0.0$ to $1.0$):
- If sign 'A' was recorded with the hand centered ($x \approx 0.5$), moving the hand to the right ($x \approx 0.8$) shifts all 63 coordinates.
- Decision trees split on absolute thresholds (e.g. `feature_0 > 0.65`), causing the model to mistake hand position for gesture class!

### The Mathematical Fix Implemented in `utils.py`:
1. **Wrist-Relative Translation**:
   Landmark $0$ (the wrist) is taken as origin $(x_0, y_0, z_0)$. For every landmark $i \in \{0, \dots, 20\}$:
   $$x'_i = x_i - x_0, \quad y'_i = y_i - y_0, \quad z'_i = z_i - z_0$$
   Now $x'_0 = 0, y'_0 = 0, z'_0 = 0$. Position on the screen has zero impact on features.

2. **Scale Invariance**:
   Find the maximum absolute coordinate span:
   $$d = \max_{i} \left( |x'_i|, |y'_i|, |z'_i| \right)$$
   Normalize all coordinates:
   $$x''_i = \frac{x'_i}{d}, \quad y''_i = \frac{y'_i}{d}, \quad z''_i = \frac{z'_i}{d}$$
   All 63 features are bounded in $[-1.0, 1.0]$, ensuring identical representations regardless of hand distance from camera.

---

## 📊 Dataset & Model Evaluation

### Dataset Distribution (`dataset/*.csv`):
- **Sign A**: 1,346 samples
- **Sign B**: 306 samples
- **Sign C**: 201 samples
- **Sign D**: 275 samples
- **Sign E**: 764 samples
- **Total Samples**: 2,892 samples (63 landmark features + 1 label column per row)

### Evaluation Metrics (Evaluated on Independent 20% Stratified Test Split):
- **Overall Test Accuracy**: **99.48%**
- **5-Fold Stratified Cross-Validation**: **99.52% (± 0.17%)**

#### Classification Report:
| Sign | Precision | Recall | F1-Score | Support |
|:---:|:---:|:---:|:---:|:---:|
| **A** | 0.9890 | 1.0000 | 0.9945 | 270 |
| **B** | 1.0000 | 1.0000 | 1.0000 | 61 |
| **C** | 1.0000 | 1.0000 | 1.0000 | 40 |
| **D** | 1.0000 | 0.9455 | 0.9720 | 55 |
| **E** | 1.0000 | 1.0000 | 1.0000 | 153 |
| **Average** | **0.9978** | **0.9891** | **0.9933** | **579** |

#### Confusion Matrix:
```
           Pred_A   Pred_B   Pred_C   Pred_D   Pred_E
True_A       270        0        0        0        0
True_B         0       61        0        0        0
True_C         0        0       40        0        0
True_D         3        0        0       52        0
True_E         0        0        0        0      153
```

---

## 📁 Project Directory Structure

```
SignLanguageProject/
│
├── app.py                      # Main Streamlit web application
├── predict.py                  # Real-time desktop OpenCV application with HUD & smoothing
├── collect_data.py             # Data collection script for recording new sign samples
├── train_model.py              # Model training, evaluation, and serialization script
├── utils.py                    # Feature extraction, normalization, smoothing, & TTS helpers
├── main.py                     # System diagnostic launcher & camera validation
├── requirements.txt            # Dependency specification
├── README.md                   # Complete project documentation
│
├── dataset/                    # Landmark coordinate CSV files
│   ├── A.csv
│   ├── B.csv
│   ├── C.csv
│   ├── D.csv
│   └── E.csv
│
└── models/                     # Trained models and evaluation metadata
    ├── sign_language_model.pkl
    └── model_metadata.json
```

---

## ⚙️ Installation

### 1. Clone or Open Workspace
Ensure you are in the project root directory:
```bash
cd "SignLanguageProject"
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🖥️ How to Run

### Option 1: Streamlit Web Application (Primary Prototype)
Runs the browser-accessible interface with camera input, confidence bar, text builder, and speech:
```bash
python -m streamlit run app.py
```
*Access in browser at:* `http://localhost:8501`

### Option 2: High-Performance Real-Time Desktop App
Runs the real-time webcam feed with live heads-up display (HUD), continuous smoothing, keyboard shortcuts, and speech:
```bash
python predict.py
```
*Controls in Desktop App:*
- `[A]` : Commit current stable sign into recognized text
- `[SPACE]` : Insert space between words
- `[D]` / `[Backspace]` : Delete last letter
- `[C]` : Clear all text
- `[S]` : 🔊 Speak recognized sentence aloud
- `[Q]` : Quit

### Option 3: Collect Additional Sign Data
To add new letters (e.g., F, G, H, I, etc.) or augment existing signs:
```bash
python collect_data.py
```
- Enter the sign label (e.g., `F`).
- Press `[S]` or `[SPACE]` to start/pause recording frames.
- Press `[Q]` to save and exit.

### Option 4: Retrain & Evaluate Model
After collecting new data, run:
```bash
python train_model.py
```
Generates updated `models/sign_language_model.pkl`, tests 5-fold cross validation, and outputs evaluation metrics.

### Option 5: Verify Preprocessing Parity Test Suite
Run the automated mathematical parity and invariance validation suite:
```bash
python test_preprocessing_parity.py
```

---

## 🎯 Expected Output

1. **Camera Detection**: The camera activates and MediaPipe draws green landmark nodes connected by white bone lines over hand joints.
2. **Prediction**: The app outputs the predicted sign (e.g., `Predicted Sign: A`) along with its confidence percentage (e.g., `Confidence: 96.4%`).
3. **Low-Confidence Handling**: If confidence is under 70%, the badge displays `Uncertain Sign` and shows candidate distributions.
4. **Text Assembly**: Clicking `➕ Add Sign` or pressing `[A]` builds words incrementally (e.g., `B` $\rightarrow$ `BE` $\rightarrow$ `BED`).
5. **Speech Synthesis**: Clicking `🔊 Speak` audibly speaks the assembled text through the speakers.

---

## ⚠️ Known Limitations

As an honest academic project prototype, the system currently has specific operational boundaries:
1. **Static Alphabet Recognition**: The current pipeline classifies static hand poses (individual alphabet letters A-E). It does not natively recognize dynamic gestures involving motion trajectories (such as "J" or "Z").
2. **Independent Sign Sequencing**: Natural sign languages have distinct grammatical structures. Composing sentences letter-by-letter represents finger-spelling rather than full fluent sign language grammar.
3. **Single-Hand Focus**: The primary classifier processes one dominant hand at a time. Two-handed signs (such as in BSL or complex ASL words) are not yet classified.
4. **Lighting & Occlusion Sensitivity**: Extreme backlighting or heavy finger self-occlusion can degrade MediaPipe's landmark tracking precision.

---

## 🔮 Future Enhancements

- [ ] **Temporal Sequence Modeling**: Incorporate LSTM, GRU, or Temporal Convolutional Networks (TCN) to classify dynamic signs and continuous motion.
- [ ] **Extended Alphabet & Digits**: Expand the dataset to all 26 English alphabet letters (A-Z) and numbers (0-9).
- [ ] **Full ASL Vocabulary & NLP**: Integrate transformer-based sequence-to-sequence translation models for continuous phrase understanding.
- [ ] **Two-Hand & Facial Landmark Integration**: Utilize MediaPipe Holistic to track both hands and facial expressions (grammatical non-manual markers).
- [ ] **Cross-Platform Mobile App**: Port the model using ONNX Runtime / TensorFlow Lite for Android and iOS mobile deployment.
