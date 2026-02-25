# 🤟 Silent Link

**Real-time, offline sign language recognition and communication app powered by Deep Learning — built for Android.**

Silent Link is an assistive communication application designed to bridge the communication gap between deaf or speech-impaired individuals and non-sign-language users. It provides a unified mobile platform for real-time gesture-to-text and text/speech-to-gesture conversion — entirely offline, with no internet dependency.

---

## 📌 Overview

The application uses a machine learning–based gesture recognition system to interpret hand gestures captured via the device camera and convert them into readable text. Conversely, text input can be converted into corresponding sign-language gesture animations (GIFs), allowing deaf users to understand responses visually.

Silent Link is designed to work **entirely offline**, ensuring privacy, low latency, and usability in environments with no network connectivity. All processing — including gesture recognition — is performed locally on the device using optimized DL models (TensorFlow Lite). The app runs within a single device and does not rely on Bluetooth, peer-to-peer networking, or cloud services.

---

## 🎯 Key Objectives

- **Real-time two-way communication** for deaf users without internet dependency
- **Gesture → Text** using a deep learning LSTM model with MediaPipe landmark tracking
- **Speech/Text → ISL Animation** via 100+ phrase GIFs and A–Z fingerspelling
- **Fully on-device processing** ensuring complete data privacy
- **Single unified mobile app** with all features integrated into one interface
- **Cross-platform** — Android (primary), with future iOS support

---

## 🧠 How It Works

### Feature 1: Gesture → Text (Deep Learning Pipeline)

```
Camera Feed → MediaPipe Holistic → Keypoint Extraction → LSTM Model → Predicted Action (Text)
```

1. **Capture** — The device camera captures a live video feed of the user performing sign language gestures.
2. **Pose Estimation** — MediaPipe Holistic extracts body pose, face, and hand landmarks from each frame.
3. **Keypoint Extraction** — Landmark coordinates are flattened into a 1662-dimensional feature vector per frame (33×4 pose + 468×3 face + 21×3 left hand + 21×3 right hand).
4. **Sequence Collection** — 30 consecutive frames of keypoints form one input sequence.
5. **LSTM Prediction** — The on-device TFLite model classifies the gesture sequence into an action label.
6. **Output** — The predicted action is displayed as on-screen text in real time.

### Feature 2: Speech/Text → ISL Animation

```
Speech/Text Input → Phrase Matching → ISL GIF Animation / Letter Fingerspelling
```

1. **Input** — User speaks into the microphone or types a phrase.
2. **Speech-to-Text** — Audio is converted to text via on-device speech recognition.
3. **Phrase Matching** — Text is matched against a vocabulary of 100+ ISL phrases.
4. **GIF Display** — If a matching phrase is found, the corresponding ISL GIF animation is displayed.
5. **Fingerspelling Fallback** — For unrecognized words, individual letters are displayed as sign language alphabet images (A–Z).

---

## 🏗️ Model Architecture

The gesture recognition model is a **3-layer LSTM** deep neural network:

```
Input (30 frames × 1662 features)
    ↓
LSTM (64 units, return_sequences=True, ReLU)
    ↓
LSTM (128 units, return_sequences=True, ReLU)
    ↓
LSTM (64 units, return_sequences=False, ReLU)
    ↓
Dense (64 units, ReLU)
    ↓
Dense (32 units, ReLU)
    ↓
Dense (N actions, Softmax)
```

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Epochs | 2000 |
| Sequence Length | 30 frames |
| Feature Vector | 1662 dimensions |
| Confidence Threshold | 0.8 |
| Model Size | ~2.4 MB (TFLite) |

---

## 📁 Project Structure

```
silent_link/
│
├── app.py                     # Application entry point / web prototype
├── main_train.py              # Data collection + DL model training
├── main_predictor.py          # Gesture recognition inference module
├── converter_tens_flite.py    # Model converter (.h5 → .tflite for mobile)
├── action.h5                  # Trained DL model weights (Keras)
├── action.tflite              # TFLite model (optimized for Android)
├── signlang.png               # Application banner
├── .env                       # Environment configuration
│
├── templates/                 # UI templates
│   └── index.html             # Application interface
│
├── ISL_Gifs/                  # ISL phrase animations (100+ GIFs)
│   ├── hello.gif
│   ├── good morning.gif
│   ├── what is your name.gif
│   └── ...
│
├── letters/                   # Sign language alphabet (A–Z)
│   ├── a.jpg ... z.jpg
│
├── MP_Data/                   # Training keypoint data (.npy)
├── Logs/                      # TensorBoard training logs
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.8+ (for model training)
- Node.js & npm (for React Native mobile build)
- Android Studio (for Android deployment)
- Webcam and Microphone (for data collection)

### Model Training (Python Backend)

```bash
pip install tensorflow opencv-python mediapipe scikit-learn numpy matplotlib SpeechRecognition pyaudio pillow python-dotenv
```

```bash
# Train the gesture recognition model
python main_train.py

# Convert to TFLite for mobile deployment
python converter_tens_flite.py
```

### Mobile App (React Native)

```bash
npm install
npx react-native run-android
```

---

## 🚀 Application Features

| Feature | Description |
|---|---|
| **Live Gesture Recognition** | Point your camera and perform sign language gestures — the app predicts and displays the action as text in real time |
| **Speech → ISL Translation** | Tap the mic, speak a phrase, and see the corresponding ISL GIF animation |
| **Text → ISL Translation** | Type a word or phrase to see the sign language animation |
| **Fingerspelling** | Unrecognized words are broken down into individual letter signs (A–Z) |
| **Confidence Display** | Real-time probability visualization for predicted gestures |
| **Offline Mode** | Works entirely without internet — all inference runs on-device via TFLite |

---

## 📊 Training Visualization

```bash
tensorboard --logdir=Logs
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **React Native** | Cross-platform mobile application framework |
| **TensorFlow Lite** | On-device deep learning inference (Android) |
| **TensorFlow / Keras** | LSTM model training (Python backend) |
| **MediaPipe Holistic** | Pose, face, and hand landmark detection |
| **OpenCV** | Camera capture and video processing |
| **SpeechRecognition** | Speech-to-text conversion |
| **Python** | Model training and data processing |
| **NumPy / scikit-learn** | Data processing and evaluation |

---

## 🌟 Features

- ✅ **Android mobile app** — Native mobile experience via React Native
- ✅ **Unified single application** — All features in one integrated interface
- ✅ **Deep Learning powered** — LSTM neural network for gesture classification
- ✅ **Two-way communication** — Gesture → Text and Speech → Sign Language
- ✅ **100+ ISL phrase GIFs** — Comprehensive sign language animation library
- ✅ **A–Z fingerspelling** — Letter-by-letter fallback for full coverage
- ✅ **Offline-first** — No internet required, all processing on-device via TFLite
- ✅ **Real-time inference** — Live gesture detection at camera framerate
- ✅ **Multi-landmark tracking** — Pose + face + hand keypoints (1662 features)
- ✅ **Privacy-preserving** — No data leaves the device
- ✅ **Lightweight** — ~2.4 MB TFLite model optimized for mobile

---

## 🔮 Future Enhancements

- [ ] iOS support via React Native
- [ ] Expanded gesture vocabulary (50+ custom signs)
- [ ] Multi-language sign language support (ASL, BSL, ISL)
- [ ] Text-to-speech voice output
- [ ] Continuous gesture sentence formation
- [ ] Real-time two-way video conversation mode

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

> *Silent Link demonstrates how deep learning and mobile computing can be combined to create inclusive, accessible, and reliable communication tools for the hearing-impaired community.*
