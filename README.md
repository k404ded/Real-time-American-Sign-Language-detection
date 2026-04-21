# ✋ ASL Detection + Text-to-Speech

A real-time **American Sign Language (ASL) detection system** that translates hand gestures into text and speech using computer vision.

Built using **MediaPipe** and **OpenCV**, this project detects hand landmarks, classifies gestures using rule-based logic, and converts them into **words and spoken sentences** with text-to-speech.

---

## 🚀 Features

* 🎥 Real-time hand tracking using webcam
* ✋ ASL alphabet recognition (static signs)
* 🧠 Rule-based gesture classification
* 📊 Stability-based prediction (reduces noise & errors)
* 🔊 Offline Text-to-Speech (TTS) output
* ⏱️ Hold-based input system for accuracy
* ⌨️ Keyboard controls for interaction

---

## 🛠️ Tech Stack

| Technology | Usage                                   |
| ---------- | --------------------------------------- |
| OpenCV     | Webcam capture & UI                     |
| MediaPipe  | Hand landmark detection (21 key points) |
| NumPy      | Mathematical computations               |
| pyttsx3    | Offline text-to-speech                  |

---

## 📂 Project Structure

```
ASL-Detector/
│── asl_detector.py
│── README.md
│── requirements.txt
```

---

## ⚙️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ASL-Detector.git

# Navigate into the folder
cd ASL-Detector

# Create virtual environment (recommended)
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the project
python asl_detector.py
```

---

## 🎮 How to Use

1. Open the webcam.
2. Show an ASL sign clearly in front of the camera.
3. Hold the gesture steady (~1.2 seconds).
4. The system detects and adds the letter to a word.
5. Use open hand (✋) or space key to add space.

---

## ⌨️ Controls

| Key     | Function           |
| ------- | ------------------ |
| `S`     | Speak the sentence |
| `C`     | Clear text         |
| `Space` | Add space          |
| `Q`     | Quit               |

---

## ✋ Supported Signs

* Alphabets:
  `A B D E F G H I K L M N O P R S T U V W X Y`
* `SPACE` (open hand)

> ⚠️ Note: Dynamic signs like **J** and **Z** are not supported.

---

## 🧠 How It Works

```
Webcam Input
      ↓
MediaPipe Hand Detection
      ↓
21 Landmark Extraction
      ↓
Rule-Based Classification
      ↓
Temporal Smoothing (Buffer)
      ↓
Stable Letter Output
      ↓
Text Formation + TTS
```

---

## 📊 Improvements & Future Work

* 🤖 Train ML model for higher accuracy
* 🔄 Add dynamic gesture detection (J, Z)
* 📱 Deploy as mobile/web app
* 💬 Add word prediction/autocomplete
* 🌐 Multi-language support

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo and improve the model.

---


## 📜 License

This project is open-source and available under the MIT License.

---

## 👩‍💻 Author

**Kimaya Bhave**

---

## 🌟 Show Your Support

If you like this project, give it a ⭐ on GitHub!
