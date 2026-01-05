# 🎤 Speech Emotion Recognition

Dự án nhận diện cảm xúc từ giọng nói sử dụng Deep Learning với dataset CREMA-D.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🚀 QUICK START - Chạy ngay (3 bước)

```bash
# 1. Clone repository
git clone https://github.com/BinhBill210/voice-recognition-.git
cd voice-recognition-

# 2. Tạo environment và cài đặt
conda env create -f environment.yml
conda activate voice-recognition

# 3. Chạy test và training
python test_imports.py
python safe_run.py --quick  # 10 epochs, ~10-15 phút
```

**Hoặc với pip:**
```bash
pip install -r requirements.txt
python run_pipeline.py --quick
```

### ⚡ Commands cơ bản

| Task | Command |
|------|---------|
| **Test imports** | `python test_imports.py` |
| **Quick training** | `python safe_run.py --quick` |
| **Full training** | `python safe_run.py --epochs 50` |
| **Prediction** | `python src/predict.py audio.wav` |
| **Web demo** | `streamlit run demo/app.py` |
| **Recording** | `python src/record.py` |

---

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Dataset](#-dataset)
- [Kiến trúc Model](#-kiến-trúc-model)
- [Cài đặt](#-cài-đặt)
- [Sử dụng](#-sử-dụng)
- [Cấu trúc Project](#-cấu-trúc-project)
- [API Documentation](#-api-documentation)
- [Troubleshooting](#-troubleshooting)
- [Roadmap](#-roadmap)

---

## 🎯 Giới thiệu

Project này xây dựng một hệ thống **nhận diện cảm xúc từ giọng nói** sử dụng Convolutional Neural Network (CNN) với Mel Spectrogram. Hệ thống có thể:

- ✅ Nhận diện 6 cảm xúc: Anger, Happiness, Sadness, Neutral, Disgust, Fear
- ✅ Xử lý audio files (WAV format)
- ✅ Ghi âm real-time từ microphone
- ✅ Web interface với Streamlit
- ✅ Batch prediction
- ✅ Visualization và analysis tools

### 🎭 6 Cảm xúc được nhận diện

| Code | Tên tiếng Anh | Tên tiếng Việt | Emoji |
|------|---------------|----------------|-------|
| ANG  | Anger         | Giận dữ        | 😠    |
| HAP  | Happiness     | Vui vẻ         | 😊    |
| SAD  | Sadness       | Buồn bã        | 😢    |
| NEU  | Neutral       | Trung tính     | 😐    |
| DIS  | Disgust       | Ghê tởm        | 🤢    |
| FEA  | Fear          | Sợ hãi         | 😨    |

---

## 📊 Dataset

### CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)

- **📦 Files:** 7,442 audio clips
- **🎭 Emotions:** 6 classes (ANG, HAP, SAD, NEU, DIS, FEA)
- **👥 Speakers:** 91 actors (48 male, 43 female)
- **🎚️ Format:** WAV, 16kHz
- **⏱️ Duration:** ~2-3 seconds per clip

**Tên file format:** `ActorID_Sentence_Emotion_Intensity.wav`

Ví dụ: `1001_DFA_ANG_XX.wav`
- `1001`: Actor ID
- `DFA`: Sentence identifier  
- `ANG`: Emotion (Anger)
- `XX`: Intensity level

---

## 🏗️ Kiến trúc Model

### Pipeline Overview

```
Audio Input (.wav)
    ↓
Preprocessing (librosa)
    ↓
Mel Spectrogram (128 bands)
    ↓
2D CNN (4 Conv blocks)
    ↓
Dense Layers
    ↓
Softmax (6 classes)
    ↓
Emotion Prediction
```

### CNN Architecture

```python
Input: (128, 216, 1)  # Mel spectrogram

Conv Block 1: 64 filters
    ├── Conv2D(3x3) + BatchNorm + ReLU
    ├── Conv2D(3x3) + BatchNorm + ReLU
    ├── MaxPool(2x2)
    └── Dropout(0.3)

Conv Block 2: 128 filters
Conv Block 3: 256 filters
Conv Block 4: 512 filters

Flatten
    ↓
Dense(512) + BatchNorm + ReLU + Dropout(0.5)
    ↓
Dense(256) + BatchNorm + ReLU + Dropout(0.5)
    ↓
Dense(6) + Softmax

Total Parameters: ~5M
```

---

## 🔧 Cài đặt

### Requirements

- **Python:** 3.9+
- **OS:** macOS 10.15+, Ubuntu 18.04+, Windows 10+
- **RAM:** 4GB minimum (8GB recommended)
- **Disk:** 10GB free space

### Option 1: Conda (Khuyến nghị)

```bash
# Clone repository
git clone https://github.com/BinhBill210/voice-recognition-.git
cd voice-recognition-

# Tạo environment từ file
conda env create -f environment.yml
conda activate voice-recognition

# Verify
python test_imports.py
```

### Option 2: Pip

```bash
# Clone repository
git clone https://github.com/BinhBill210/voice-recognition-.git
cd voice-recognition-

# Tạo virtual environment
python3.9 -m venv venv
source venv/bin/activate  # macOS/Linux
# hoặc: venv\Scripts\activate  # Windows

# Install
pip install -r requirements.txt

# Verify
python test_imports.py
```

---

## 💻 Sử dụng

### 1. Training

```bash
# Quick test (10 epochs, ~10-15 phút)
python safe_run.py --quick

# Full training (50 epochs, ~45-60 phút)
python safe_run.py --epochs 50

# Custom
python run_pipeline.py --epochs 30 --batch-size 16
```

**Output:**
```
Train set: 5,358 samples
Validation set: 1,340 samples
Test set: 744 samples

Test Accuracy: 0.7250 (72.50%)

Per-class accuracy:
  ANG: 0.7450 (74.50%)
  HAP: 0.6890 (68.90%)
  SAD: 0.7320 (73.20%)
  NEU: 0.7150 (71.50%)
  DIS: 0.7080 (70.80%)
  FEA: 0.7610 (76.10%)
```

### 2. Prediction

```bash
# Single file
python src/predict.py data/CREMA-D/AudioWAV/1001_DFA_ANG_XX.wav

# Output:
# Emotion: ANG (Anger)
# Confidence: 0.89
# Probabilities:
#   ANG: 89.2%
#   DIS: 5.3%
#   NEU: 2.1%
#   ...
```

### 3. Recording & Real-time Prediction

```bash
python src/record.py

# Output:
# Recording... (Press Ctrl+C to stop)
# Detected emotion: HAP (Happiness)
# Confidence: 0.76
```

### 4. Web Demo

```bash
streamlit run demo/app.py

# Opens browser at http://localhost:8501
# Features:
# - Upload audio file
# - Record from microphone
# - View spectrogram
# - See predictions
```

### 5. Batch Prediction

```python
from src.predict import EmotionPredictor

predictor = EmotionPredictor('best_model.keras')

files = ['audio1.wav', 'audio2.wav', 'audio3.wav']
results = predictor.predict_batch(files)

for file, (emotion, prob) in zip(files, results):
    print(f"{file}: {emotion} ({prob:.2%})")
```

---

## 📁 Cấu trúc Project

```
voice-recognition-/
│
├── data/
│   └── CREMA-D/
│       └── AudioWAV/           # 7,442 audio files
│
├── src/                         # Source code (9 modules)
│   ├── __init__.py
│   ├── config.py               # Configuration
│   ├── preprocess.py           # Audio preprocessing
│   ├── dataset.py              # Dataset management
│   ├── model.py                # CNN architecture
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation
│   ├── predict.py              # Prediction
│   └── record.py               # Audio recording
│
├── demo/
│   └── app.py                  # Streamlit web app
│
├── notebooks/
│   └── exploratory.ipynb       # EDA notebook
│
├── models/                      # Saved models
│   ├── checkpoints/
│   └── final/
│
├── results/                     # Training results
│   ├── logs/
│   ├── metrics/
│   └── plots/
│
├── run_pipeline.py             # Main pipeline runner
├── safe_run.py                 # macOS-safe wrapper
├── test_imports.py             # Test script
│
├── environment.yml             # Conda environment
├── requirements.txt            # Pip dependencies
└── README.md                   # This file
```

---

## 📚 API Documentation

### `src/config.py`

Central configuration file.

```python
from src.config import (
    AUDIO_WAV_DIR,       # Path to audio files
    EMOTION_MAP,         # Emotion code → label
    EMOTION_NAMES,       # List of emotion names
    SAMPLE_RATE,         # 22050 Hz
    N_MELS,              # 128 mel bands
    SPECTROGRAM_SHAPE,   # (128, 216)
    NUM_CLASSES,         # 6
)
```

### `src/preprocess.py`

Audio preprocessing functions.

```python
from src.preprocess import (
    extract_emotion_from_filename,  # Parse emotion from filename
    load_audio,                      # Load WAV file
    audio_to_melspectrogram,         # Convert to mel spec
    pad_or_crop_spectrogram,         # Normalize shape
    process_audio_file,              # All-in-one
    load_dataset,                    # Load full dataset
)

# Example
spec, label = process_audio_file('audio.wav')
X, y = load_dataset('data/CREMA-D/AudioWAV')
```

### `src/model.py`

CNN model definition.

```python
from src.model import create_model

model = create_model(
    input_shape=(128, 216, 1),
    num_classes=6,
    learning_rate=0.001
)

model.summary()
```

### `src/train.py`

Training functions.

```python
from src.train import train_model

model, history = train_model(
    data_dir='data/CREMA-D/AudioWAV',
    batch_size=32,
    epochs=50,
    validation_split=0.2,
    test_split=0.1,
    learning_rate=0.001
)
```

### `src/predict.py`

Prediction interface.

```python
from src.predict import EmotionPredictor

predictor = EmotionPredictor('best_model.keras')

# Single prediction
emotion, probs = predictor.predict('audio.wav')

# Batch prediction
results = predictor.predict_batch(['audio1.wav', 'audio2.wav'])
```

### `src/record.py`

Audio recording.

```python
from src.record import record_audio, realtime_predict

# Record audio
audio = record_audio(duration=3, sample_rate=22050)

# Real-time prediction
realtime_predict(predictor, duration=3)
```

---

## ⚠️ Troubleshooting

### 🍎 macOS: Mutex Lock Warning

```
[mutex.cc : 452] RAW: Lock blocking 0x102b754b8
```

**Giải pháp:**
```bash
# Option 1: Dùng safe wrapper (KHUYẾN NGHỊ)
python safe_run.py --quick

# Option 2: Set environment variables
NUMBA_CACHE_DIR=/tmp python run_pipeline.py --quick

# Option 3: Ignore (warning không ảnh hưởng chức năng)
```

### ❌ Module Not Found

```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or with conda
conda env create -f environment.yml
```

### ❌ AudioWAV Directory Not Found

```bash
# Check path
python -c "import sys; sys.path.append('src'); from config import AUDIO_WAV_DIR; print(AUDIO_WAV_DIR)"

# Should be: /path/to/voice/data/CREMA-D/AudioWAV
```

### ❌ Out of Memory

```bash
# Reduce batch size
python run_pipeline.py --quick --batch-size 16

# Or 8
python safe_run.py --quick --batch-size 8
```

### ❌ Streamlit Demo Crashes

```bash
# Kill all Python processes
pkill -9 python

# Restart terminal
conda activate voice-recognition
streamlit run demo/app.py
```

---

## 📈 Kết quả mong đợi

### Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 70-75% |
| **Training Time** | 10-15 min (10 epochs) |
| **Inference Time** | < 1 second/file |
| **Model Size** | ~20MB |

### Per-Class Performance

| Emotion | Accuracy | Notes |
|---------|----------|-------|
| Anger (ANG) | 73-76% | Tốt nhất |
| Fear (FEA) | 74-77% | Tốt |
| Sadness (SAD) | 71-74% | Khá tốt |
| Neutral (NEU) | 70-73% | Trung bình |
| Disgust (DIS) | 69-72% | Trung bình |
| Happiness (HAP) | 67-70% | Khó nhất |

---

## 🎮 Demo

### Streamlit Web App

![Demo Screenshot](demo_screenshot.png)

**Features:**
- Upload audio file
- Record from microphone
- View waveform and spectrogram
- See prediction probabilities
- Interactive visualization

```bash
streamlit run demo/app.py
```

### CLI Demo

```bash
# Predict from file
python src/predict.py sample.wav

# Record and predict
python src/record.py
```

---

## 🗺️ Roadmap

- [x] Basic CNN model
- [x] Data preprocessing
- [x] Training pipeline
- [x] Prediction API
- [x] Web demo
- [x] Real-time recording
- [ ] Data augmentation improvements
- [ ] Transfer learning (VGGish, YAMNet)
- [ ] LSTM/Transformer models
- [ ] Multi-language support
- [ ] Mobile deployment (TFLite)
- [ ] REST API (FastAPI)
- [ ] Docker containerization

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📚 References

### Dataset
- CREMA-D: https://github.com/CheyneyComputerScience/CREMA-D
- Paper: Cao et al. (2014). CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset

### Libraries
- TensorFlow: https://www.tensorflow.org/
- Librosa: https://librosa.org/
- Streamlit: https://streamlit.io/

---

## 📧 Contact

- **GitHub:** https://github.com/BinhBill210/voice-recognition-.git
- **Issues:** https://github.com/BinhBill210/voice-recognition-/issues

---

## 🙏 Acknowledgments

- CREMA-D dataset creators
- TensorFlow team
- Librosa developers
- Open source community

---

**⭐ If you find this project useful, please give it a star!**

**Last updated:** January 5, 2026

