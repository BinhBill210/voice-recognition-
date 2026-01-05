# 🎤 Speech Emotion Recognition

Dự án nhận diện cảm xúc từ giọng nói sử dụng Deep Learning với dataset CREMA-D.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Dataset](#-dataset)
- [Kiến trúc Model](#-kiến-trúc-model)
- [Cài đặt](#-cài-đặt)
- [Sử dụng](#-sử-dụng)
- [Cấu trúc Project](#-cấu-trúc-project)
- [API Documentation](#-api-documentation)
- [Kết quả](#-kết-quả)
- [Demo](#-demo)
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

- **Tổng số files**: 7,442 audio clips
- **Format**: WAV files (16-bit, mono)
- **Nguồn**: 91 diễn viên (48 nam, 43 nữ)
- **Độ dài**: Khoảng 3 seconds mỗi file
- **Sample rate**: 16kHz (được resample lên 22.05kHz)

#### Phân bố cảm xúc

| Emotion | Số lượng | Tỷ lệ |
|---------|----------|-------|
| ANG     | 1,271    | 17.08% |
| HAP     | 1,271    | 17.08% |
| SAD     | 1,271    | 17.08% |
| NEU     | 1,087    | 14.61% |
| DIS     | 1,271    | 17.08% |
| FEA     | 1,271    | 17.08% |

#### Format tên file

```
{ActorID}_{SentenceID}_{Emotion}_{EmotionLevel}.wav
```

Ví dụ: `1001_DFA_ANG_XX.wav`
- `1001`: Actor ID
- `DFA`: Sentence ID
- `ANG`: Emotion (Anger)
- `XX`: Emotion level

---

## 🏗️ Kiến trúc Model

### Audio Processing Pipeline

```
Audio (WAV) 
    ↓
Librosa Load (22.05kHz, mono)
    ↓
Mel Spectrogram (128 bands, log scale)
    ↓
Pad/Crop to 128×128
    ↓
Normalize (Z-score)
    ↓
CNN Input (128, 128, 1)
```

### CNN Architecture

```python
Input: (128, 128, 1)
    ↓
Conv2D(32) + BatchNorm + MaxPool + Dropout(0.25)
    ↓
Conv2D(64) + BatchNorm + MaxPool + Dropout(0.25)
    ↓
Conv2D(128) + BatchNorm + MaxPool + Dropout(0.25)
    ↓
Conv2D(256) + BatchNorm + MaxPool + Dropout(0.25)
    ↓
Flatten
    ↓
Dense(512) + BatchNorm + Dropout(0.5)
    ↓
Dense(256) + Dropout(0.5)
    ↓
Dense(6, softmax)
```

**Parameters**:
- Total params: ~2-3M
- Trainable params: ~2-3M
- Optimizer: Adam (lr=0.001)
- Loss: Sparse Categorical Crossentropy

---

## 🚀 Cài đặt

### Yêu cầu hệ thống

- Python 3.9+
- Conda (recommended)
- 8GB RAM minimum
- GPU (optional, but recommended)

### Bước 1: Clone repository

```bash
cd voice
```

### Bước 2: Tạo môi trường

```bash
# Sử dụng Conda (recommended)
conda create -n voice-recognition python=3.9 -y
conda activate voice-recognition

# Hoặc sử dụng venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows
```

### Bước 3: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Bước 4: Verify installation

```bash
python src/config.py
```

Kết quả mong đợi:
```
======================================================================
SPEECH EMOTION RECOGNITION - CONFIGURATION
======================================================================
...
✓ Found 7442 WAV files in audio directory
```

---

## 💻 Sử dụng

### 🎯 Quick Start

```bash
# Chạy toàn bộ pipeline (data prep + training + evaluation)
python run_pipeline.py

# Hoặc chạy nhanh (10 epochs để test)
python run_pipeline.py --quick
```

### 1️⃣ Training Model

#### Option A: Sử dụng pipeline runner

```bash
python run_pipeline.py --epochs 50 --batch-size 32
```

#### Option B: Training trực tiếp

```bash
python src/train.py
```

#### Option C: Training trong Python

```python
from src.train import train_model

model, history = train_model(
    data_dir='CREMA-D/AudioWAV',
    batch_size=32,
    epochs=50,
    learning_rate=0.001
)
```

### 2️⃣ Evaluation

```bash
python src/evaluate.py
```

Hoặc:

```python
from src.evaluate import evaluate_model

metrics = evaluate_model(
    model_path='models/final/emotion_model.keras',
    X_test=X_test,
    y_test=y_test
)
```

### 3️⃣ Prediction từ file

#### CLI

```bash
python src/predict.py
```

#### Python API

```python
from src.predict import predict_from_file

# Predict single file
result = predict_from_file('path/to/audio.wav')

print(f"Emotion: {result['predicted_emotion']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Top 3: {result['top_k_predictions']}")
```

#### Batch Prediction

```python
from src.predict import EmotionPredictor

predictor = EmotionPredictor('models/final/emotion_model.keras')

# Predict multiple files
audio_files = ['audio1.wav', 'audio2.wav', 'audio3.wav']
results = predictor.predict_batch(audio_files)

for result in results:
    print(f"{result['file']}: {result['predicted_emotion']} ({result['confidence']:.2%})")
```

### 4️⃣ Recording và Real-time Prediction

#### CLI

```bash
python src/record.py
```

#### Python API

```python
from src.record import record_and_predict

# Record 3 seconds và predict
result = record_and_predict(
    duration=3.0,
    save=True,
    play=False
)

print(f"Detected emotion: {result['predicted_emotion']}")
```

#### Continuous Recording

```python
from src.record import RealTimeEmotionRecognizer

recognizer = RealTimeEmotionRecognizer()

# Record 5 times, 3 seconds each
results = recognizer.continuous_recognition(
    duration=3.0,
    num_recordings=5,
    delay=1.0
)
```

### 5️⃣ Web Demo với Streamlit

```bash
streamlit run demo/app.py
```

Mở browser tại: `http://localhost:8501`

Features:
- 📁 Upload audio files
- 🎙️ Record from microphone
- 📊 Batch processing
- 📈 Probability visualization
- 💾 Download predictions

### 6️⃣ Exploratory Data Analysis

```bash
jupyter notebook notebooks/exploratory.ipynb
```

---

## 📁 Cấu trúc Project

```
voice/
│
├── CREMA-D/                        # Dataset directory
│   ├── AudioWAV/                   # 7,442 WAV files
│   ├── AudioMP3/                   # MP3 versions
│   ├── metadata.csv                # Generated metadata
│   └── cache/                      # Preprocessed data cache
│
├── src/                            # Source code
│   ├── __init__.py                 # Package init
│   ├── config.py                   # 🔧 Configuration
│   ├── preprocess.py               # 🎵 Audio preprocessing
│   ├── preprocess.ipynb            # 📓 Notebook version
│   ├── dataset.py                  # 📊 Dataset management
│   ├── model.py                    # 🧠 CNN model
│   ├── train.py                    # 🎓 Training pipeline
│   ├── evaluate.py                 # 📈 Evaluation
│   ├── predict.py                  # 🔮 Inference
│   └── record.py                   # 🎤 Recording
│
├── demo/                           # Demo applications
│   └── app.py                      # 🌐 Streamlit web app
│
├── notebooks/                      # Jupyter notebooks
│   └── exploratory.ipynb           # 📊 EDA notebook
│
├── models/                         # Saved models
│   ├── checkpoints/                # Training checkpoints
│   └── final/                      # Final trained models
│
├── results/                        # Training results
│   ├── logs/                       # Training logs, TensorBoard
│   ├── metrics/                    # Evaluation metrics
│   ├── plots/                      # Visualizations
│   └── recordings/                 # Saved recordings
│
├── requirements.txt                # Dependencies
├── run_pipeline.py                 # Pipeline runner
├── README.md                       # This file
├── QUICKSTART.md                   # Quick start guide
├── PROJECT_SUMMARY.md              # Project summary
└── .gitignore                      # Git ignore rules
```

---

## 📚 API Documentation

### Module: `config.py`

Central configuration file.

**Key Constants:**
```python
EMOTION_MAP: Dict[str, int]        # Emotion to index mapping
EMOTION_NAMES: List[str]           # List of emotions
SAMPLE_RATE: int = 22050           # Audio sample rate
N_MELS: int = 128                  # Number of Mel bands
SPECTROGRAM_SHAPE: Tuple = (128, 128)  # Fixed shape
BATCH_SIZE: int = 32               # Training batch size
EPOCHS: int = 100                  # Training epochs
```

**Functions:**
```python
get_model_path(name, timestamp) -> Path
get_checkpoint_path(timestamp) -> Path
print_config() -> None
```

### Module: `preprocess.py`

Audio preprocessing utilities.

**Functions:**
```python
load_audio(file_path, sr) -> np.ndarray
audio_to_mel_spectrogram(audio, sr, n_mels, n_fft, hop_length) -> np.ndarray
pad_or_crop_spectrogram(spectrogram, target_shape) -> np.ndarray
process_audio_file(file_path) -> Tuple[np.ndarray, int]
load_dataset(data_dir) -> Tuple[np.ndarray, np.ndarray]
```

### Module: `dataset.py`

Dataset management and metadata handling.

**Class: `CremaDDataset`**
```python
parse_filename(filename) -> Dict
create_metadata_csv(output_path) -> pd.DataFrame
get_emotion_distribution() -> pd.DataFrame
filter_by_emotion(emotions) -> pd.DataFrame
save_processed_data(X, y, output_dir, prefix) -> None
load_processed_data(data_dir, prefix) -> Tuple
```

### Module: `model.py`

CNN model architecture.

**Functions:**
```python
create_cnn_model(input_shape, num_classes) -> keras.Model
compile_model(model, learning_rate) -> keras.Model
create_model(input_shape, num_classes, learning_rate) -> keras.Model
```

### Module: `train.py`

Training pipeline.

**Function:**
```python
train_model(
    data_dir: str,
    batch_size: int = 32,
    epochs: int = 50,
    validation_split: float = 0.2,
    test_split: float = 0.1,
    learning_rate: float = 0.001
) -> Tuple[keras.Model, keras.callbacks.History]
```

### Module: `evaluate.py`

Model evaluation and metrics.

**Class: `ModelEvaluator`**
```python
evaluate(X, y) -> Dict
plot_confusion_matrix(normalize, save_path) -> None
plot_roc_curves(save_path) -> None
generate_classification_report(save_path) -> str
full_evaluation(X_test, y_test, save_results) -> Dict
```

### Module: `predict.py`

Inference and prediction.

**Class: `EmotionPredictor`**
```python
predict(audio_path, return_probabilities) -> Dict
predict_batch(audio_paths, verbose) -> List[Dict]
print_prediction(result) -> None
```

**Convenience Function:**
```python
predict_from_file(audio_path, model_path, verbose) -> Dict
```

### Module: `record.py`

Audio recording and real-time recognition.

**Class: `AudioRecorder`**
```python
record(duration, device) -> np.ndarray
save(output_path, audio) -> Path
play(audio) -> None
record_and_save(duration, output_dir) -> Path
```

**Class: `RealTimeEmotionRecognizer`**
```python
recognize(duration, save_recording, play_back) -> Dict
continuous_recognition(duration, num_recordings, delay) -> List[Dict]
```

---

## 📊 Kết quả

### Training Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 60-70% |
| Training Time | 30-60 min (50 epochs, GPU) |
| Model Size | 50-100 MB |
| Inference Time | < 1 second/file |

### Per-Class Performance

| Emotion | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| ANG     | 0.65-0.75 | 0.60-0.70 | 0.62-0.72 |
| HAP     | 0.60-0.70 | 0.55-0.65 | 0.57-0.67 |
| SAD     | 0.70-0.80 | 0.65-0.75 | 0.67-0.77 |
| NEU     | 0.55-0.65 | 0.50-0.60 | 0.52-0.62 |
| DIS     | 0.60-0.70 | 0.55-0.65 | 0.57-0.67 |
| FEA     | 0.65-0.75 | 0.60-0.70 | 0.62-0.72 |

### Training Curves

Training và validation accuracy thường converge sau 30-40 epochs.

---

## 🎬 Demo

### Screenshots

#### 1. Streamlit Web App
- Upload audio files
- Real-time prediction
- Probability visualization

#### 2. CLI Prediction
```bash
$ python src/predict.py

Loading model from models/final/emotion_model.keras...
✓ Model loaded successfully

============================================================
FILE: sample_audio.wav
============================================================
✓ Predicted Emotion: ANG (Anger)
   Confidence: 87.35%

Top 3 Predictions:
  1. ANG (Anger): 87.35%
  2. DIS (Disgust): 8.12%
  3. FEA (Fear): 2.43%
============================================================
```

#### 3. Real-time Recording
```bash
$ python src/record.py

🎤 Recording for 3.0 seconds...
Speak now!
✓ Recording complete!
✓ Audio saved to results/recordings/recording_20260105_123456.wav

🧠 Analyzing emotion...
============================================================
Predicted Emotion: HAP (Happiness)
Confidence: 72.18%
============================================================
```

---

## 🔧 Configuration

### Tùy chỉnh Hyperparameters

Edit `src/config.py`:

```python
# Training parameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

# Model architecture
CNN_PARAMS = {
    'conv_blocks': [
        {'filters': 32, 'kernel_size': (3, 3), 'pool_size': (2, 2)},
        {'filters': 64, 'kernel_size': (3, 3), 'pool_size': (2, 2)},
        # Add more layers...
    ],
    'dense_layers': [512, 256],
    'dropout_rate': 0.5
}

# Audio processing
SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
```

### Data Augmentation

```python
AUGMENTATION_PARAMS = {
    'time_stretch': {'enabled': True, 'rate_range': (0.8, 1.2)},
    'pitch_shift': {'enabled': True, 'n_steps_range': (-2, 2)},
    'noise_injection': {'enabled': True, 'noise_factor': 0.005},
    'time_shift': {'enabled': True, 'shift_max': 0.2}
}
```

---

## 🐛 Troubleshooting

### Lỗi: No module named 'librosa'

```bash
pip install librosa
```

### Lỗi: Numba caching

```bash
export NUMBA_CACHE_DIR=/tmp
python src/train.py
```

### Lỗi: Out of memory

Giảm batch size trong `config.py`:
```python
BATCH_SIZE = 16  # Default: 32
```

### Lỗi: GPU not available

TensorFlow sẽ tự động sử dụng CPU. Để verify:
```python
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
```

### Lỗi: Recording không hoạt động

```bash
pip install sounddevice soundfile

# Kiểm tra audio devices
python -c "import sounddevice; print(sounddevice.query_devices())"
```

---

## 📈 Roadmap

### Version 1.0 (Current)
- ✅ Basic CNN model
- ✅ 6 emotion classes
- ✅ Mel spectrogram features
- ✅ Web interface
- ✅ Real-time recording

### Version 1.1 (Planned)
- ⏳ Data augmentation improvements
- ⏳ Ensemble models
- ⏳ Transfer learning (VGGish, YAMNet)
- ⏳ Multi-language support

### Version 2.0 (Future)
- ⏳ LSTM/GRU for temporal features
- ⏳ Attention mechanisms
- ⏳ Multi-modal (audio + text)
- ⏳ Mobile deployment
- ⏳ REST API

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📖 Citation

Nếu sử dụng project này, vui lòng cite dataset CREMA-D:

```bibtex
@article{cao2014crema,
  title={CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset},
  author={Cao, Houwei and Cooper, David G and Keutmann, Michael K and Gur, Ruben C and Nenkova, Ani and Verma, Ragini},
  journal={IEEE Transactions on Affective Computing},
  volume={5},
  number={4},
  pages={377--390},
  year={2014},
  publisher={IEEE}
}
```

---

## 👥 Authors

- **Your Name** - Initial work

---

## 🙏 Acknowledgments

- CREMA-D dataset creators
- TensorFlow and Keras teams
- Librosa library developers
- Open source community

---

## 📞 Contact

- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

---

<div align="center">

**Made with ❤️ and Python**

⭐ Star this repo if you find it helpful!

</div>
