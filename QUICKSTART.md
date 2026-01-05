# Quick Start Guide - Speech Emotion Recognition

Hướng dẫn nhanh để chạy project Speech Emotion Recognition với CREMA-D dataset.

## 🚀 Khởi động nhanh (Quick Start)

### 1. Cài đặt môi trường

```bash
# Tạo môi trường conda
conda create -n voice-recognition python=3.9 -y
conda activate voice-recognition

# Cài đặt dependencies
cd voice
pip install -r requirements.txt
```

### 2. Kiểm tra cấu hình

```bash
python src/config.py
```

### 3. Chạy toàn bộ pipeline

```bash
# Chạy full pipeline (data prep + training + evaluation)
python run_pipeline.py

# Hoặc chạy nhanh với 10 epochs để test
python run_pipeline.py --quick
```

## 📂 Cấu trúc Project

```
voice/
├── CREMA-D/                   # Dataset (7,442 audio files)
│   └── AudioWAV/
├── src/                        # Source code
│   ├── config.py              # Cấu hình tất cả
│   ├── preprocess.py          # Xử lý audio
│   ├── dataset.py             # Load data, parse labels
│   ├── data_loader.py         # Data loading & augmentation
│   ├── data_preparation.py    # Data preparation pipeline
│   ├── model.py               # CNN model
│   ├── train.py               # Training
│   ├── evaluate.py            # Evaluation
│   ├── predict.py             # Inference
│   └── record.py              # Ghi âm từ mic
├── demo/
│   └── app.py                 # Streamlit web app
├── notebooks/
│   └── exploratory.ipynb      # Phân tích dữ liệu
└── results/                    # Kết quả training
```

## 🎯 Các chức năng chính

### 1. Training Model

```bash
# Cách 1: Chạy full pipeline
python run_pipeline.py

# Cách 2: Chỉ training
python src/train.py
```

### 2. Evaluation

```bash
python src/evaluate.py
```

### 3. Prediction từ file audio

```bash
python src/predict.py
```

Hoặc trong Python:

```python
from src.predict import predict_from_file

result = predict_from_file('path/to/audio.wav')
print(f"Emotion: {result['predicted_emotion']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### 4. Ghi âm và dự đoán

```bash
python src/record.py
```

Hoặc:

```python
from src.record import record_and_predict

result = record_and_predict(duration=3.0)
```

### 5. Web Demo với Streamlit

```bash
streamlit run demo/app.py
```

Sau đó mở browser tại `http://localhost:8501`

### 6. Exploratory Analysis (Jupyter)

```bash
jupyter notebook notebooks/exploratory.ipynb
```

## 📊 Kết quả mong đợi

- **Training Time**: ~30-60 phút (50 epochs, GPU)
- **Accuracy**: ~60-70% trên test set
- **Model Size**: ~50-100 MB

## 🔧 Tùy chỉnh

### Thay đổi hyperparameters

Edit `src/config.py`:

```python
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
```

### Thay đổi model architecture

Edit `src/model.py`:

```python
CNN_PARAMS = {
    'conv_blocks': [
        {'filters': 32, 'kernel_size': (3, 3)},
        {'filters': 64, 'kernel_size': (3, 3)},
        # Thêm layers...
    ]
}
```

## 🎤 6 Emotions được nhận diện

1. **ANG** - Anger (Giận dữ) 😠
2. **HAP** - Happiness (Vui vẻ) 😊
3. **SAD** - Sadness (Buồn bã) 😢
4. **NEU** - Neutral (Trung tính) 😐
5. **DIS** - Disgust (Ghê tởm) 🤢
6. **FEA** - Fear (Sợ hãi) 😨

## 📝 Files quan trọng

| File | Mô tả |
|------|-------|
| `config.py` | Tất cả cấu hình, hyperparameters |
| `train.py` | Training loop chính |
| `model.py` | CNN architecture |
| `predict.py` | Inference từ file |
| `record.py` | Ghi âm realtime |
| `app.py` | Web demo |

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

### Lỗi: GPU not found

TensorFlow sẽ tự động dùng CPU nếu không có GPU.

### Lỗi: Recording không hoạt động

```bash
pip install sounddevice soundfile
```

## 📚 Tài liệu đầy đủ

Xem `README.md` để biết chi tiết về:
- Architecture
- Dataset
- API documentation
- Advanced features

## 🎓 Citation

```
Cao, H., Cooper, D. G., Keutmann, M. K., Gur, R. C., Nenkova, A., & Verma, R. (2014).
CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset.
IEEE Transactions on Affective Computing, 5(4), 377-390.
```

## 💡 Tips

1. **Dùng GPU** để training nhanh hơn
2. **Cache data** lần đầu sẽ chậm, lần sau nhanh hơn
3. **Augmentation** giúp cải thiện accuracy
4. **Batch size** nhỏ hơn nếu bị out of memory
5. **Early stopping** tự động dừng khi không cải thiện

---

**Chúc bạn thành công! 🎉**

