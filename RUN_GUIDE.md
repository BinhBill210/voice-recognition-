# 🚀 Hướng dẫn Chạy Project

## ✅ Cleanup đã hoàn thành

Đã xóa 2 files trùng lặp:
- ❌ `src/data_loader.py` (đã có trong `preprocess.py`)
- ❌ `src/data_preparation.py` (đã có trong `train.py`)

Đã đồng bộ hóa tất cả imports và paths!

---

## 📋 Bước 1: Test Imports (QUAN TRỌNG!)

Chạy test script để đảm bảo mọi thứ hoạt động:

```bash
cd /Users/macbook/Library/CloudStorage/OneDrive-SwinburneUniversity/Documents/Project/voice
conda activate voice-recognition
python test_imports.py
```

**Kết quả mong đợi:**
```
✅ ALL TESTS PASSED!

You can now run the pipeline:
  python run_pipeline.py --quick
```

---

## 📋 Bước 2: Chạy Pipeline

### Option 1: Quick Test (10 epochs - ~10-15 phút) 🎯 KHUYẾN NGHỊ

```bash
python run_pipeline.py --quick
```

Sẽ thực hiện:
1. Load 7,442 audio files từ `data/CREMA-D/AudioWAV/`
2. Preprocess → Mel spectrograms (128 bands)
3. Train CNN với 10 epochs
4. In ra test accuracy

### Option 2: Full Training (50 epochs - ~45-60 phút)

```bash
python run_pipeline.py --epochs 50
```

### Option 3: Custom

```bash
python run_pipeline.py --epochs 20 --batch-size 16
```

---

## 📊 Kết quả mong đợi

```
======================================================================
SPEECH EMOTION RECOGNITION - FULL PIPELINE
======================================================================

[1/1] MODEL TRAINING & EVALUATION
----------------------------------------------------------------------
Audio directory: /Users/.../data/CREMA-D/AudioWAV
Found WAV files: 7442

============================================================
Speech Emotion Recognition - CREMA-D Dataset
============================================================

[1/4] Loading and preprocessing audio files...
Processing audio files: 100%|████████████| 7442/7442 [XX:XX<00:00]

Dataset shape: (7442, 128, 216, 1)
Labels shape: (7442,)
Number of classes: 6

Class distribution:
  ANG: 1271
  DIS: 1271
  FEA: 1271
  HAP: 1636
  NEU: 1087
  SAD: 1271

[2/4] Splitting dataset...
Train set: XXXX samples
Validation set: XXXX samples
Test set: XXXX samples

[3/4] Creating model...
[4/4] Training model...

Epoch 1/10
...

Test Accuracy: X.XXXX (XX.XX%)
Test Loss: X.XXXX

Per-class accuracy on test set:
  ANG: X.XXXX (XX.XX%)
  DIS: X.XXXX (XX.XX%)
  FEA: X.XXXX (XX.XX%)
  HAP: X.XXXX (XX.XX%)
  NEU: X.XXXX (XX.XX%)
  SAD: X.XXXX (XX.XX%)

======================================================================
PIPELINE COMPLETED SUCCESSFULLY!
======================================================================

Model saved to: best_model.keras
```

---

## 🔍 Troubleshooting

### Lỗi: "AudioWAV directory not found"

```bash
# Check đường dẫn
python -c "import sys; sys.path.append('src'); from config import AUDIO_WAV_DIR; print(AUDIO_WAV_DIR); print(AUDIO_WAV_DIR.exists())"

# Nên thấy:
# /Users/.../voice/data/CREMA-D/AudioWAV
# True
```

### Lỗi: "No module named 'librosa'"

```bash
pip install -r requirements.txt
```

### Lỗi: Import errors

```bash
# Run test script
python test_imports.py

# Sẽ cho biết module nào bị lỗi
```

### Lỗi: TensorFlow mutex lock

Đây là lỗi thông thường của TensorFlow trên macOS, không ảnh hưởng chức năng.
Chỉ cần chạy lại hoặc dùng `NUMBA_CACHE_DIR=/tmp`:

```bash
NUMBA_CACHE_DIR=/tmp python run_pipeline.py --quick
```

---

## 📁 Files còn lại sau Cleanup

```
voice/
├── src/                           ✅ 9 Python files
│   ├── config.py                  → Central configuration
│   ├── preprocess.py              → Audio processing  
│   ├── dataset.py                 → Dataset management
│   ├── model.py                   → CNN architecture
│   ├── train.py                   → Training logic
│   ├── evaluate.py                → Evaluation
│   ├── predict.py                 → Inference
│   └── record.py                  → Audio recording
│
├── run_pipeline.py                ✅ Main runner
├── test_imports.py                ✅ Test script
├── requirements.txt               ✅ Dependencies
│
├── data/
│   └── CREMA-D/
│       └── AudioWAV/              ✅ 7,442 .wav files
│
├── README.md                      📚 Full documentation
├── QUICKSTART.md                  📚 Quick start (Vietnamese)
├── CLEANUP_SUMMARY.md             📚 Cleanup details
└── RUN_GUIDE.md                   📚 This file
```

---

## 🎯 Các lệnh hữu ích

```bash
# Test config
python src/config.py

# Test preprocessing 1 file
python src/preprocess.py

# Train trực tiếp (không qua pipeline)
python src/train.py

# Test prediction
python src/predict.py path/to/audio.wav

# Record và predict real-time
python src/record.py

# Web demo
streamlit run demo/app.py

# Explore dataset
jupyter notebook notebooks/exploratory.ipynb
```

---

## 💡 Tips

1. **Lần đầu chạy**: Dùng `--quick` để test (10-15 phút)
2. **Preprocessing lâu**: Preprocessing 7,442 files mất ~5-10 phút (chỉ lần đầu)
3. **Save model**: Model tự động save vào `best_model.keras`
4. **Monitor training**: Có thể dùng TensorBoard nếu enable trong callbacks
5. **Memory**: Cần ~4-8GB RAM để load toàn bộ dataset

---

## ✅ Checklist trước khi chạy

- [ ] Conda environment activated: `voice-recognition`
- [ ] All requirements installed: `pip install -r requirements.txt`
- [ ] Test imports passed: `python test_imports.py`
- [ ] Audio directory exists: 7,442 WAV files
- [ ] Enough disk space: ~500MB cho model + logs

---

**Ready to go!** 🚀

Chạy ngay:
```bash
python test_imports.py && python run_pipeline.py --quick
```

