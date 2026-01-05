# 🚀 START HERE - Quick Reference

## ⚡ Chạy ngay (3 bước):

```bash
# 1. Activate environment
conda activate voice-recognition

# 2. Test (30 giây)
python test_imports.py

# 3. Train (10-15 phút)
python run_pipeline.py --quick
```

---

## 📚 Documentation

- **`README.md`** - Tài liệu đầy đủ (17KB)
- **`QUICKSTART.md`** - Hướng dẫn nhanh (tiếng Việt)
- **`FINAL_STATUS.md`** - Trạng thái project & verification

---

## 🎯 Common Tasks

### Training
```bash
# Quick (10 epochs)
python run_pipeline.py --quick

# Full (50 epochs)
python run_pipeline.py --epochs 50
```

### Testing
```bash
# Test all imports
python test_imports.py

# Test config
python src/config.py

# Test preprocessing
python src/preprocess.py
```

### Prediction
```bash
# Predict emotion from audio file
python src/predict.py data/CREMA-D/AudioWAV/1001_DFA_ANG_XX.wav
```

### Demo
```bash
# Launch web demo
streamlit run demo/app.py
```

---

## 📁 Project Structure

```
voice/
├── src/                    # 9 Python modules
├── demo/app.py             # Streamlit demo
├── notebooks/              # Jupyter notebooks
├── run_pipeline.py         # Main runner
├── test_imports.py         # Test script
└── README.md               # Full docs
```

---

## 🔗 Links

- **GitHub:** https://github.com/BinhBill210/voice-recognition-.git
- **Dataset:** CREMA-D (7,442 audio files)
- **Emotions:** ANG, HAP, SAD, NEU, DIS, FEA (6 classes)

---

## ⚠️ Common Issues

### Lỗi: "Module not found"
```bash
pip install -r requirements.txt
```

### Lỗi: "AudioWAV directory not found"
```bash
# Check path
python -c "import sys; sys.path.append('src'); from config import AUDIO_WAV_DIR; print(AUDIO_WAV_DIR)"
```

### Lỗi: TensorFlow mutex lock (macOS)
```bash
NUMBA_CACHE_DIR=/tmp python run_pipeline.py --quick
```

---

**Status:** ✅ Ready to use | **Last updated:** Jan 5, 2026

