# 🚀 START HERE - Quick Reference

## ⚡ Chạy ngay (3 bước):

```bash
# 1. Activate environment
conda activate voice-recognition

# 2. Test (30 giây)
python test_imports.py

# 3. Train (10-15 phút) - Dùng safe_run để tránh mutex lock error
python safe_run.py --quick

# Hoặc nếu muốn chạy trực tiếp:
# python run_pipeline.py --quick
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
# Quick (10 epochs) - KHUYẾN NGHỊ dùng safe_run trên macOS
python safe_run.py --quick

# Full (50 epochs)
python safe_run.py --epochs 50

# Hoặc chạy trực tiếp (có thể thấy mutex warnings):
# python run_pipeline.py --quick
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
# Giải pháp 1: Dùng safe wrapper (KHUYẾN NGHỊ)
python safe_run.py --quick

# Giải pháp 2: Set environment variables
NUMBA_CACHE_DIR=/tmp python run_pipeline.py --quick

# Giải pháp 3: Ignore warnings (không ảnh hưởng chức năng)
# Lỗi "[mutex.cc : 452] RAW: Lock blocking" là bình thường trên macOS
```

---

**Status:** ✅ Ready to use | **Last updated:** Jan 5, 2026

