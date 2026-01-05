# 📝 Command Reference Guide

Danh sách tất cả lệnh thường dùng trong project.

---

## 🚀 Quick Start

```bash
# 1. Activate environment
conda activate voice-recognition

# 2. Test everything
python test_imports.py

# 3. Run pipeline (10 epochs)
python run_pipeline.py --quick
```

---

## 🏃 Running the Pipeline

### Quick Test (10 epochs, ~10-15 phút)
```bash
python run_pipeline.py --quick
```

### Full Training (50 epochs, ~45-60 phút)
```bash
python run_pipeline.py --epochs 50
```

### Custom Configuration
```bash
python run_pipeline.py --epochs 20 --batch-size 16
```

### With Environment Variables (nếu gặp lỗi)
```bash
NUMBA_CACHE_DIR=/tmp python run_pipeline.py --quick
```

---

## 🧪 Testing

### Test All Imports
```bash
python test_imports.py
```

### Test Configuration
```bash
python src/config.py
```

### Test Preprocessing
```bash
python src/preprocess.py
```

### Test Dataset
```bash
python src/dataset.py
```

### Test Model Creation
```bash
python -c "from src.model import create_model; model = create_model(); print('✓ Model OK')"
```

---

## 🎯 Training (Manual)

### Train Model Directly
```bash
python src/train.py
```

### With Custom Parameters
```bash
python -c "
from src.train import train_model
train_model(
    data_dir='data/CREMA-D/AudioWAV',
    batch_size=16,
    epochs=10,
    learning_rate=0.0001
)
"
```

---

## 📊 Evaluation

### Evaluate Latest Model
```bash
python src/evaluate.py
```

### Custom Evaluation
```bash
python -c "
from src.evaluate import evaluate_model
evaluate_model('best_model.keras', X_test, y_test)
"
```

---

## 🎤 Prediction

### Predict Single File
```bash
python src/predict.py path/to/audio.wav
```

### Predict with Probabilities
```bash
python -c "
from src.predict import EmotionPredictor
predictor = EmotionPredictor('best_model.keras')
emotion, probs = predictor.predict('path/to/audio.wav')
print(f'Emotion: {emotion}')
print(f'Probabilities: {probs}')
"
```

---

## 🎙️ Recording

### Record from Microphone
```bash
python src/record.py
```

### Custom Recording Duration
```bash
python -c "
from src.record import record_audio
audio = record_audio(duration=5, sample_rate=22050)
print(f'Recorded {len(audio)} samples')
"
```

---

## 🌐 Demo App

### Using Wrapper (KHUYẾN NGHỊ - Tránh mutex lock errors)
```bash
python run_demo.py
```

### Using Shell Script
```bash
./run_demo.sh
```

### Direct Streamlit (có thể gặp lỗi trên macOS)
```bash
streamlit run demo/app.py
```

### With Environment Variables
```bash
NUMBA_CACHE_DIR=/tmp \
TF_CPP_MIN_LOG_LEVEL=2 \
KMP_DUPLICATE_LIB_OK=TRUE \
streamlit run demo/app.py
```

### Custom Port
```bash
python run_demo.py
# Nếu port 8501 đã dùng, sửa trong run_demo.py
```

---

## 📓 Jupyter Notebooks

### Start Jupyter
```bash
jupyter notebook
```

### Run Exploratory Notebook
```bash
jupyter notebook notebooks/exploratory.ipynb
```

### Run Preprocessing Notebook
```bash
jupyter notebook src/preprocess.ipynb
```

---

## 🔍 Debugging

### Check Audio Directory
```bash
python -c "
import sys
sys.path.append('src')
from config import AUDIO_WAV_DIR
print(f'Path: {AUDIO_WAV_DIR}')
print(f'Exists: {AUDIO_WAV_DIR.exists()}')
if AUDIO_WAV_DIR.exists():
    wav_files = list(AUDIO_WAV_DIR.glob('*.wav'))
    print(f'WAV files: {len(wav_files)}')
"
```

### Check Model
```bash
ls -lh best_model.keras models/final/*.keras 2>/dev/null
```

### Check Emotion Labels
```bash
python -c "
import sys
sys.path.append('src')
from config import EMOTION_MAP, EMOTION_NAMES
print('Emotion Map:', EMOTION_MAP)
print('Emotion Names:', EMOTION_NAMES)
"
```

### Test Single Audio File
```bash
python -c "
import sys
sys.path.append('src')
from preprocess import extract_emotion_from_filename, process_audio_file
file = 'data/CREMA-D/AudioWAV/1001_DFA_ANG_XX.wav'
emotion = extract_emotion_from_filename(file)
print(f'Emotion: {emotion}')
spec, label = process_audio_file(file)
print(f'Spectrogram: {spec.shape}, Label: {label}')
"
```

---

## 🛠️ Maintenance

### Clean Cache
```bash
rm -rf __pycache__ src/__pycache__ demo/__pycache__
rm -rf .ipynb_checkpoints src/.ipynb_checkpoints
```

### Clean Processed Data
```bash
rm -rf data/processed/*.npy
```

### Clean Models
```bash
rm -f best_model.keras
rm -rf models/checkpoints/*
rm -rf models/final/*
```

### Clean Logs
```bash
rm -rf results/logs/*
rm -rf results/metrics/*
rm -rf results/plots/*
```

### Clean Everything (Careful!)
```bash
rm -rf __pycache__ src/__pycache__
rm -rf data/processed/*.npy
rm -f best_model.keras
rm -rf models/checkpoints/* models/final/*
rm -rf results/logs/* results/metrics/* results/plots/*
```

---

## 📦 Package Management

### Install/Update Dependencies
```bash
pip install -r requirements.txt
```

### Freeze Current Dependencies
```bash
pip freeze > requirements_frozen.txt
```

### Update Single Package
```bash
pip install --upgrade tensorflow
```

### Check Installed Packages
```bash
conda list | grep -E "tensorflow|librosa|numpy|scikit-learn"
```

---

## 🔄 Environment Management

### Create New Environment
```bash
conda create -n voice-recognition python=3.9 -y
```

### Activate Environment
```bash
conda activate voice-recognition
```

### Deactivate Environment
```bash
conda deactivate
```

### List Environments
```bash
conda env list
```

### Remove Environment
```bash
conda env remove -n voice-recognition
```

### Export Environment
```bash
conda env export > environment.yml
```

### Create from Environment File
```bash
conda env create -f environment.yml
```

---

## 📈 Monitoring

### Monitor Training (if TensorBoard enabled)
```bash
tensorboard --logdir results/logs
```

### Watch GPU Usage (if available)
```bash
watch -n 1 nvidia-smi
```

### Monitor Memory
```bash
# macOS
top -o mem

# Or
Activity Monitor app
```

---

## 🔐 Common Workflows

### Complete Pipeline from Scratch
```bash
# 1. Setup
conda activate voice-recognition
pip install -r requirements.txt

# 2. Test
python test_imports.py

# 3. Train
python run_pipeline.py --quick

# 4. Predict
python src/predict.py data/CREMA-D/AudioWAV/1001_DFA_ANG_XX.wav

# 5. Demo
python run_demo.py
```

### Quick Experiment
```bash
# Train with different parameters
python run_pipeline.py --epochs 10 --batch-size 16

# Evaluate
python src/evaluate.py

# Test predictions
python src/predict.py test_audio.wav
```

### Development Workflow
```bash
# 1. Make changes to code
vim src/model.py

# 2. Test syntax
python -m py_compile src/model.py

# 3. Test functionality
python test_imports.py

# 4. Quick train test
python run_pipeline.py --quick

# 5. If OK, full train
python run_pipeline.py --epochs 50
```

---

## 🆘 Troubleshooting Commands

### Kill Stuck Processes
```bash
pkill -9 python
pkill -9 streamlit
pkill -9 jupyter
```

### Check Port Usage
```bash
lsof -ti:8501  # Streamlit default port
lsof -ti:8888  # Jupyter default port
```

### Kill Process on Port
```bash
lsof -ti:8501 | xargs kill -9
```

### Reset Everything
```bash
# Kill processes
pkill -9 python streamlit jupyter

# Clean cache
rm -rf __pycache__ src/__pycache__

# Restart terminal
exit
# Open new terminal
conda activate voice-recognition
```

---

## 📋 Useful Aliases

Add to your `~/.zshrc` or `~/.bashrc`:

```bash
# Voice recognition project aliases
alias voice-activate='conda activate voice-recognition'
alias voice-cd='cd /Users/macbook/Library/CloudStorage/OneDrive-SwinburneUniversity/Documents/Project/voice'
alias voice-test='voice-cd && voice-activate && python test_imports.py'
alias voice-train='voice-cd && voice-activate && python run_pipeline.py --quick'
alias voice-demo='voice-cd && voice-activate && python run_demo.py'
alias voice-clean='voice-cd && rm -rf __pycache__ src/__pycache__ data/processed/*.npy'

# Combined
alias voice='voice-cd && voice-activate'
```

Then reload:
```bash
source ~/.zshrc  # or ~/.bashrc
```

Usage:
```bash
voice           # cd + activate
voice-test      # Run tests
voice-train     # Quick training
voice-demo      # Launch demo
```

---

## 💡 Tips & Tricks

### Run in Background
```bash
nohup python run_pipeline.py --epochs 50 > training.log 2>&1 &
```

### Multiple Experiments
```bash
# Experiment 1: Small batch
python run_pipeline.py --epochs 20 --batch-size 8

# Experiment 2: Large batch
python run_pipeline.py --epochs 20 --batch-size 64

# Experiment 3: More epochs
python run_pipeline.py --epochs 100 --batch-size 32
```

### Quick Smoke Test
```bash
# Test everything in 1 command
python test_imports.py && \
python src/config.py && \
echo "✅ All tests passed!"
```

---

**Last updated:** January 5, 2026

**See also:**
- `README.md` - Full documentation
- `QUICKSTART.md` - Quick start guide
- `TROUBLESHOOTING.md` - Error solutions
- `RUN_GUIDE.md` - How to run

