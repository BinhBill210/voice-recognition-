#!/bin/bash
# Safe wrapper to run Streamlit demo on macOS
# Fixes mutex lock error

echo "🚀 Starting Streamlit Demo..."
echo ""

# Set environment variables to prevent mutex lock
export NUMBA_CACHE_DIR=/tmp
export TF_CPP_MIN_LOG_LEVEL=2
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Change to script directory
cd "$(dirname "$0")"

# Check if conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  Warning: Conda environment not activated"
    echo "Please run: conda activate voice-recognition"
    echo ""
    exit 1
fi

# Check if model exists
if [ ! -f "best_model.keras" ] && [ ! -f "models/final/best_model.keras" ]; then
    echo "❌ No trained model found!"
    echo "Please train a model first: python safe_run.py --quick"
    echo ""
    exit 1
fi

echo "✅ Environment: $CONDA_DEFAULT_ENV"
echo "✅ Environment variables set"
echo ""
echo "📝 Note: If you still see mutex errors:"
echo "   1. Close all Python/Jupyter processes"
echo "   2. Restart terminal"
echo "   3. Run this script again"
echo ""
echo "Starting Streamlit on http://localhost:8501"
echo "Press Ctrl+C to stop"
echo ""

# Run Streamlit
streamlit run demo/app.py

