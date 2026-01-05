#!/bin/bash
# Script to run Streamlit demo with proper environment variables

echo "🚀 Starting Speech Emotion Recognition Demo..."
echo ""

# Set environment variables to avoid mutex lock errors
export NUMBA_CACHE_DIR=/tmp
export TF_CPP_MIN_LOG_LEVEL=2
export KMP_DUPLICATE_LIB_OK=TRUE

# Check if conda env is activated
if [[ "$CONDA_DEFAULT_ENV" != "voice-recognition" ]]; then
    echo "⚠️  Warning: 'voice-recognition' conda environment is not activated"
    echo "Run: conda activate voice-recognition"
    echo ""
fi

# Check if model exists
if [ ! -f "best_model.keras" ] && [ ! -f "models/final/*.keras" ]; then
    echo "⚠️  Warning: No trained model found"
    echo "Please train a model first: python run_pipeline.py --quick"
    echo ""
fi

# Run Streamlit
echo "📱 Launching Streamlit app..."
echo "If browser doesn't open automatically, go to: http://localhost:8501"
echo ""

streamlit run demo/app.py

