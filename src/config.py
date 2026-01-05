"""
Configuration file for Speech Emotion Recognition project.
Contains all hyperparameters, paths, and settings.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Get project root directory (voice/)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data" / "CREMA-D"
AUDIO_WAV_DIR = DATA_DIR / "AudioWAV"
AUDIO_MP3_DIR = DATA_DIR / "AudioMP3"

# Output directories
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
FINAL_MODELS_DIR = MODELS_DIR / "final"

RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = RESULTS_DIR / "logs"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"

# Create directories if they don't exist
for directory in [MODELS_DIR, CHECKPOINTS_DIR, FINAL_MODELS_DIR, 
                  RESULTS_DIR, LOGS_DIR, METRICS_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================================
# EMOTION LABELS
# ============================================================================

# Emotion mapping from CREMA-D filenames
EMOTION_MAP: Dict[str, int] = {
    'ANG': 0,  # Anger
    'HAP': 1,  # Happiness
    'SAD': 2,  # Sadness
    'NEU': 3,  # Neutral
    'DIS': 4,  # Disgust
    'FEA': 5   # Fear
}

# Reverse mapping (index to emotion name)
EMOTION_NAMES: List[str] = ['ANG', 'HAP', 'SAD', 'NEU', 'DIS', 'FEA']

# Full emotion names
EMOTION_FULL_NAMES: Dict[str, str] = {
    'ANG': 'Anger',
    'HAP': 'Happiness',
    'SAD': 'Sadness',
    'NEU': 'Neutral',
    'DIS': 'Disgust',
    'FEA': 'Fear'
}

NUM_CLASSES: int = len(EMOTION_MAP)


# ============================================================================
# AUDIO PROCESSING PARAMETERS
# ============================================================================

# Audio loading parameters
SAMPLE_RATE: int = 22050  # Hz
DURATION: float = None  # None = load full audio, or specify in seconds
MONO: bool = True  # Convert to mono

# Mel spectrogram parameters
N_MELS: int = 128  # Number of Mel bands
N_FFT: int = 2048  # FFT window size
HOP_LENGTH: int = 512  # Number of samples between successive frames
WIN_LENGTH: int = None  # Window length (None = N_FFT)
WINDOW: str = 'hann'  # Window function
F_MIN: float = 0.0  # Minimum frequency
F_MAX: float = None  # Maximum frequency (None = SAMPLE_RATE / 2)

# Spectrogram shape (time_steps, mel_bands)
SPECTROGRAM_SHAPE: Tuple[int, int] = (128, 128)
TARGET_TIME_STEPS: int = SPECTROGRAM_SHAPE[0]
TARGET_MEL_BANDS: int = SPECTROGRAM_SHAPE[1]

# Input shape for CNN (height, width, channels)
INPUT_SHAPE: Tuple[int, int, int] = (TARGET_TIME_STEPS, TARGET_MEL_BANDS, 1)


# ============================================================================
# DATA AUGMENTATION PARAMETERS
# ============================================================================

# Audio augmentation
USE_AUGMENTATION: bool = True
AUGMENTATION_PARAMS: Dict = {
    'time_stretch': {
        'enabled': True,
        'rate_range': (0.8, 1.2)  # Speed up or slow down
    },
    'pitch_shift': {
        'enabled': True,
        'n_steps_range': (-2, 2)  # Semitones
    },
    'noise_injection': {
        'enabled': True,
        'noise_factor': 0.005
    },
    'time_shift': {
        'enabled': True,
        'shift_max': 0.2  # Maximum shift as fraction of length
    }
}


# ============================================================================
# DATA SPLITTING PARAMETERS
# ============================================================================

# Train/validation/test split ratios
TEST_SIZE: float = 0.1  # 10% for testing
VALIDATION_SIZE: float = 0.2  # 20% of remaining (18% of total)
# Training will be 72% of total

# Random seed for reproducibility
RANDOM_SEED: int = 42

# Stratify by emotion labels
STRATIFY: bool = True


# ============================================================================
# MODEL ARCHITECTURE PARAMETERS
# ============================================================================

# CNN architecture
CNN_PARAMS: Dict = {
    'conv_blocks': [
        {'filters': 32, 'kernel_size': (3, 3), 'pool_size': (2, 2)},
        {'filters': 64, 'kernel_size': (3, 3), 'pool_size': (2, 2)},
        {'filters': 128, 'kernel_size': (3, 3), 'pool_size': (2, 2)},
        {'filters': 256, 'kernel_size': (3, 3), 'pool_size': (2, 2)},
    ],
    'dense_layers': [512, 256],
    'dropout_rate': 0.5,
    'conv_dropout_rate': 0.25,
    'use_batch_norm': True,
    'activation': 'relu'
}


# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

# Training hyperparameters
BATCH_SIZE: int = 32
EPOCHS: int = 100
LEARNING_RATE: float = 0.001

# Optimizer
OPTIMIZER: str = 'adam'  # 'adam', 'sgd', 'rmsprop'
OPTIMIZER_PARAMS: Dict = {
    'adam': {'learning_rate': LEARNING_RATE, 'beta_1': 0.9, 'beta_2': 0.999},
    'sgd': {'learning_rate': LEARNING_RATE, 'momentum': 0.9},
    'rmsprop': {'learning_rate': LEARNING_RATE, 'rho': 0.9}
}

# Loss function
LOSS: str = 'sparse_categorical_crossentropy'  # For integer labels

# Metrics
METRICS: List[str] = ['accuracy']


# ============================================================================
# CALLBACKS PARAMETERS
# ============================================================================

# Early stopping
EARLY_STOPPING: Dict = {
    'enabled': True,
    'monitor': 'val_accuracy',
    'patience': 15,
    'mode': 'max',
    'restore_best_weights': True,
    'verbose': 1
}

# Reduce learning rate on plateau
REDUCE_LR: Dict = {
    'enabled': True,
    'monitor': 'val_loss',
    'factor': 0.5,
    'patience': 7,
    'min_lr': 1e-7,
    'mode': 'min',
    'verbose': 1
}

# Model checkpoint
MODEL_CHECKPOINT: Dict = {
    'enabled': True,
    'monitor': 'val_accuracy',
    'mode': 'max',
    'save_best_only': True,
    'save_weights_only': False,
    'verbose': 1
}

# TensorBoard
TENSORBOARD: Dict = {
    'enabled': True,
    'update_freq': 'epoch',
    'profile_batch': 0
}

# CSV Logger
CSV_LOGGER: Dict = {
    'enabled': True,
    'append': False
}


# ============================================================================
# EVALUATION PARAMETERS
# ============================================================================

# Confusion matrix
PLOT_CONFUSION_MATRIX: bool = True
CONFUSION_MATRIX_NORMALIZE: str = 'true'  # 'true', 'pred', 'all', or None

# Classification report
SAVE_CLASSIFICATION_REPORT: bool = True

# ROC curves (for multi-class)
PLOT_ROC_CURVES: bool = True


# ============================================================================
# INFERENCE PARAMETERS
# ============================================================================

# Prediction confidence threshold
CONFIDENCE_THRESHOLD: float = 0.5

# Top-K predictions to return
TOP_K: int = 3


# ============================================================================
# LOGGING AND VERBOSITY
# ============================================================================

# Verbosity levels
VERBOSE_TRAINING: int = 1  # 0 = silent, 1 = progress bar, 2 = one line per epoch
VERBOSE_EVALUATION: int = 1

# Logging
LOG_LEVEL: str = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_timestamp() -> str:
    """Get current timestamp string for file naming."""
    from datetime import datetime
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def get_model_path(model_name: str = None, timestamp: bool = True) -> Path:
    """
    Get path for saving model.
    
    Args:
        model_name: Name of the model (default: 'emotion_model')
        timestamp: Whether to include timestamp in filename
    
    Returns:
        Path object for model file
    """
    if model_name is None:
        model_name = 'emotion_model'
    
    if timestamp:
        filename = f"{model_name}_{get_timestamp()}.keras"
    else:
        filename = f"{model_name}.keras"
    
    return FINAL_MODELS_DIR / filename


def get_checkpoint_path(timestamp: bool = True) -> Path:
    """Get path for model checkpoint."""
    if timestamp:
        filename = f"checkpoint_{get_timestamp()}.keras"
    else:
        filename = "checkpoint.keras"
    
    return CHECKPOINTS_DIR / filename


def get_log_dir(timestamp: bool = True) -> Path:
    """Get directory for TensorBoard logs."""
    if timestamp:
        dirname = f"tensorboard_{get_timestamp()}"
    else:
        dirname = "tensorboard"
    
    return LOGS_DIR / dirname


def get_csv_log_path(timestamp: bool = True) -> Path:
    """Get path for CSV training log."""
    if timestamp:
        filename = f"training_log_{get_timestamp()}.csv"
    else:
        filename = "training_log.csv"
    
    return LOGS_DIR / filename


def get_history_path(timestamp: bool = True) -> Path:
    """Get path for training history JSON."""
    if timestamp:
        filename = f"history_{get_timestamp()}.json"
    else:
        filename = "history.json"
    
    return LOGS_DIR / filename


def print_config() -> None:
    """Print configuration summary."""
    print("=" * 70)
    print("SPEECH EMOTION RECOGNITION - CONFIGURATION")
    print("=" * 70)
    print(f"\nPROJECT PATHS:")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Audio Data: {AUDIO_WAV_DIR}")
    print(f"  Models Dir: {MODELS_DIR}")
    print(f"  Results Dir: {RESULTS_DIR}")
    
    print(f"\nEMOTIONS:")
    print(f"  Number of classes: {NUM_CLASSES}")
    print(f"  Emotions: {', '.join(EMOTION_NAMES)}")
    
    print(f"\nAUDIO PROCESSING:")
    print(f"  Sample Rate: {SAMPLE_RATE} Hz")
    print(f"  Mel Bands: {N_MELS}")
    print(f"  Spectrogram Shape: {SPECTROGRAM_SHAPE}")
    print(f"  Input Shape: {INPUT_SHAPE}")
    
    print(f"\nMODEL ARCHITECTURE:")
    print(f"  Conv Blocks: {len(CNN_PARAMS['conv_blocks'])}")
    print(f"  Dense Layers: {CNN_PARAMS['dense_layers']}")
    print(f"  Dropout Rate: {CNN_PARAMS['dropout_rate']}")
    
    print(f"\nTRAINING:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Optimizer: {OPTIMIZER}")
    
    print(f"\nDATA SPLIT:")
    print(f"  Test: {TEST_SIZE * 100:.1f}%")
    print(f"  Validation: {VALIDATION_SIZE * 100:.1f}%")
    print(f"  Train: {(1 - TEST_SIZE - VALIDATION_SIZE * (1 - TEST_SIZE)) * 100:.1f}%")
    
    print("=" * 70)


if __name__ == "__main__":
    # Print configuration when run directly
    print_config()
    
    # Check if data directory exists
    if not AUDIO_WAV_DIR.exists():
        print(f"\n⚠️  WARNING: Audio directory not found at {AUDIO_WAV_DIR}")
    else:
        wav_count = len(list(AUDIO_WAV_DIR.glob('*.wav')))
        print(f"\n✓ Found {wav_count} WAV files in audio directory")

