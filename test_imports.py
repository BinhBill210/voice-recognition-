#!/usr/bin/env python
"""
Quick test script to verify all imports and basic functionality.
Run this before starting the full pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_imports():
    """Test all module imports."""
    print("=" * 70)
    print("TESTING MODULE IMPORTS")
    print("=" * 70)
    
    tests = [
        ("config", "from config import *"),
        ("preprocess", "from preprocess import load_dataset, extract_emotion_from_filename"),
        ("dataset", "from dataset import CremaDDataset"),
        ("model", "from model import create_model"),
        ("train", "from train import train_model"),
        ("evaluate", "from evaluate import evaluate_model"),
        ("predict", "from predict import load_trained_model"),
        ("record", "from record import record_audio"),
    ]
    
    failed = []
    for name, import_stmt in tests:
        try:
            print(f"Testing {name:20s}...", end=" ")
            exec(import_stmt)
            print("✓ OK")
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed.append(name)
    
    print("\n" + "=" * 70)
    if failed:
        print(f"❌ {len(failed)} modules failed: {', '.join(failed)}")
        return False
    else:
        print(f"✅ All {len(tests)} modules imported successfully!")
        return True

def test_config():
    """Test configuration."""
    print("\n" + "=" * 70)
    print("TESTING CONFIGURATION")
    print("=" * 70)
    
    from config import AUDIO_WAV_DIR, EMOTION_MAP, EMOTION_NAMES
    
    print(f"Audio directory: {AUDIO_WAV_DIR}")
    print(f"Exists: {AUDIO_WAV_DIR.exists()}")
    
    if AUDIO_WAV_DIR.exists():
        wav_files = list(AUDIO_WAV_DIR.glob('*.wav'))
        print(f"✓ Found {len(wav_files)} WAV files")
        
        print(f"\nEmotion map: {EMOTION_MAP}")
        print(f"Emotion names: {EMOTION_NAMES}")
        
        return len(wav_files) > 0
    else:
        print(f"❌ Audio directory not found!")
        return False

def test_preprocess():
    """Test preprocessing functions."""
    print("\n" + "=" * 70)
    print("TESTING PREPROCESSING")
    print("=" * 70)
    
    from config import AUDIO_WAV_DIR
    from preprocess import extract_emotion_from_filename, process_audio_file
    
    if not AUDIO_WAV_DIR.exists():
        print("❌ Cannot test: Audio directory not found")
        return False
    
    # Get first WAV file
    wav_files = list(AUDIO_WAV_DIR.glob('*.wav'))
    if not wav_files:
        print("❌ No WAV files found")
        return False
    
    test_file = wav_files[0]
    print(f"Testing with: {test_file.name}")
    
    # Test emotion extraction
    emotion = extract_emotion_from_filename(str(test_file))
    print(f"Extracted emotion: {emotion}")
    
    # Test audio processing
    try:
        spec, label = process_audio_file(str(test_file))
        print(f"Spectrogram shape: {spec.shape}")
        print(f"Label: {label}")
        print("✓ Preprocessing works!")
        return True
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return False

def test_model():
    """Test model creation."""
    print("\n" + "=" * 70)
    print("TESTING MODEL CREATION")
    print("=" * 70)
    
    try:
        from model import create_model
        from config import SPECTROGRAM_SHAPE, NUM_CLASSES
        
        input_shape = (*SPECTROGRAM_SHAPE, 1)
        model = create_model(input_shape=input_shape, num_classes=NUM_CLASSES)
        
        total_params = model.count_params()
        print(f"✓ Model created successfully!")
        print(f"  Input shape: {input_shape}")
        print(f"  Output classes: {NUM_CLASSES}")
        print(f"  Total parameters: {total_params:,}")
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("VOICE RECOGNITION PROJECT - IMPORT TEST")
    print("=" * 70)
    print()
    
    results = {
        "Imports": test_imports(),
        "Configuration": test_config(),
        "Preprocessing": test_preprocess(),
        "Model": test_model(),
    }
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("\nYou can now run the pipeline:")
        print("  python run_pipeline.py --quick")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease fix the issues before running the pipeline.")
    print("=" * 70)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

