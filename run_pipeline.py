"""
Complete pipeline runner for Speech Emotion Recognition project.
Runs data preparation, training, and evaluation.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

import argparse
from src.train import train_model
from src.config import AUDIO_WAV_DIR, FINAL_MODELS_DIR


def run_full_pipeline(epochs: int = 50, batch_size: int = 32):
    """
    Run complete pipeline: train -> evaluate.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    print("\n" + "="*70)
    print("SPEECH EMOTION RECOGNITION - FULL PIPELINE")
    print("="*70)
    
    # Check if data directory exists
    if not AUDIO_WAV_DIR.exists():
        print(f"\n❌ Error: Audio directory not found at {AUDIO_WAV_DIR}")
        print("Please ensure the data/CREMA-D/AudioWAV folder exists.")
        return
    
    # Training (includes data loading, preprocessing, and evaluation)
    print("\n[1/1] MODEL TRAINING & EVALUATION")
    print("-"*70)
    print(f"Audio directory: {AUDIO_WAV_DIR}")
    print(f"Found WAV files: {len(list(AUDIO_WAV_DIR.glob('*.wav')))}")
    
    model, history = train_model(
        data_dir=str(AUDIO_WAV_DIR),
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        test_split=0.1,
        learning_rate=0.001
    )
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nModel saved to: best_model.keras")
    print("\nNext steps:")
    print("  1. View training history: Check best_model.keras")
    print("  2. Test predictions: python src/predict.py")
    print("  3. Try demo app: streamlit run demo/app.py")
    print("  4. Record audio: python src/record.py")
    print("  5. Evaluate model: python src/evaluate.py")


def main():
    parser = argparse.ArgumentParser(
        description="Speech Emotion Recognition Pipeline"
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: 10 epochs for testing'
    )
    
    args = parser.parse_args()
    
    epochs = 10 if args.quick else args.epochs
    
    try:
        run_full_pipeline(epochs=epochs, batch_size=args.batch_size)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

