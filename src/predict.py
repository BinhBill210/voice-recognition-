"""
Inference module for emotion prediction from audio files.
Supports single file and batch prediction.
"""

import numpy as np
import librosa
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional
import tensorflow as tf
from tensorflow import keras

from config import (
    EMOTION_NAMES, EMOTION_FULL_NAMES,
    SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH,
    SPECTROGRAM_SHAPE, CONFIDENCE_THRESHOLD, TOP_K,
    FINAL_MODELS_DIR
)
from preprocess import (
    load_audio, audio_to_mel_spectrogram,
    pad_or_crop_spectrogram
)


class EmotionPredictor:
    """
    Predictor for emotion recognition from audio files.
    """
    
    def __init__(self, model_path: Union[str, Path]):
        """
        Initialize predictor with trained model.
        
        Args:
            model_path: Path to trained Keras model
        """
        self.model_path = Path(model_path)
        self.model = None
        self.load_model()
        
    def load_model(self) -> None:
        """Load trained model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        print(f"Loading model from {self.model_path}...")
        self.model = keras.models.load_model(self.model_path)
        print("✓ Model loaded successfully")
    
    def preprocess_audio(self, audio_path: Union[str, Path]) -> np.ndarray:
        """
        Preprocess audio file for prediction.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed spectrogram ready for model input
        """
        # Load audio
        audio = load_audio(str(audio_path), sr=SAMPLE_RATE)
        
        # Convert to Mel spectrogram
        mel_spec = audio_to_mel_spectrogram(
            audio,
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
        
        # Pad or crop to fixed shape
        mel_spec_fixed = pad_or_crop_spectrogram(mel_spec, SPECTROGRAM_SHAPE)
        
        # Add channel dimension
        mel_spec_fixed = np.expand_dims(mel_spec_fixed, axis=-1)
        
        # Add batch dimension
        mel_spec_fixed = np.expand_dims(mel_spec_fixed, axis=0)
        
        return mel_spec_fixed
    
    def predict(self, 
                audio_path: Union[str, Path],
                return_probabilities: bool = True) -> Dict:
        """
        Predict emotion from audio file.
        
        Args:
            audio_path: Path to audio file
            return_probabilities: Whether to return probabilities for all classes
            
        Returns:
            Dictionary with prediction results
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Preprocess
        spectrogram = self.preprocess_audio(audio_path)
        
        # Predict
        probabilities = self.model.predict(spectrogram, verbose=0)[0]
        
        # Get top prediction
        predicted_idx = np.argmax(probabilities)
        predicted_emotion = EMOTION_NAMES[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        result = {
            'file': str(audio_path.name),
            'predicted_emotion': predicted_emotion,
            'emotion_full_name': EMOTION_FULL_NAMES[predicted_emotion],
            'confidence': confidence,
            'is_confident': confidence >= CONFIDENCE_THRESHOLD
        }
        
        if return_probabilities:
            result['all_probabilities'] = {
                emotion: float(prob)
                for emotion, prob in zip(EMOTION_NAMES, probabilities)
            }
            
            # Get top-k predictions
            top_k_idx = np.argsort(probabilities)[::-1][:TOP_K]
            result['top_k_predictions'] = [
                {
                    'emotion': EMOTION_NAMES[idx],
                    'emotion_full_name': EMOTION_FULL_NAMES[EMOTION_NAMES[idx]],
                    'confidence': float(probabilities[idx])
                }
                for idx in top_k_idx
            ]
        
        return result
    
    def predict_batch(self, 
                     audio_paths: List[Union[str, Path]],
                     verbose: bool = True) -> List[Dict]:
        """
        Predict emotions for multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            verbose: Whether to show progress
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i, audio_path in enumerate(audio_paths):
            if verbose:
                print(f"Processing {i+1}/{len(audio_paths)}: {Path(audio_path).name}")
            
            try:
                result = self.predict(audio_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                results.append({
                    'file': str(Path(audio_path).name),
                    'error': str(e)
                })
        
        return results
    
    def print_prediction(self, result: Dict) -> None:
        """
        Print prediction result in a formatted way.
        
        Args:
            result: Prediction result dictionary
        """
        if 'error' in result:
            print(f"❌ Error for {result['file']}: {result['error']}")
            return
        
        confidence_icon = "✓" if result['is_confident'] else "⚠"
        
        print("\n" + "=" * 60)
        print(f"FILE: {result['file']}")
        print("=" * 60)
        print(f"{confidence_icon} Predicted Emotion: {result['predicted_emotion']} ({result['emotion_full_name']})")
        print(f"   Confidence: {result['confidence']:.2%}")
        
        if 'top_k_predictions' in result:
            print(f"\nTop {len(result['top_k_predictions'])} Predictions:")
            for i, pred in enumerate(result['top_k_predictions'], 1):
                print(f"  {i}. {pred['emotion']} ({pred['emotion_full_name']}): {pred['confidence']:.2%}")
        
        if 'all_probabilities' in result:
            print("\nAll Probabilities:")
            for emotion, prob in sorted(result['all_probabilities'].items(), 
                                       key=lambda x: x[1], reverse=True):
                bar = '█' * int(prob * 50)
                print(f"  {emotion}: {bar} {prob:.2%}")
        
        print("=" * 60)


def predict_from_file(audio_path: Union[str, Path],
                     model_path: Optional[Union[str, Path]] = None,
                     verbose: bool = True) -> Dict:
    """
    Convenience function to predict emotion from audio file.
    
    Args:
        audio_path: Path to audio file
        model_path: Path to model (uses latest if None)
        verbose: Whether to print results
        
    Returns:
        Prediction result dictionary
    """
    # Find model
    if model_path is None:
        model_files = list(FINAL_MODELS_DIR.glob('*.keras'))
        if not model_files:
            raise FileNotFoundError("No trained model found. Please train a model first.")
        model_path = max(model_files, key=lambda p: p.stat().st_mtime)
        print(f"Using latest model: {model_path.name}")
    
    # Create predictor
    predictor = EmotionPredictor(model_path)
    
    # Predict
    result = predictor.predict(audio_path)
    
    if verbose:
        predictor.print_prediction(result)
    
    return result


if __name__ == "__main__":
    """Test prediction module."""
    import sys
    from config import AUDIO_WAV_DIR
    import glob
    
    print("Testing Emotion Prediction Module...\n")
    
    # Get sample audio files
    sample_files = glob.glob(str(AUDIO_WAV_DIR / "*.wav"))[:3]
    
    if not sample_files:
        print("No audio files found for testing.")
        sys.exit(1)
    
    # Find latest model
    model_files = list(FINAL_MODELS_DIR.glob('*.keras'))
    if not model_files:
        print("No trained model found. Please train a model first.")
        sys.exit(1)
    
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    print(f"Using model: {latest_model.name}\n")
    
    # Create predictor
    predictor = EmotionPredictor(latest_model)
    
    # Test single prediction
    print("\n" + "=" * 60)
    print("SINGLE FILE PREDICTION")
    print("=" * 60)
    result = predictor.predict(sample_files[0])
    predictor.print_prediction(result)
    
    # Test batch prediction
    print("\n" + "=" * 60)
    print("BATCH PREDICTION")
    print("=" * 60)
    results = predictor.predict_batch(sample_files, verbose=True)
    
    print(f"\n✓ Predicted {len(results)} files successfully!")
