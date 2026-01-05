"""
Audio recording module for real-time emotion prediction.
Supports recording from microphone and saving to file.
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union
import time
from datetime import datetime

from config import SAMPLE_RATE, RESULTS_DIR
from predict import EmotionPredictor, FINAL_MODELS_DIR


class AudioRecorder:
    """
    Audio recorder for emotion recognition.
    Records from microphone and saves to file.
    """
    
    def __init__(self, 
                 sample_rate: int = SAMPLE_RATE,
                 channels: int = 1):
        """
        Initialize audio recorder.
        
        Args:
            sample_rate: Sample rate for recording
            channels: Number of audio channels (1 for mono)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording = None
        
    def list_devices(self) -> None:
        """List available audio input devices."""
        print("\nAvailable Audio Devices:")
        print("=" * 60)
        print(sd.query_devices())
        print("=" * 60)
    
    def record(self, 
              duration: float = 3.0,
              device: Optional[int] = None) -> np.ndarray:
        """
        Record audio from microphone.
        
        Args:
            duration: Recording duration in seconds
            device: Device index (None for default)
            
        Returns:
            Recorded audio as numpy array
        """
        print(f"\n🎤 Recording for {duration} seconds...")
        print("Speak now!")
        
        # Record
        self.recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            device=device,
            dtype='float32'
        )
        
        # Wait for recording to finish
        sd.wait()
        
        print("✓ Recording complete!")
        
        # Convert to mono if needed
        if self.channels > 1:
            self.recording = np.mean(self.recording, axis=1)
        else:
            self.recording = self.recording.flatten()
        
        return self.recording
    
    def save(self, 
            output_path: Union[str, Path],
            audio: Optional[np.ndarray] = None) -> Path:
        """
        Save recorded audio to file.
        
        Args:
            output_path: Path to save audio file
            audio: Audio array (uses last recording if None)
            
        Returns:
            Path to saved file
        """
        if audio is None:
            audio = self.recording
        
        if audio is None:
            raise ValueError("No audio to save. Record first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        sf.write(output_path, audio, self.sample_rate)
        print(f"✓ Audio saved to {output_path}")
        
        return output_path
    
    def play(self, audio: Optional[np.ndarray] = None) -> None:
        """
        Play back recorded audio.
        
        Args:
            audio: Audio array (uses last recording if None)
        """
        if audio is None:
            audio = self.recording
        
        if audio is None:
            raise ValueError("No audio to play. Record first.")
        
        print("▶ Playing audio...")
        sd.play(audio, self.sample_rate)
        sd.wait()
        print("✓ Playback complete")
    
    def record_and_save(self,
                       duration: float = 3.0,
                       output_dir: Optional[Path] = None,
                       device: Optional[int] = None) -> Path:
        """
        Record audio and save to file with timestamp.
        
        Args:
            duration: Recording duration in seconds
            output_dir: Output directory (default: results/recordings)
            device: Device index
            
        Returns:
            Path to saved file
        """
        # Record
        audio = self.record(duration, device)
        
        # Generate filename with timestamp
        if output_dir is None:
            output_dir = RESULTS_DIR / "recordings"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        output_path = output_dir / filename
        
        # Save
        return self.save(output_path, audio)


class RealTimeEmotionRecognizer:
    """
    Real-time emotion recognition from microphone input.
    Combines recording and prediction.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize real-time recognizer.
        
        Args:
            model_path: Path to trained model (uses latest if None)
        """
        self.recorder = AudioRecorder()
        
        # Load model
        if model_path is None:
            model_files = list(FINAL_MODELS_DIR.glob('*.keras'))
            if not model_files:
                raise FileNotFoundError("No trained model found.")
            model_path = max(model_files, key=lambda p: p.stat().st_mtime)
        
        self.predictor = EmotionPredictor(model_path)
    
    def recognize(self,
                 duration: float = 3.0,
                 save_recording: bool = True,
                 play_back: bool = False) -> Dict:
        """
        Record audio and predict emotion.
        
        Args:
            duration: Recording duration in seconds
            save_recording: Whether to save recording
            play_back: Whether to play back recording
            
        Returns:
            Prediction result dictionary
        """
        # Record
        if save_recording:
            audio_path = self.recorder.record_and_save(duration)
        else:
            audio = self.recorder.record(duration)
            # Save to temporary file for prediction
            temp_path = RESULTS_DIR / "temp_recording.wav"
            audio_path = self.recorder.save(temp_path, audio)
        
        # Play back if requested
        if play_back:
            self.recorder.play()
        
        # Predict
        print("\n🧠 Analyzing emotion...")
        result = self.predictor.predict(audio_path)
        
        # Print result
        self.predictor.print_prediction(result)
        
        return result
    
    def continuous_recognition(self,
                             duration: float = 3.0,
                             num_recordings: int = 5,
                             delay: float = 1.0) -> List[Dict]:
        """
        Continuously record and predict emotions.
        
        Args:
            duration: Duration of each recording
            num_recordings: Number of recordings to make
            delay: Delay between recordings
            
        Returns:
            List of prediction results
        """
        results = []
        
        print(f"\n🎙️ Starting continuous recognition...")
        print(f"Will record {num_recordings} times, {duration}s each")
        print("=" * 60)
        
        for i in range(num_recordings):
            print(f"\n[Recording {i+1}/{num_recordings}]")
            
            try:
                result = self.recognize(duration, save_recording=True)
                results.append(result)
            except Exception as e:
                print(f"❌ Error: {e}")
            
            # Delay before next recording
            if i < num_recordings - 1:
                print(f"\nWaiting {delay}s before next recording...")
                time.sleep(delay)
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        emotion_counts = {}
        for result in results:
            if 'predicted_emotion' in result:
                emotion = result['predicted_emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print("\nEmotion Distribution:")
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {emotion}: {count} ({count/len(results)*100:.1f}%)")
        
        return results


def record_and_predict(duration: float = 3.0,
                      save: bool = True,
                      play: bool = False) -> Dict:
    """
    Convenience function to record and predict emotion.
    
    Args:
        duration: Recording duration in seconds
        save: Whether to save recording
        play: Whether to play back recording
        
    Returns:
        Prediction result
    """
    recognizer = RealTimeEmotionRecognizer()
    return recognizer.recognize(duration, save, play)


if __name__ == "__main__":
    """Test recording module."""
    import sys
    
    print("=" * 60)
    print("REAL-TIME EMOTION RECOGNITION")
    print("=" * 60)
    
    # Check if model exists
    model_files = list(FINAL_MODELS_DIR.glob('*.keras'))
    if not model_files:
        print("\n❌ No trained model found. Please train a model first.")
        sys.exit(1)
    
    print("\nInstructions:")
    print("- Speak clearly into your microphone")
    print("- Express an emotion (anger, happiness, sadness, etc.)")
    print("- Recording will start automatically")
    
    try:
        # Create recognizer
        recognizer = RealTimeEmotionRecognizer()
        
        # List available devices
        recognizer.recorder.list_devices()
        
        # Single recognition
        print("\n" + "=" * 60)
        print("SINGLE RECORDING")
        print("=" * 60)
        result = recognizer.recognize(duration=3.0, save_recording=True, play_back=False)
        
        print("\n✓ Test completed!")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Recording interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
