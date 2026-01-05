"""
Audio preprocessing for CREMA-D dataset.
Converts WAV files to 128-band Mel spectrograms (log scale).
"""

import os
import numpy as np
import librosa
from typing import Tuple, List
import glob


# Emotion mapping from CREMA-D filenames
EMOTION_MAP = {
    'ANG': 0,  # Anger
    'HAP': 1,  # Happiness
    'SAD': 2,  # Sadness
    'NEU': 3,  # Neutral
    'DIS': 4,  # Disgust
    'FEA': 5   # Fear
}

# Fixed spectrogram shape (time_steps, mel_bands)
SPECTROGRAM_SHAPE = (128, 128)  # (time, frequency)


def extract_emotion_from_filename(filename: str) -> str:
    """
    Extract emotion label from CREMA-D filename.
    Format: {actor_id}_{sentence}_{emotion}_{intensity}.wav
    Example: 1001_DFA_ANG_XX.wav -> ANG
    """
    parts = os.path.basename(filename).split('_')
    if len(parts) >= 3:
        emotion = parts[2]
        return emotion if emotion in EMOTION_MAP else None
    return None


def load_audio(file_path: str, sr: int = 22050) -> np.ndarray:
    """
    Load audio file using librosa.
    
    Args:
        file_path: Path to WAV file
        sr: Sample rate (default 22050)
    
    Returns:
        Audio signal as numpy array
    """
    audio, _ = librosa.load(file_path, sr=sr, mono=True)
    return audio


def audio_to_mel_spectrogram(audio: np.ndarray, 
                             sr: int = 22050,
                             n_mels: int = 128,
                             n_fft: int = 2048,
                             hop_length: int = 512) -> np.ndarray:
    """
    Convert audio to 128-band Mel spectrogram (log scale).
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_mels: Number of Mel bands (128)
        n_fft: FFT window size
        hop_length: Hop length for STFT
    
    Returns:
        Mel spectrogram as numpy array (mel_bands, time_steps)
    """
    # Compute Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db


def pad_or_crop_spectrogram(spectrogram: np.ndarray, 
                            target_shape: Tuple[int, int] = SPECTROGRAM_SHAPE) -> np.ndarray:
    """
    Pad or crop spectrogram to fixed shape.
    
    Args:
        spectrogram: Input spectrogram (mel_bands, time_steps)
        target_shape: Target shape (time_steps, mel_bands)
    
    Returns:
        Padded/cropped spectrogram with shape (time_steps, mel_bands)
    """
    mel_bands, time_steps = spectrogram.shape
    target_time, target_mel = target_shape
    
    # Transpose to (time_steps, mel_bands) for easier processing
    spec = spectrogram.T  # (time_steps, mel_bands)
    
    # Crop or pad time dimension
    if time_steps > target_time:
        # Crop: take middle portion
        start = (time_steps - target_time) // 2
        spec = spec[start:start + target_time, :]
    elif time_steps < target_time:
        # Pad: add zeros at the end
        pad_width = target_time - time_steps
        spec = np.pad(spec, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)
    
    # Crop or pad mel dimension
    if mel_bands > target_mel:
        # Crop: take first target_mel bands
        spec = spec[:, :target_mel]
    elif mel_bands < target_mel:
        # Pad: add zeros
        pad_width = target_mel - mel_bands
        spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
    
    return spec


def process_audio_file(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Process a single audio file: load, convert to Mel spectrogram, pad/crop.
    
    Args:
        file_path: Path to WAV file
    
    Returns:
        Tuple of (spectrogram, emotion_label)
        Returns (None, None) if emotion cannot be extracted
    """
    # Extract emotion label
    emotion = extract_emotion_from_filename(file_path)
    if emotion is None or emotion not in EMOTION_MAP:
        return None, None
    
    # Load audio
    audio = load_audio(file_path)
    
    # Convert to Mel spectrogram
    mel_spec = audio_to_mel_spectrogram(audio)
    
    # Pad or crop to fixed shape
    mel_spec_fixed = pad_or_crop_spectrogram(mel_spec)
    
    # Get emotion label index
    label = EMOTION_MAP[emotion]
    
    return mel_spec_fixed, label


def load_dataset(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all WAV files from directory and convert to spectrograms.
    
    Args:
        data_dir: Path to AudioWAV directory
    
    Returns:
        Tuple of (spectrograms, labels)
        spectrograms: numpy array of shape (n_samples, time_steps, mel_bands, 1)
        labels: numpy array of shape (n_samples,)
    """
    # Find all WAV files
    wav_files = glob.glob(os.path.join(data_dir, '*.wav'))
    
    spectrograms = []
    labels = []
    
    print(f"Processing {len(wav_files)} audio files...")
    
    for i, file_path in enumerate(wav_files):
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(wav_files)} files...")
        
        spec, label = process_audio_file(file_path)
        
        if spec is not None and label is not None:
            # Add channel dimension for CNN: (time, mel) -> (time, mel, 1)
            spec = np.expand_dims(spec, axis=-1)
            spectrograms.append(spec)
            labels.append(label)
    
    print(f"Successfully processed {len(spectrograms)} files.")
    
    # Convert to numpy arrays
    X = np.array(spectrograms)
    y = np.array(labels)
    
    return X, y


def get_emotion_names() -> List[str]:
    """Get list of emotion names in order."""
    return ['ANG', 'HAP', 'SAD', 'NEU', 'DIS', 'FEA']


if __name__ == "__main__":
    import sys
    
    # Test the preprocessing functions
    print("Testing preprocessing functions...")
    
    # Check if AudioWAV directory exists
    audio_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'CREMA-D',
        'AudioWAV'
    )
    
    if not os.path.exists(audio_dir):
        print(f"Error: AudioWAV directory not found at {audio_dir}")
        print("Please ensure the CREMA-D/AudioWAV folder exists.")
        sys.exit(1)
    
    # Test with a few sample files
    import glob
    wav_files = glob.glob(os.path.join(audio_dir, '*.wav'))
    
    if len(wav_files) == 0:
        print(f"Error: No WAV files found in {audio_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(wav_files)} WAV files")
    print(f"Testing with first 5 files...\n")
    
    # Test processing a few files
    for i, file_path in enumerate(wav_files[:5]):
        filename = os.path.basename(file_path)
        emotion = extract_emotion_from_filename(file_path)
        print(f"File {i+1}: {filename}")
        print(f"  Extracted emotion: {emotion}")
        
        if emotion:
            spec, label = process_audio_file(file_path)
            if spec is not None:
                print(f"  Spectrogram shape: {spec.shape}")
                print(f"  Label: {label} ({get_emotion_names()[label]})")
            else:
                print(f"  Failed to process file")
        else:
            print(f"  Could not extract emotion")
        print()
    
    print("Preprocessing functions test completed successfully!")

