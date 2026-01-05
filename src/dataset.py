"""
Dataset module for loading and parsing CREMA-D audio data.
Handles label parsing, data loading, and batch generation.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import json

from config import (
    AUDIO_WAV_DIR, EMOTION_MAP, EMOTION_NAMES,
    DATA_DIR, RANDOM_SEED
)


class CremaDDataset:
    """
    CREMA-D dataset handler for emotion recognition.
    Parses filenames and manages data loading.
    """
    
    def __init__(self, audio_dir: Path = AUDIO_WAV_DIR):
        """
        Initialize dataset.
        
        Args:
            audio_dir: Directory containing WAV files
        """
        self.audio_dir = Path(audio_dir)
        self.file_paths = []
        self.labels = []
        self.metadata_df = None
        
    def parse_filename(self, filename: str) -> Optional[Dict]:
        """
        Parse CREMA-D filename to extract metadata.
        
        Format: {ActorID}_{SentenceID}_{Emotion}_{EmotionLevel}.wav
        Example: 1001_DFA_ANG_XX.wav
        
        Args:
            filename: Audio filename
            
        Returns:
            Dictionary with metadata or None if invalid
        """
        parts = os.path.basename(filename).replace('.wav', '').split('_')
        
        if len(parts) != 4:
            return None
        
        actor_id, sentence_id, emotion, emotion_level = parts
        
        if emotion not in EMOTION_MAP:
            return None
        
        return {
            'filename': filename,
            'actor_id': actor_id,
            'sentence_id': sentence_id,
            'emotion': emotion,
            'emotion_label': EMOTION_MAP[emotion],
            'emotion_level': emotion_level
        }
    
    def create_metadata_csv(self, output_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Create metadata CSV from audio filenames.
        
        Args:
            output_path: Path to save CSV (optional)
            
        Returns:
            DataFrame with metadata
        """
        metadata_list = []
        
        # Get all WAV files
        wav_files = list(self.audio_dir.glob('*.wav'))
        
        for wav_file in wav_files:
            metadata = self.parse_filename(str(wav_file))
            if metadata:
                metadata['file_path'] = str(wav_file)
                metadata_list.append(metadata)
        
        df = pd.DataFrame(metadata_list)
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Metadata saved to {output_path}")
        
        self.metadata_df = df
        return df
    
    def load_metadata_csv(self, csv_path: Path) -> pd.DataFrame:
        """
        Load existing metadata CSV.
        
        Args:
            csv_path: Path to metadata CSV
            
        Returns:
            DataFrame with metadata
        """
        self.metadata_df = pd.read_csv(csv_path)
        return self.metadata_df
    
    def get_emotion_distribution(self) -> pd.DataFrame:
        """
        Get distribution of emotions in dataset.
        
        Returns:
            DataFrame with emotion counts and percentages
        """
        if self.metadata_df is None:
            self.create_metadata_csv()
        
        emotion_counts = self.metadata_df['emotion'].value_counts()
        emotion_dist = pd.DataFrame({
            'emotion': emotion_counts.index,
            'count': emotion_counts.values,
            'percentage': (emotion_counts.values / len(self.metadata_df) * 100).round(2)
        })
        
        return emotion_dist.sort_values('emotion')
    
    def get_actor_distribution(self) -> pd.DataFrame:
        """
        Get distribution by actor.
        
        Returns:
            DataFrame with actor statistics
        """
        if self.metadata_df is None:
            self.create_metadata_csv()
        
        actor_stats = self.metadata_df.groupby('actor_id').agg({
            'filename': 'count',
            'emotion': lambda x: x.value_counts().to_dict()
        }).rename(columns={'filename': 'total_samples'})
        
        return actor_stats
    
    def filter_by_emotion(self, emotions: List[str]) -> pd.DataFrame:
        """
        Filter dataset by specific emotions.
        
        Args:
            emotions: List of emotion codes (e.g., ['ANG', 'HAP'])
            
        Returns:
            Filtered DataFrame
        """
        if self.metadata_df is None:
            self.create_metadata_csv()
        
        return self.metadata_df[self.metadata_df['emotion'].isin(emotions)]
    
    def filter_by_actor(self, actor_ids: List[str]) -> pd.DataFrame:
        """
        Filter dataset by specific actors.
        
        Args:
            actor_ids: List of actor IDs
            
        Returns:
            Filtered DataFrame
        """
        if self.metadata_df is None:
            self.create_metadata_csv()
        
        return self.metadata_df[self.metadata_df['actor_id'].isin(actor_ids)]
    
    def get_train_val_test_split_info(self, 
                                      train_ratio: float = 0.7,
                                      val_ratio: float = 0.15,
                                      test_ratio: float = 0.15) -> Dict:
        """
        Get information about data splits.
        
        Args:
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            
        Returns:
            Dictionary with split information
        """
        if self.metadata_df is None:
            self.create_metadata_csv()
        
        total = len(self.metadata_df)
        
        return {
            'total_samples': total,
            'train_samples': int(total * train_ratio),
            'val_samples': int(total * val_ratio),
            'test_samples': int(total * test_ratio),
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio
        }
    
    def save_processed_data(self, 
                           X: np.ndarray, 
                           y: np.ndarray,
                           output_dir: Path,
                           prefix: str = '') -> None:
        """
        Save processed data to disk.
        
        Args:
            X: Feature array
            y: Label array
            output_dir: Output directory
            prefix: Prefix for filenames (e.g., 'train_', 'val_')
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        X_path = output_dir / f'{prefix}X.npy'
        y_path = output_dir / f'{prefix}y.npy'
        
        np.save(X_path, X)
        np.save(y_path, y)
        
        print(f"Saved {prefix}X.npy: {X.shape}")
        print(f"Saved {prefix}y.npy: {y.shape}")
        
        # Save label map
        label_map_path = output_dir / 'label_map.json'
        if not label_map_path.exists():
            label_map = {
                'emotion_map': EMOTION_MAP,
                'emotion_names': EMOTION_NAMES,
                'num_classes': len(EMOTION_MAP)
            }
            with open(label_map_path, 'w') as f:
                json.dump(label_map, f, indent=2)
            print(f"Saved label_map.json")
    
    def load_processed_data(self, 
                           data_dir: Path,
                           prefix: str = '') -> Tuple[np.ndarray, np.ndarray]:
        """
        Load processed data from disk.
        
        Args:
            data_dir: Directory containing processed data
            prefix: Prefix for filenames
            
        Returns:
            Tuple of (X, y)
        """
        data_dir = Path(data_dir)
        
        X_path = data_dir / f'{prefix}X.npy'
        y_path = data_dir / f'{prefix}y.npy'
        
        if not X_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"Processed data not found in {data_dir}")
        
        X = np.load(X_path)
        y = np.load(y_path)
        
        print(f"Loaded {prefix}X.npy: {X.shape}")
        print(f"Loaded {prefix}y.npy: {y.shape}")
        
        return X, y
    
    def print_summary(self) -> None:
        """Print dataset summary."""
        if self.metadata_df is None:
            self.create_metadata_csv()
        
        print("=" * 60)
        print("CREMA-D DATASET SUMMARY")
        print("=" * 60)
        print(f"Total samples: {len(self.metadata_df)}")
        print(f"Unique actors: {self.metadata_df['actor_id'].nunique()}")
        print(f"Unique sentences: {self.metadata_df['sentence_id'].nunique()}")
        print(f"\nEmotion distribution:")
        print(self.get_emotion_distribution().to_string(index=False))
        print("=" * 60)


def create_metadata_file(output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Convenience function to create metadata CSV.
    
    Args:
        output_path: Path to save CSV
        
    Returns:
        DataFrame with metadata
    """
    if output_path is None:
        output_path = DATA_DIR / 'metadata.csv'
    
    dataset = CremaDDataset()
    df = dataset.create_metadata_csv(output_path)
    dataset.print_summary()
    
    return df


if __name__ == "__main__":
    """Test dataset module."""
    print("Testing CREMA-D Dataset Module...\n")
    
    # Create dataset instance
    dataset = CremaDDataset()
    
    # Create metadata CSV
    metadata_path = DATA_DIR / 'metadata.csv'
    df = dataset.create_metadata_csv(metadata_path)
    
    # Print summary
    dataset.print_summary()
    
    # Show sample records
    print("\nSample records:")
    print(df.head(10).to_string(index=False))
    
    # Show emotion distribution
    print("\nEmotion distribution:")
    print(dataset.get_emotion_distribution().to_string(index=False))
    
    print("\n✓ Dataset module test completed!")

