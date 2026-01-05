"""
Training script for speech emotion recognition.
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from preprocess import load_dataset, get_emotion_names
from model import create_model


def train_model(data_dir: str,
                batch_size: int = 32,
                epochs: int = 50,
                validation_split: float = 0.2,
                test_split: float = 0.1,
                learning_rate: float = 0.001,
                random_seed: int = 42):
    """
    Load data, train model, and print accuracy.
    
    Args:
        data_dir: Path to AudioWAV directory
        batch_size: Batch size for training
        epochs: Number of training epochs
        validation_split: Fraction of data for validation
        test_split: Fraction of data for testing
        learning_rate: Learning rate for optimizer
        random_seed: Random seed for reproducibility
    """
    # Set random seeds for reproducibility
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    
    print("=" * 60)
    print("Speech Emotion Recognition - CREMA-D Dataset")
    print("=" * 60)
    
    # Load dataset
    print("\n[1/4] Loading and preprocessing audio files...")
    X, y = load_dataset(data_dir)
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Print class distribution
    emotion_names = get_emotion_names()
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for emotion_idx, count in zip(unique, counts):
        print(f"  {emotion_names[emotion_idx]}: {count}")
    
    # Split data: train, validation, test
    print("\n[2/4] Splitting dataset...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_split, random_state=random_seed, stratify=y
    )
    
    val_size = validation_split / (1 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_seed, stratify=y_temp
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create model
    print("\n[3/4] Creating model...")
    input_shape = X_train.shape[1:]
    num_classes = len(emotion_names)
    
    model = create_model(
        input_shape=input_shape,
        num_classes=num_classes,
        learning_rate=learning_rate
    )
    
    print("\nModel architecture:")
    model.summary()
    
    # Define callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath='best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print("\n[4/4] Training model...")
    print(f"Batch size: {batch_size}, Epochs: {epochs}")
    print("-" * 60)
    
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nTest Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Per-class accuracy
    print("\nPer-class accuracy on test set:")
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    for emotion_idx in range(num_classes):
        mask = y_test == emotion_idx
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred_classes[mask] == y_test[mask])
            print(f"  {emotion_names[emotion_idx]}: {class_acc:.4f} ({class_acc * 100:.2f}%)")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    
    return model, history


if __name__ == "__main__":
    # Path to AudioWAV directory
    from config import AUDIO_WAV_DIR
    audio_dir = str(AUDIO_WAV_DIR)
    
    # Check if directory exists
    if not os.path.exists(audio_dir):
        print(f"Error: AudioWAV directory not found at {audio_dir}")
        print("Please ensure the data/CREMA-D/AudioWAV folder exists.")
        exit(1)
    
    # Train model
    model, history = train_model(
        data_dir=audio_dir,
        batch_size=32,
        epochs=50,
        validation_split=0.2,
        test_split=0.1,
        learning_rate=0.001
    )

