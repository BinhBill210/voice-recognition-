"""
2D CNN model for speech emotion recognition.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple


def create_cnn_model(input_shape: Tuple[int, int, int] = (128, 128, 1),
                     num_classes: int = 6) -> keras.Model:
    """
    Create a 2D CNN model for emotion recognition.
    
    Args:
        input_shape: Shape of input spectrogram (time_steps, mel_bands, channels)
        num_classes: Number of emotion classes (6 for CREMA-D)
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth convolutional block
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def compile_model(model: keras.Model, 
                  learning_rate: float = 0.001) -> keras.Model:
    """
    Compile the model with optimizer and loss function.
    
    Args:
        model: Keras model
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled model
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_model(input_shape: Tuple[int, int, int] = (128, 128, 1),
                 num_classes: int = 6,
                 learning_rate: float = 0.001) -> keras.Model:
    """
    Create and compile the CNN model.
    
    Args:
        input_shape: Shape of input spectrogram
        num_classes: Number of emotion classes
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled Keras model
    """
    model = create_cnn_model(input_shape, num_classes)
    model = compile_model(model, learning_rate)
    return model

