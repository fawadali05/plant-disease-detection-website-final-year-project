"""
ML Model Training Script for Plant Disease Detection
====================================================
Ye script aapke PlantVillage dataset se model train karega.

Usage:
    python train_model.py --data_path /path/to/plantvillage/dataset
"""

import os
import argparse
import numpy as np
from pathlib import Path
import shutil
from datetime import datetime

# Try to import tensorflow - if not available, will use mock for demo
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("TensorFlow not installed. Install with: pip install tensorflow")


# Dataset classes (PlantVillage - 38 classes)
PLANTVILLAGE_CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy',
    'Corn___Cercospora_leaf_spot_Gray_leaf_spot', 'Corn___Common_rust', 
    'Corn___Northern_Leaf_Blight', 'Corn___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_Isariopsis_Leaf_Spot', 
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites_Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]


def create_model(num_classes, img_size=(224, 224)):
    """
    Create a CNN model using MobileNetV2 as base.
    
    Args:
        num_classes: Number of disease classes
        img_size: Input image size (width, height)
    
    Returns:
        Compiled Keras model
    """
    # Load pre-trained MobileNetV2 without top layers
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size[0], img_size[1], 3)
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Build the model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def prepare_data_generators(data_dir, img_size=(224, 224), batch_size=32):
    """
    Prepare training and validation data generators.
    
    Args:
        data_dir: Path to dataset directory
        img_size: Target image size
        batch_size: Batch size for training
    
    Returns:
        train_gen, val_gen, class_names
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 20% for validation
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Training generator
    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    val_gen = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_gen, val_gen


def train_model(data_dir, output_path='ml_model/plant_disease_model.h5', 
                epochs=20, batch_size=32, img_size=(224, 224)):
    """
    Train the plant disease detection model.
    
    Args:
        data_dir: Path to dataset directory (structure: data_dir/class_name/images)
        output_path: Where to save the trained model
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Input image size
    """
    if not HAS_TENSORFLOW:
        print("ERROR: TensorFlow is required to train the model.")
        print("Install with: pip install tensorflow")
        return None
    
    print("=" * 60)
    print("Plant Disease Detection Model Training")
    print("=" * 60)
    print(f"Dataset: {data_dir}")
    print(f"Output: {output_path}")
    print(f"Image size: {img_size}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print("=" * 60)
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"\nERROR: Dataset directory not found: {data_dir}")
        print("\nPlease organize your dataset like this:")
        print("dataset/")
        print("├── Apple___Apple_scab/")
        print("├── Apple___healthy/")
        print("├── Tomato___Early_blight/")
        print("└── ... (38 folders for 38 classes)")
        return None
    
    # Prepare data generators
    print("\nPreparing data generators...")
    train_gen, val_gen = prepare_data_generators(data_dir, img_size, batch_size)
    
    print(f"\nFound {train_gen.samples} training images")
    print(f"Found {val_gen.samples} validation images")
    print(f"Number of classes: {train_gen.num_classes}")
    
    # Show class indices
    print("\nClass indices:")
    for class_name, idx in sorted(train_gen.class_indices.items(), key=lambda x: x[1]):
        print(f"  {idx}: {class_name}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(train_gen.num_classes, img_size)
    model.summary()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            output_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Train the model
    print("\nStarting training...")
    print("-" * 60)
    
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Print final results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {output_path}")
    print(f"\nFinal Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
    
    # Save training history
    history_path = output_path.replace('.h5', '_history.npy')
    np.save(history_path, history.history)
    print(f"Training history saved to: {history_path}")
    
    return model


def setup_dataset_structure():
    """Show how to organize the dataset."""
    print("\n" + "=" * 60)
    print("Dataset Structure Required")
    print("=" * 60)
    print("""
Your PlantVillage dataset should be organized like this:

plantvillage_dataset/
├── train/
│   ├── Apple___Apple_scab/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── Apple___Black_rot/
│   ├── Apple___Cedar_apple_rust/
│   ├── Apple___healthy/
│   ├── Blueberry___healthy/
│   ├── Cherry___Powdery_mildew/
│   ├── Cherry___healthy/
│   ├── Corn___Cercospora_leaf_spot_Gray_leaf_spot/
│   ├── Corn___Common_rust/
│   ├── Corn___Northern_Leaf_Blight/
│   ├── Corn___healthy/
│   ├── Grape___Black_rot/
│   ├── Grape___Esca_(Black_Measles)/
│   ├── Grape___Leaf_blight_Isariopsis_Leaf_Spot/
│   ├── Grape___healthy/
│   ├── Orange___Haunglongbing_(Citrus_greening)/
│   ├── Peach___Bacterial_spot/
│   ├── Peach___healthy/
│   ├── Pepper,_bell___Bacterial_spot/
│   ├── Pepper,_bell___healthy/
│   ├── Potato___Early_blight/
│   ├── Potato___Late_blight/
│   ├── Potato___healthy/
│   ├── Raspberry___healthy/
│   ├── Soybean___healthy/
│   ├── Squash___Powdery_mildew/
│   ├── Strawberry___Leaf_scorch/
│   ├── Strawberry___healthy/
│   ├── Tomato___Bacterial_spot/
│   ├── Tomato___Early_blight/
│   ├── Tomato___Late_blight/
│   ├── Tomato___Leaf_Mold/
│   ├── Tomato___Septoria_leaf_spot/
│   ├── Tomato___Spider_mites_Two-spotted_spider_mite/
│   ├── Tomato___Target_Spot/
│   ├── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
│   ├── Tomato___Tomato_mosaic_virus/
│   └── Tomato___healthy/
└── (same structure for validation)

Or use a single folder with 80-20 train-val split.
""")


def download_dataset_instructions():
    """Show how to download the PlantVillage dataset."""
    print("\n" + "=" * 60)
    print("How to Download PlantVillage Dataset")
    print("=" * 60)
    print("""
Option 1: Kaggle (Recommended)
------------------------------
1. Go to: https://www.kaggle.com/datasets/emmarex/plantdisease
2. Download the dataset
3. Extract to your project folder

Option 2: Using Kaggle CLI
--------------------------
pip install kaggle
kaggle datasets download -d emmarex/plantdisease
unzip plantdisease.zip

Option 3: Hugging Face
----------------------
Search for "plant disease classification" on Hugging Face
Many pre-trained models are available there.

Note: The dataset has ~50,000 images across 38 classes.
      Make sure you have enough storage space (~2GB).
""")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Plant Disease Detection Model')
    parser.add_argument('--data_path', type=str, default='dataset',
                        help='Path to dataset directory')
    parser.add_argument('--output', type=str, default='ml_model/plant_disease_model.h5',
                        help='Output model path')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size (square)')
    parser.add_argument('--download', action='store_true',
                        help='Show download instructions')
    parser.add_argument('--structure', action='store_true',
                        help='Show required dataset structure')
    
    args = parser.parse_args()
    
    if args.download:
        download_dataset_instructions()
    elif args.structure:
        setup_dataset_structure()
    else:
        train_model(
            data_dir=args.data_path,
            output_path=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=(args.img_size, args.img_size)
        )
