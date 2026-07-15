"""
CNN Model Training Script
=========================
Ye script aapke PlantVillage dataset se CNN model train karega.

Dataset Download: https://www.kaggle.com/datasets/emmarex/plantdisease

Usage:
    python train_cnn.py --data_path ./dataset --epochs 20
"""

import os
import sys
import argparse
from pathlib import Path

# Check TensorFlow
try:
    import tensorflow as tf
    print(f"TensorFlow Version: {tf.__version__}")
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.applications import MobileNetV2
    print("✅ TensorFlow loaded successfully!")
except ImportError:
    print("❌ TensorFlow is required!")
    print("   Install: pip install tensorflow")
    sys.exit(1)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


# Dataset Classes (38 classes)
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy',
    'Corn___Cercospora_leaf_spot_Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight',
    'Corn___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_Isariopsis_Leaf_Spot', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites_Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]


def create_model(num_classes, img_size=224):
    """
    Create CNN model using MobileNetV2 transfer learning.
    
    MobileNetV2 is recommended because:
    - Fast inference
    - Good accuracy
    - Low memory usage
    - Pre-trained on ImageNet
    """
    print(f"\n📱 Creating MobileNetV2 model for {num_classes} classes...")
    
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet',  # Use ImageNet pre-trained weights
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    
    # Freeze base model layers first (train only new layers)
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
    
    # Compile
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Model created successfully!")
    return model


def create_data_generators(data_dir, img_size=224, batch_size=32):
    """
    Create training and validation data generators with augmentation.
    
    Data augmentation helps:
    - Reduce overfitting
    - Improve generalization
    - Better accuracy on real images
    """
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Validation data generator (only rescaling)
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator


def train_model(data_dir, output_path='ml_model/plant_disease_model.h5', 
                epochs=20, batch_size=32, img_size=224, learning_rate=0.001):
    """
    Train the plant disease detection model.
    
    Args:
        data_dir: Path to dataset directory
        output_path: Where to save trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
        learning_rate: Learning rate
    """
    print("=" * 60)
    print("🌿 PLANT DISEASE CNN MODEL TRAINING")
    print("=" * 60)
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"\n❌ Dataset not found: {data_dir}")
        print("\n📁 Please organize your dataset like this:")
        print("""
dataset/
├── Apple___Apple_scab/
│   ├── image1.jpg
│   └── ...
├── Apple___Black_rot/
├── Tomato___Early_blight/
└── ... (38 folders)
        """)
        return None
    
    # Count classes
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    print(f"\n📊 Found {len(folders)} folders in dataset")
    print(f"📁 Dataset: {data_dir}")
    
    # Create data generators
    print("\n📸 Creating data generators...")
    train_gen, val_gen = create_data_generators(data_dir, img_size, batch_size)
    
    print(f"\n✅ Training samples: {train_gen.samples}")
    print(f"✅ Validation samples: {val_gen.samples}")
    print(f"✅ Number of classes: {train_gen.num_classes}")
    
    # Show class mapping
    print("\n📋 Class mapping:")
    for class_name, idx in sorted(train_gen.class_indices.items(), key=lambda x: x[1]):
        print(f"   {idx}: {class_name}")
    
    # Create model
    model = create_model(train_gen.num_classes, img_size)
    
    # Print model summary
    print("\n📐 Model Architecture:")
    print(f"   Base: MobileNetV2 (pre-trained on ImageNet)")
    print(f"   Input: {img_size}x{img_size}x3")
    print(f"   Output: {train_gen.num_classes} classes")
    
    # Callbacks
    callbacks = [
        # Save best model
        ModelCheckpoint(
            output_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Stop early if no improvement
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate if stuck
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Print training info
    print("\n" + "=" * 60)
    print("📈 TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"   Epochs: {epochs}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Image Size: {img_size}x{img_size}")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Training Samples: {train_gen.samples}")
    print(f"   Steps per Epoch: {train_gen.samples // batch_size}")
    print("=" * 60)
    
    # Train the model
    print("\n🚀 Starting training...")
    print("   (This may take 30-60 minutes depending on your hardware)\n")
    
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 TRAINING RESULTS")
    print("=" * 60)
    print(f"   Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"   Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"   Final Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"   Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
    print("=" * 60)
    
    # Save the model
    model.save(output_path)
    print(f"\n✅ Model saved to: {output_path}")
    
    # Print tips
    print("\n" + "=" * 60)
    print("💡 NEXT STEPS")
    print("=" * 60)
    print("""
1. Copy the trained model to ml_model folder:
   copy plant_disease_model.h5 ml_model/

2. Update MODEL_PATH in .env file:
   MODEL_PATH=ml_model/plant_disease_model.h5

3. Run the Flask app:
   python run.py

4. Test the detection with a plant leaf image!
    """)
    
    return model


def download_dataset_guide():
    """Show how to download the PlantVillage dataset."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║           📥 HOW TO DOWNLOAD PLANTVILLAGE DATASET            ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  OPTION 1: KAGGLE (Recommended)                              ║
║  ────────────────────────────────                            ║
║  1. Go to: https://www.kaggle.com/datasets/emmarex/         ║
║             plantdisease                                     ║
║  2. Click "Download" button                                  ║
║  3. Extract the ZIP file                                     ║
║  4. Folder structure should have 38 class folders           ║
║                                                              ║
║  OPTION 2: KAGGLE CLI                                       ║
║  ──────────────────────────                                  ║
║  pip install kaggle                                         ║
║  kaggle datasets download -d emmarex/plantdisease           ║
║  unzip plantdisease.zip                                      ║
║                                                              ║
║  OPTION 3: HUGGING FACE (Pre-trained Models)                ║
║  ───────────────────────────────────────────                 ║
║  1. Go to: https://huggingface.co/models                    ║
║  2. Search: "plant disease classification"                    ║
║  3. Download a pre-trained model                            ║
║                                                              ║
║  📁 EXPECTED FOLDER STRUCTURE:                              ║
║  ─────────────────────────────────                          ║
║  plantvillage/                                              ║
║  ├── Apple___Apple_scab/                                    ║
║  ├── Apple___Black_rot/                                     ║
║  ├── Apple___healthy/                                       ║
║  ├── Tomato___Early_blight/                                 ║
║  └── ... (38 folders total)                                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


def show_dataset_structure():
    """Show required dataset structure."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║              📁 REQUIRED DATASET STRUCTURE                   ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Your dataset folder MUST have these 38 folders:            ║
║                                                              ║
║  📂 Apple (4)                                               ║
║     ├── Apple___Apple_scab                                  ║
║     ├── Apple___Black_rot                                   ║
║     ├── Apple___Cedar_apple_rust                            ║
║     └── Apple___healthy                                     ║
║                                                              ║
║  📂 Blueberry (1)                                            ║
║     └── Blueberry___healthy                                 ║
║                                                              ║
║  📂 Cherry (2)                                              ║
║     ├── Cherry___Powdery_mildew                             ║
║     └── Cherry___healthy                                    ║
║                                                              ║
║  📂 Corn (4)                                                ║
║     ├── Corn___Cercospora_leaf_spot_Gray_leaf_spot          ║
║     ├── Corn___Common_rust                                  ║
║     ├── Corn___Northern_Leaf_Blight                         ║
║     └── Corn___healthy                                      ║
║                                                              ║
║  📂 Grape (4)                                               ║
║     ├── Grape___Black_rot                                   ║
║     ├── Grape___Esca_(Black_Measles)                        ║
║     ├── Grape___Leaf_blight_Isariopsis_Leaf_Spot           ║
║     └── Grape___healthy                                     ║
║                                                              ║
║  📂 Orange (1)                                              ║
║     └── Orange___Haunglongbing_(Citrus_greening)           ║
║                                                              ║
║  📂 Peach (2)                                               ║
║     ├── Peach___Bacterial_spot                             ║
║     └── Peach___healthy                                     ║
║                                                              ║
║  📂 Pepper (2)                                              ║
║     ├── Pepper,_bell___Bacterial_spot                       ║
║     └── Pepper,_bell___healthy                             ║
║                                                              ║
║  📂 Potato (3)                                              ║
║     ├── Potato___Early_blight                               ║
║     ├── Potato___Late_blight                                ║
║     └── Potato___healthy                                    ║
║                                                              ║
║  📂 Raspberry (1)                                           ║
║     └── Raspberry___healthy                                 ║
║                                                              ║
║  📂 Soybean (1)                                             ║
║     └── Soybean___healthy                                   ║
║                                                              ║
║  📂 Squash (1)                                              ║
║     └── Squash___Powdery_mildew                             ║
║                                                              ║
║  📂 Strawberry (2)                                          ║
║     ├── Strawberry___Leaf_scorch                            ║
║     └── Strawberry___healthy                               ║
║                                                              ║
║  📂 Tomato (9)                                              ║
║     ├── Tomato___Bacterial_spot                             ║
║     ├── Tomato___Early_blight                               ║
║     ├── Tomato___Late_blight                                ║
║     ├── Tomato___Leaf_Mold                                  ║
║     ├── Tomato___Septoria_leaf_spot                        ║
║     ├── Tomato___Spider_mites_Two-spotted_spider_mite      ║
║     ├── Tomato___Target_Spot                                ║
║     ├── Tomato___Tomato_Yellow_Leaf_Curl_Virus             ║
║     ├── Tomato___Tomato_mosaic_virus                        ║
║     └── Tomato___healthy                                    ║
║                                                              ║
║  📊 TOTAL: 38 Folders, ~50,000 Images                       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


# ================================
# MAIN EXECUTION
# ================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Plant Disease CNN Model')
    parser.add_argument('--data_path', type=str, default='dataset',
                        help='Path to dataset directory')
    parser.add_argument('--output', type=str, default='ml_model/plant_disease_model.h5',
                        help='Output model path')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size (default: 224)')
    parser.add_argument('--download', action='store_true',
                        help='Show download instructions')
    parser.add_argument('--structure', action='store_true',
                        help='Show required dataset structure')
    
    args = parser.parse_args()
    
    if args.download:
        download_dataset_guide()
    elif args.structure:
        show_dataset_structure()
    else:
        train_model(
            data_dir=args.data_path,
            output_path=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size
        )
