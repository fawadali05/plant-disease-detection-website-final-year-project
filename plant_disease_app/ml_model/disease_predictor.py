"""
Plant Disease Detection CNN Model
================================
Ye file CNN model use karke plant disease detect karti hai.

Models Supported:
- MobileNetV2 (Fast, Lightweight) ✅ Recommended
- ResNet50 (Accurate, Heavy)
- VGG16 (Very Accurate, Very Heavy)

Dataset: PlantVillage - 38 Disease Classes
"""

import os
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Image size for CNN models
IMG_SIZE = (224, 224)

# PlantVillage Dataset - 38 Classes
CLASS_NAMES = [
    # Apple (4 classes)
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    
    # Blueberry (1 class)
    'Blueberry___healthy',
    
    # Cherry (2 classes)
    'Cherry___Powdery_mildew',
    'Cherry___healthy',
    
    # Corn (4 classes)
    'Corn___Cercospora_leaf_spot_Gray_leaf_spot',
    'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight',
    'Corn___healthy',
    
    # Grape (4 classes)
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_Isariopsis_Leaf_Spot',
    'Grape___healthy',
    
    # Orange (1 class)
    'Orange___Haunglongbing_(Citrus_greening)',
    
    # Peach (2 classes)
    'Peach___Bacterial_spot',
    'Peach___healthy',
    
    # Pepper Bell (2 classes)
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    
    # Potato (3 classes)
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    
    # Raspberry (1 class)
    'Raspberry___healthy',
    
    # Soybean (1 class)
    'Soybean___healthy',
    
    # Squash (1 class)
    'Squash___Powdery_mildew',
    
    # Strawberry (2 classes)
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    
    # Tomato (9 classes)
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites_Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]

# Disease Information Database
DISEASE_INFO = {
    'Apple___Apple_scab': {
        'disease': 'Apple Scab',
        'plant': 'Apple',
        'category': 'Fungal',
        'severity': 'Moderate',
        'treatment': 'Apply fungicides containing captan or myclobutanil. Remove infected leaves and fruit.',
        'prevention': 'Plant resistant varieties, maintain good air circulation, apply preventive fungicides in spring.',
        'pesticides': 'Captan, Myclobutanil, Sulfur-based fungicides'
    },
    'Apple___Black_rot': {
        'disease': 'Black Rot',
        'plant': 'Apple',
        'category': 'Fungal',
        'severity': 'High',
        'treatment': 'Prune infected branches, apply copper fungicides, remove mummified fruit.',
        'prevention': 'Avoid tree wounds, maintain tree health, proper pruning.',
        'pesticides': 'Copper fungicides, Thiophanate-methyl'
    },
    'Apple___Cedar_apple_rust': {
        'disease': 'Cedar Apple Rust',
        'plant': 'Apple',
        'category': 'Fungal',
        'severity': 'Moderate',
        'treatment': 'Apply fungicides at petal fall and continue through summer.',
        'prevention': 'Remove nearby cedar/juniper hosts, plant resistant varieties.',
        'pesticides': 'Myclobutanil, Propiconazole, Sulfur'
    },
    'Apple___healthy': {
        'disease': 'Healthy Apple',
        'plant': 'Apple',
        'category': 'Healthy',
        'severity': 'None',
        'treatment': 'Continue regular care and maintenance.',
        'prevention': 'Maintain proper soil pH, adequate watering, annual pruning.',
        'pesticides': 'None needed'
    },
    'Blueberry___healthy': {
        'disease': 'Healthy Blueberry',
        'plant': 'Blueberry',
        'category': 'Healthy',
        'severity': 'None',
        'treatment': 'Continue regular care and maintenance.',
        'prevention': 'Maintain proper soil pH (4.5-5.5), adequate watering.',
        'pesticides': 'None needed'
    },
    'Cherry___Powdery_mildew': {
        'disease': 'Powdery Mildew',
        'plant': 'Cherry',
        'category': 'Fungal',
        'severity': 'Moderate',
        'treatment': 'Apply sulfur-based or systemic fungicides. Improve air circulation.',
        'prevention': 'Plant resistant varieties, avoid overhead irrigation.',
        'pesticides': 'Sulfur, Potassium bicarbonate, Neem oil'
    },
    'Cherry___healthy': {
        'disease': 'Healthy Cherry',
        'plant': 'Cherry',
        'category': 'Healthy',
        'severity': 'None',
        'treatment': 'Continue regular maintenance.',
        'prevention': 'Proper pruning, adequate spacing, good drainage.',
        'pesticides': 'None needed'
    },
    'Corn___Cercospora_leaf_spot_Gray_leaf_spot': {
        'disease': 'Gray Leaf Spot',
        'plant': 'Corn',
        'category': 'Fungal',
        'severity': 'High',
        'treatment': 'Apply foliar fungicides at early tassel emergence.',
        'prevention': 'Rotate crops, till residues, plant resistant hybrids.',
        'pesticides': 'Strobilurin fungicides, Triazole fungicides'
    },
    'Corn___Common_rust': {
        'disease': 'Common Rust',
        'plant': 'Corn',
        'category': 'Fungal',
        'severity': 'Moderate',
        'treatment': 'Apply fungicides if severe, focus on protecting upper leaves.',
        'prevention': 'Plant early maturing varieties, resistant hybrids.',
        'pesticides': 'Mancozeb, Propiconazole, Azoxystrobin'
    },
    'Corn___Northern_Leaf_Blight': {
        'disease': 'Northern Leaf Blight',
        'plant': 'Corn',
        'category': 'Fungal',
        'severity': 'High',
        'treatment': 'Apply foliar fungicides at disease onset.',
        'prevention': 'Rotate crops, till residue, plant resistant hybrids.',
        'pesticides': 'Strobilurin + Triazole tank mix'
    },
    'Corn___healthy': {
        'disease': 'Healthy Corn',
        'plant': 'Corn',
        'category': 'Healthy',
        'severity': 'None',
        'treatment': 'Continue regular care.',
        'prevention': 'Balanced fertilization, adequate spacing, proper irrigation.',
        'pesticides': 'None needed'
    },
    'Grape___Black_rot': {
        'disease': 'Black Rot',
        'plant': 'Grape',
        'category': 'Fungal',
        'severity': 'High',
        'treatment': 'Remove infected plant parts, apply fungicides preventively.',
        'prevention': 'Good sanitation, proper canopy management, fungicide program.',
        'pesticides': 'Mancozeb, Captan, Myclobutanil'
    },
    'Grape___Esca_(Black_Measles)': {
        'disease': 'Esca (Black Measles)',
        'plant': 'Grape',
        'category': 'Fungal',
        'severity': 'Critical',
        'treatment': 'No effective chemical treatment. Remove severely infected vines.',
        'prevention': 'Avoid wounds during pruning, use wound protectants.',
        'pesticides': 'None effective - focus on prevention'
    },
    'Grape___Leaf_blight_Isariopsis_Leaf_Spot': {
        'disease': 'Leaf Blight',
        'plant': 'Grape',
        'category': 'Fungal',
        'severity': 'Moderate',
        'treatment': 'Apply fungicides, remove infected leaves.',
        'prevention': 'Improve air circulation, remove leaf litter.',
        'pesticides': 'Copper-based fungicides, Mancozeb'
    },
    'Grape___healthy': {
        'disease': 'Healthy Grape',
        'plant': 'Grape',
        'category': 'Healthy',
        'severity': 'None',
        'treatment': 'Continue regular maintenance.',
        'prevention': 'Proper pruning, good air circulation, balanced nutrition.',
        'pesticides': 'None needed'
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'disease': 'Citrus Greening (HLB)',
        'plant': 'Orange',
        'category': 'Bacterial',
        'severity': 'Critical',
        'treatment': 'No cure - remove infected trees, control psyllid vector.',
        'prevention': 'Use certified disease-free plants, control insect vectors.',
        'pesticides': 'Imidacloprid, Cyantraniliprole for psyllid control'
    },
    'Peach___Bacterial_spot': {
        'disease': 'Bacterial Spot',
        'plant': 'Peach',
        'category': 'Bacterial',
        'severity': 'Moderate',
        'treatment': 'Apply copper bactericides, remove infected plant parts.',
        'prevention': 'Plant resistant varieties, avoid overhead irrigation.',
        'pesticides': 'Copper-based bactericides, Streptomycin'
    },
    'Peach___healthy': {
        'disease': 'Healthy Peach',
        'plant': 'Peach',
        'category': 'Healthy',
        'severity': 'None',
        'treatment': 'Continue regular care.',
        'prevention': 'Proper pruning, balanced fertilization, adequate watering.',
        'pesticides': 'None needed'
    },
    'Pepper,_bell___Bacterial_spot': {
        'disease': 'Bacterial Spot',
        'plant': 'Pepper',
        'category': 'Bacterial',
        'severity': 'Moderate',
        'treatment': 'Apply copper bactericides, improve plant spacing.',
        'prevention': 'Use disease-free seeds, rotate crops, avoid overhead irrigation.',
        'pesticides': 'Copper bactericides, Streptomycin'
    },
    'Pepper,_bell___healthy': {
        'disease': 'Healthy Pepper',
        'plant': 'Pepper',
        'category': 'Healthy',
        'severity': 'None',
        'treatment': 'Continue regular care.',
        'prevention': 'Adequate spacing, proper staking, consistent watering.',
        'pesticides': 'None needed'
    },
    'Potato___Early_blight': {
        'disease': 'Early Blight',
        'plant': 'Potato',
        'category': 'Fungal',
        'severity': 'Moderate',
        'treatment': 'Apply foliar fungicides, remove infected foliage.',
        'prevention': 'Rotate crops (2-3 years), plant certified seed.',
        'pesticides': 'Chlorothalonil, Mancozeb, Azoxystrobin'
    },
    'Potato___Late_blight': {
        'disease': 'Late Blight',
        'plant': 'Potato',
        'category': 'Fungal',
        'severity': 'Critical',
        'treatment': 'Apply systemic fungicides immediately, destroy infected plants.',
        'prevention': 'Plant resistant varieties, avoid overhead irrigation.',
        'pesticides': 'Mancozeb + Cymoxanil, Metalaxyl, Fluopicolide'
    },
    'Potato___healthy': {
        'disease': 'Healthy Potato',
        'plant': 'Potato',
        'category': 'Healthy',
        'severity': 'None',
        'treatment': 'Continue regular care.',
        'prevention': 'Crop rotation, proper hilling, adequate drainage.',
        'pesticides': 'None needed'
    },
    'Raspberry___healthy': {
        'disease': 'Healthy Raspberry',
        'plant': 'Raspberry',
        'category': 'Healthy',
        'severity': 'None',
        'treatment': 'Continue regular care.',
        'prevention': 'Annual pruning, good air circulation, remove old canes.',
        'pesticides': 'None needed'
    },
    'Soybean___healthy': {
        'disease': 'Healthy Soybean',
        'plant': 'Soybean',
        'category': 'Healthy',
        'severity': 'None',
        'treatment': 'Continue regular care.',
        'prevention': 'Crop rotation, balanced fertilization, proper plant density.',
        'pesticides': 'None needed'
    },
    'Squash___Powdery_mildew': {
        'disease': 'Powdery Mildew',
        'plant': 'Squash',
        'category': 'Fungal',
        'severity': 'Moderate',
        'treatment': 'Apply sulfur-based or organic fungicides.',
        'prevention': 'Plant resistant varieties, improve air circulation.',
        'pesticides': 'Sulfur, Neem oil, Potassium bicarbonate'
    },
    'Strawberry___Leaf_scorch': {
        'disease': 'Leaf Scorch',
        'plant': 'Strawberry',
        'category': 'Fungal',
        'severity': 'Moderate',
        'treatment': 'Remove infected leaves, apply fungicides.',
        'prevention': 'Use resistant varieties, remove plant debris.',
        'pesticides': 'Captan, Pyaclostrobin, Myclobutanil'
    },
    'Strawberry___healthy': {
        'disease': 'Healthy Strawberry',
        'plant': 'Strawberry',
        'category': 'Healthy',
        'severity': 'None',
        'treatment': 'Continue regular care.',
        'prevention': 'Mulching, proper watering, regular feeding.',
        'pesticides': 'None needed'
    },
    'Tomato___Bacterial_spot': {
        'disease': 'Bacterial Spot',
        'plant': 'Tomato',
        'category': 'Bacterial',
        'severity': 'Moderate',
        'treatment': 'Apply copper bactericides, remove infected plants.',
        'prevention': 'Use disease-free seeds, rotate crops, avoid overhead irrigation.',
        'pesticides': 'Copper bactericides (copper hydroxide, copper sulfate)'
    },
    'Tomato___Early_blight': {
        'disease': 'Early Blight',
        'plant': 'Tomato',
        'category': 'Fungal',
        'severity': 'Moderate',
        'treatment': 'Apply fungicides, remove infected leaves, improve air circulation.',
        'prevention': 'Rotate crops (2-3 years), stake plants, mulch.',
        'pesticides': 'Chlorothalonil, Mancozeb, Azoxystrobin'
    },
    'Tomato___Late_blight': {
        'disease': 'Late Blight',
        'plant': 'Tomato',
        'category': 'Fungal',
        'severity': 'Critical',
        'treatment': 'Remove infected plants immediately, apply systemic fungicides.',
        'prevention': 'Plant certified disease-free transplants, avoid overhead irrigation.',
        'pesticides': 'Mancozeb + Cymoxanil, Metalaxyl-m'
    },
    'Tomato___Leaf_Mold': {
        'disease': 'Leaf Mold',
        'plant': 'Tomato',
        'category': 'Fungal',
        'severity': 'Moderate',
        'treatment': 'Apply fungicides, reduce humidity, improve air circulation.',
        'prevention': 'Stake plants, prune lower leaves, maintain spacing.',
        'pesticides': 'Chlorothalonil, Copper fungicides'
    },
    'Tomato___Septoria_leaf_spot': {
        'disease': 'Septoria Leaf Spot',
        'plant': 'Tomato',
        'category': 'Fungal',
        'severity': 'Moderate',
        'treatment': 'Apply fungicides, remove infected leaves, improve air circulation.',
        'prevention': 'Rotate crops, avoid overhead irrigation, remove plant debris.',
        'pesticides': 'Chlorothalonil, Mancozeb, Copper fungicides'
    },
    'Tomato___Spider_mites_Two-spotted_spider_mite': {
        'disease': 'Spider Mites',
        'plant': 'Tomato',
        'category': 'Pest',
        'severity': 'Moderate',
        'treatment': 'Apply miticides, spray with water to dislodge mites.',
        'prevention': 'Maintain humidity, avoid excessive nitrogen.',
        'pesticides': 'Spinosad, Abamectin, Bifenazate, Horticultural oil'
    },
    'Tomato___Target_Spot': {
        'disease': 'Target Spot',
        'plant': 'Tomato',
        'category': 'Fungal',
        'severity': 'Moderate',
        'treatment': 'Apply fungicides, remove infected plant parts.',
        'prevention': 'Rotate crops, plant resistant varieties, proper spacing.',
        'pesticides': 'Azoxystrobin, Pyraclostrobin, Difenoconazole'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'disease': 'Yellow Leaf Curl Virus',
        'plant': 'Tomato',
        'category': 'Viral',
        'severity': 'Critical',
        'treatment': 'No cure - remove infected plants, control whitefly vector.',
        'prevention': 'Plant resistant varieties, use yellow sticky traps.',
        'pesticides': 'Imidacloprid, Thiamethoxam for whitefly control'
    },
    'Tomato___Tomato_mosaic_virus': {
        'disease': 'Tomato Mosaic Virus',
        'plant': 'Tomato',
        'category': 'Viral',
        'severity': 'High',
        'treatment': 'No cure - remove and destroy infected plants.',
        'prevention': 'Use virus-resistant varieties, disinfect tools.',
        'pesticides': 'None effective - focus on sanitation'
    },
    'Tomato___healthy': {
        'disease': 'Healthy Tomato',
        'plant': 'Tomato',
        'category': 'Healthy',
        'severity': 'None',
        'treatment': 'Continue regular care.',
        'prevention': 'Proper staking, consistent watering, balanced fertilization.',
        'pesticides': 'None needed'
    },
}


class PlantDiseaseClassifier:
    """
    Plant Disease Classification using CNN
    =====================================
    
    Ye class two modes mein kaam karti hai:
    
    1. DEMO MODE: Random predictions (model file ke bina)
       - Testing ke liye use karo
       
    2. TRAINED MODE: Real CNN predictions (model file ke sath)
       - Production ke liye use karo
    
    """
    
    def __init__(self, model_path=None, use_pretrained=True):
        """
        Initialize the classifier.
        
        Args:
            model_path: Path to trained .h5 model file
            use_pretrained: Use pre-trained ImageNet weights if no model
        """
        self.model = None
        self.model_path = model_path
        self.num_classes = len(CLASS_NAMES)
        
        # Try to load TensorFlow/Keras
        self.tensorflow_available = False
        try:
            import tensorflow as tf
            from tensorflow.keras.models import load_model
            from tensorflow.keras.preprocessing import image as keras_image
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            self.tensorflow_available = True
            print("✅ TensorFlow loaded successfully!")
        except ImportError:
            print("⚠️ TensorFlow not installed. Using DEMO mode.")
            print("   Install with: pip install tensorflow")
        
        if self.tensorflow_available:
            if model_path and os.path.exists(model_path):
                self.load_model(model_path)
            else:
                self.create_model(use_pretrained=use_pretrained)
    
    def create_model(self, use_pretrained=True):
        """Create CNN model architecture."""
        if not self.tensorflow_available:
            print("⚠️ Cannot create model without TensorFlow")
            return
        
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
        from tensorflow.keras.applications import MobileNetV2
        
        print("Creating CNN model with MobileNetV2...")
        
        if use_pretrained:
            # Use pre-trained MobileNetV2
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
            )
            # Freeze base model layers
            for layer in base_model.layers:
                layer.trainable = False
        else:
            base_model = MobileNetV2(
                weights=None,
                include_top=False,
                input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
            )
        
        # Build model
        self.model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Model created with {self.num_classes} output classes")
    
    def load_model(self, model_path):
        """Load trained model from file."""
        if not self.tensorflow_available:
            return
        
        try:
            from tensorflow.keras.models import load_model
            self.model = load_model(model_path)
            print(f"✅ Model loaded from: {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.create_model()
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for prediction.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed numpy array
        """
        try:
            # Load image
            img = Image.open(image_path)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize
            img = img.resize(IMG_SIZE)
            
            # Convert to array
            img_array = np.array(img, dtype=np.float32)
            
            # Expand dimensions (batch size = 1)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Preprocess for MobileNetV2
            img_array = img_array / 255.0  # Normalize to [0, 1]
            
            return img_array
            
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {e}")
    
    def predict(self, image_path, top_k=5):
        """
        Predict plant disease from image.
        
        Args:
            image_path: Path to image file
            top_k: Return top k predictions
            
        Returns:
            List of dictionaries with predictions
        """
        if self.tensorflow_available and self.model is not None:
            # Real prediction using CNN
            img_array = self.preprocess_image(image_path)
            predictions = self.model.predict(img_array, verbose=0)[0]
            
            # Get top k predictions
            top_indices = np.argsort(predictions)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                class_name = CLASS_NAMES[idx]
                confidence = float(predictions[idx]) * 100
                info = DISEASE_INFO.get(class_name, {})
                
                results.append({
                    'class_name': class_name,
                    'disease_name': info.get('disease', class_name),
                    'plant_type': info.get('plant', 'Unknown'),
                    'category': info.get('category', 'Unknown'),
                    'confidence': confidence,
                    'severity': info.get('severity', 'Unknown'),
                    'treatment': info.get('treatment', 'No information available'),
                    'prevention': info.get('prevention', 'No information available'),
                    'pesticides': info.get('pesticides', 'Consult local agricultural expert')
                })
            
            return results
        else:
            # Demo mode - return random predictions
            return self._demo_predict(top_k)
    
    def _demo_predict(self, top_k=5):
        """
        Demo prediction (when TensorFlow is not available).
        This is for testing the UI without a trained model.
        """
        print("⚠️ Running in DEMO mode - predictions are random")
        
        # Random predictions for demo
        np.random.seed(int.from_bytes(os.urandom(4), 'little') % 1000)
        random_probs = np.random.dirichlet(np.ones(self.num_classes))
        top_indices = np.argsort(random_probs)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            class_name = CLASS_NAMES[idx]
            confidence = float(random_probs[idx]) * 100
            info = DISEASE_INFO.get(class_name, {})
            
            results.append({
                'class_name': class_name,
                'disease_name': info.get('disease', class_name),
                'plant_type': info.get('plant', 'Unknown'),
                'category': info.get('category', 'Unknown'),
                'confidence': confidence,
                'severity': info.get('severity', 'Unknown'),
                'treatment': info.get('treatment', 'No information available'),
                'prevention': info.get('prevention', 'No information available'),
                'pesticides': info.get('pesticides', 'Consult local agricultural expert')
            })
        
        return results
    
    def get_disease_info(self, class_name):
        """Get full disease information."""
        return DISEASE_INFO.get(class_name, {})
    
    def get_supported_plants(self):
        """Get list of supported plant types."""
        plants = set()
        for info in DISEASE_INFO.values():
            plants.add(info.get('plant', 'Unknown'))
        return sorted(list(plants))
    
    def get_all_classes(self):
        """Get all supported disease classes."""
        return CLASS_NAMES


# ================================
# SINGLETON INSTANCE
# ================================
_classifier = None


def get_classifier():
    """Get or create the classifier singleton."""
    global _classifier
    if _classifier is None:
        model_path = os.environ.get('MODEL_PATH', 'ml_model/plant_disease_model.h5')
        _classifier = PlantDiseaseClassifier(model_path=model_path)
    return _classifier


# ================================
# CONVENIENCE FUNCTION
# ================================
def predict_disease(image_path, top_k=5):
    """
    Convenience function to predict disease.
    
    Args:
        image_path: Path to image
        top_k: Number of predictions
        
    Returns:
        List of predictions
    """
    classifier = get_classifier()
    return classifier.predict(image_path, top_k=top_k)


# ================================
# TEST FUNCTION
# ================================
if __name__ == '__main__':
    print("=" * 60)
    print("Plant Disease Detection - CNN Model")
    print("=" * 60)
    print(f"\nSupported Classes: {len(CLASS_NAMES)}")
    print(f"Supported Plants: {len(set(DISEASE_INFO.values()))}")
    
    # List supported plants
    plants = set()
    for info in DISEASE_INFO.values():
        plants.add(info.get('plant', 'Unknown'))
    
    print(f"\nPlants: {', '.join(sorted(plants))}")
    print("\n" + "=" * 60)
    print("USAGE INSTRUCTIONS")
    print("=" * 60)
    print("""
1. For DEMO mode (no model file):
   classifier = PlantDiseaseClassifier()
   
2. For TRAINED mode (with model file):
   classifier = PlantDiseaseClassifier(model_path='plant_disease_model.h5')
   
3. To predict:
   results = classifier.predict('leaf_image.jpg', top_k=5)
   print(results)
    """)
