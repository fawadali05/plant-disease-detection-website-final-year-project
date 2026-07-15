"""
Database Models Package
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """User model for authentication and tracking detections."""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), default='user')  # 'user' or 'admin'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    detections = db.relationship('Detection', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set the user's password."""
        from flask_bcrypt import generate_password_hash
        self.password_hash = generate_password_hash(password).decode('utf-8')
    
    def check_password(self, password):
        """Verify the user's password."""
        from flask_bcrypt import check_password_hash
        return check_password_hash(self.password_hash, password)
    
    def get_detection_count(self):
        """Get total number of detections by this user."""
        return self.detections.count()
    
    def __repr__(self):
        return f'<User {self.email}>'


class Detection(db.Model):
    """Model for storing plant disease detection records."""
    __tablename__ = 'detections'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    image_path = db.Column(db.String(500), nullable=False)
    disease_name = db.Column(db.String(100), nullable=False)
    plant_type = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    detected_at = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default='completed')
    
    def __repr__(self):
        return f'<Detection {self.id}: {self.disease_name}>'


class Disease(db.Model):
    """Model for disease information and treatment database."""
    __tablename__ = 'diseases'
    
    id = db.Column(db.Integer, primary_key=True)
    disease_name = db.Column(db.String(100), unique=True, nullable=False, index=True)
    plant_type = db.Column(db.String(50), nullable=False)
    category = db.Column(db.String(50))
    description = db.Column(db.Text, nullable=False)
    symptoms = db.Column(db.Text)
    causes = db.Column(db.Text)
    treatment = db.Column(db.Text, nullable=False)
    prevention = db.Column(db.Text)
    pesticides = db.Column(db.Text)
    recovery_time = db.Column(db.String(100))
    severity = db.Column(db.String(20), default='moderate')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<Disease {self.disease_name}>'


class AdminLog(db.Model):
    """Model for tracking admin actions."""
    __tablename__ = 'admin_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    admin_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    action = db.Column(db.String(100), nullable=False)
    target = db.Column(db.String(100))
    details = db.Column(db.Text)
    ip_address = db.Column(db.String(45))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    admin = db.relationship('User', backref='admin_logs')
    
    def __repr__(self):
        return f'<AdminLog {self.action}>'


class ContactMessage(db.Model):
    """Model for storing contact form messages."""
    __tablename__ = 'contact_messages'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    subject = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<ContactMessage {self.subject}>'


def init_db(app):
    """Initialize database with app context."""
    with app.app_context():
        db.create_all()
        
        # Create admin user if not exists
        from config import Config
        admin = User.query.filter_by(email=Config.ADMIN_EMAIL).first()
        if not admin:
            admin = User(
                name='Administrator',
                email=Config.ADMIN_EMAIL,
                role='admin'
            )
            admin.set_password(Config.ADMIN_PASSWORD)
            db.session.add(admin)
            db.session.commit()
            print(f"Admin user created: {Config.ADMIN_EMAIL}")
        
        # Seed disease data if empty
        if Disease.query.count() == 0:
            seed_diseases()
            print("Disease database seeded")


def seed_diseases():
    """Seed initial disease data from PlantVillage dataset classes."""
    diseases = [
        {'disease_name': 'Apple_Apple_scab', 'plant_type': 'Apple', 'category': 'fungal',
         'description': 'A fungal disease affecting apple trees, causing scab-like lesions on leaves and fruit.',
         'symptoms': 'Olive-green to brown spots on leaves, distorted leaves, dark lesions on fruit',
         'causes': 'Venturia inaequalis fungus, spread by rain splash',
         'treatment': 'Apply fungicides containing captan or myclobutanil. Remove infected leaves.',
         'prevention': 'Plant resistant varieties, maintain good air circulation, apply preventive fungicides',
         'pesticides': 'Captan, Myclobutanil, Sulfur-based fungicides', 'severity': 'moderate'},
        
        {'disease_name': 'Apple_Black_rot', 'plant_type': 'Apple', 'category': 'fungal',
         'description': 'A serious fungal disease causing leaf spots, fruit rot, and canker on apple trees.',
         'symptoms': 'Purple-ringed leaf spots, fruit rot with concentric rings, bark cankers',
         'causes': 'Botryosphaeria obtusa fungus, enters through wounds',
         'treatment': 'Prune infected branches, apply fungicides, remove mummified fruit',
         'prevention': 'Avoid tree wounds, maintain tree health, proper pruning',
         'pesticides': 'Copper fungicides, Thiophanate-methyl', 'severity': 'high'},
        
        {'disease_name': 'Apple_Cedar_apple_rust', 'plant_type': 'Apple', 'category': 'fungal',
         'description': 'Fungal disease requiring both apple and cedar hosts to complete its life cycle.',
         'symptoms': 'Bright orange-yellow spots on leaves, tube-like structures on cedar',
         'causes': 'Gymnosporangium juniperi-virginianae fungus',
         'treatment': 'Apply fungicides at petal fall and continue through summer',
         'prevention': 'Remove nearby cedar hosts, plant resistant varieties',
         'pesticides': 'Myclobutanil, Propiconazole, Sulfur', 'severity': 'moderate'},
        
        {'disease_name': 'Apple_healthy', 'plant_type': 'Apple', 'category': 'healthy',
         'description': 'Healthy apple plant with no signs of disease.',
         'symptoms': 'None - vibrant green leaves, normal growth',
         'causes': 'N/A',
         'treatment': 'Continue regular care and maintenance',
         'prevention': 'Maintain proper soil pH, adequate watering, annual pruning',
         'pesticides': 'None needed', 'severity': 'low'},
        
        {'disease_name': 'Blueberry_healthy', 'plant_type': 'Blueberry', 'category': 'healthy',
         'description': 'Healthy blueberry plant with no disease symptoms.',
         'symptoms': 'None',
         'causes': 'N/A',
         'treatment': 'Continue regular care and maintenance',
         'prevention': 'Maintain proper soil pH (4.5-5.5), adequate watering',
         'pesticides': 'None needed', 'severity': 'low'},
        
        {'disease_name': 'Cherry_Powdery_mildew', 'plant_type': 'Cherry', 'category': 'fungal',
         'description': 'Fungal disease creating white powdery coating on leaves and fruit.',
         'symptoms': 'White powdery coating on leaves, distorted growth',
         'causes': 'Podosphaera clandestina fungus',
         'treatment': 'Apply sulfur-based or systemic fungicides',
         'prevention': 'Improve air circulation, avoid overhead irrigation',
         'pesticides': 'Sulfur, Potassium bicarbonate, Neem oil', 'severity': 'moderate'},
        
        {'disease_name': 'Cherry_healthy', 'plant_type': 'Cherry', 'category': 'healthy',
         'description': 'Healthy cherry plant with no disease symptoms.',
         'symptoms': 'None',
         'causes': 'N/A',
         'treatment': 'Continue regular maintenance',
         'prevention': 'Proper pruning, adequate spacing, good drainage',
         'pesticides': 'None needed', 'severity': 'low'},
        
        {'disease_name': 'Corn_Cercospora_leaf_spot', 'plant_type': 'Corn', 'category': 'fungal',
         'description': 'Gray leaf spot causing elongated gray lesions on corn leaves.',
         'symptoms': 'Rectangular lesions parallel to leaf veins, gray-brown color',
         'causes': 'Cercospora zeae-maydis fungus, spread by rain splash',
         'treatment': 'Apply foliar fungicides at early tassel emergence',
         'prevention': 'Rotate crops, till residues, plant resistant hybrids',
         'pesticides': 'Strobilurin fungicides, Triazole fungicides', 'severity': 'high'},
        
        {'disease_name': 'Corn_Common_rust', 'plant_type': 'Corn', 'category': 'fungal',
         'description': 'Fungal disease causing reddish-brown pustules on corn leaves.',
         'symptoms': 'Small, round reddish-brown pustules on both leaf surfaces',
         'causes': 'Puccinia sorghi fungus, prefers cool moist conditions',
         'treatment': 'Apply fungicides if severe, focus on protecting upper leaves',
         'prevention': 'Plant early maturing varieties, resistant hybrids',
         'pesticides': 'Mancozeb, Propiconazole, Azoxystrobin', 'severity': 'moderate'},
        
        {'disease_name': 'Corn_Northern_Leaf_Blight', 'plant_type': 'Corn', 'category': 'fungal',
         'description': 'Northern corn leaf blight causes large, elliptical gray-green lesions.',
         'symptoms': 'Large, cigar-shaped gray-green lesions, extensive tissue death',
         'causes': 'Setosphaeria turcica fungus, overwinters in crop residue',
         'treatment': 'Apply foliar fungicides at disease onset',
         'prevention': 'Rotate crops, till residue, plant resistant hybrids',
         'pesticides': 'Strobilurin + Triazole tank mix', 'severity': 'high'},
        
        {'disease_name': 'Corn_healthy', 'plant_type': 'Corn', 'category': 'healthy',
         'description': 'Healthy corn plant with no disease symptoms.',
         'symptoms': 'None',
         'causes': 'N/A',
         'treatment': 'Continue regular care',
         'prevention': 'Balanced fertilization, adequate spacing, proper irrigation',
         'pesticides': 'None needed', 'severity': 'low'},
        
        {'disease_name': 'Grape_Black_rot', 'plant_type': 'Grape', 'category': 'fungal',
         'description': 'Destructive fungal disease affecting leaves, shoots, and fruit of grapes.',
         'symptoms': 'Brown circular lesions with black dots, fruit turning black',
         'causes': 'Guignardia bidwellii fungus, overwinters in infected debris',
         'treatment': 'Remove infected plant parts, apply fungicides preventively',
         'prevention': 'Good sanitation, proper canopy management',
         'pesticides': 'Mancozeb, Captan, Myclobutanil', 'severity': 'high'},
        
        {'disease_name': 'Grape_Esca', 'plant_type': 'Grape', 'category': 'fungal',
         'description': 'Trunk disease causing tiger-striped leaf pattern.',
         'symptoms': 'Tiger-stripe leaf pattern, white chalky streaks in wood',
         'causes': 'Complex of fungi including Phaeomoniella',
         'treatment': 'No effective chemical treatment, remove severely infected vines',
         'prevention': 'Avoid wounds during pruning, use wound protectants',
         'pesticides': 'None effective - focus on prevention', 'severity': 'critical'},
        
        {'disease_name': 'Grape_Leaf_blight', 'plant_type': 'Grape', 'category': 'fungal',
         'description': 'Leaf blight causing angular brown lesions on grape leaves.',
         'symptoms': 'Angular brown lesions, yellow halos, premature leaf drop',
         'causes': 'Phloeospora apiitaria fungus',
         'treatment': 'Apply fungicides, remove infected leaves',
         'prevention': 'Improve air circulation, remove leaf litter',
         'pesticides': 'Copper-based fungicides, Mancozeb', 'severity': 'moderate'},
        
        {'disease_name': 'Grape_healthy', 'plant_type': 'Grape', 'category': 'healthy',
         'description': 'Healthy grape vine with no disease symptoms.',
         'symptoms': 'None',
         'causes': 'N/A',
         'treatment': 'Continue regular maintenance',
         'prevention': 'Proper pruning, good air circulation, balanced nutrition',
         'pesticides': 'None needed', 'severity': 'low'},
        
        {'disease_name': 'Orange_Haunglongbing', 'plant_type': 'Orange', 'category': 'bacterial',
         'description': 'Citrus greening (HLB) - devastating bacterial disease.',
         'symptoms': 'Blotchy mottled leaves, lopsided green fruit, bitter taste',
         'causes': 'Candidatus Liberibacter asiaticus bacteria, spread by psyllid',
         'treatment': 'No cure - remove infected trees, control psyllid vector',
         'prevention': 'Use certified disease-free plants, control insect vectors',
         'pesticides': 'Insecticides for psyllid control (Imidacloprid)', 'severity': 'critical'},
        
        {'disease_name': 'Peach_Bacterial_spot', 'plant_type': 'Peach', 'category': 'bacterial',
         'description': 'Bacterial spot causing lesions on leaves and fruit.',
         'symptoms': 'Small dark spots on leaves, sunken lesions on fruit',
         'causes': 'Xanthomonas arboricola pv. pruni bacteria',
         'treatment': 'Apply copper bactericides, remove infected plant parts',
         'prevention': 'Plant resistant varieties, avoid overhead irrigation',
         'pesticides': 'Copper-based bactericides, Streptomycin', 'severity': 'moderate'},
        
        {'disease_name': 'Peach_healthy', 'plant_type': 'Peach', 'category': 'healthy',
         'description': 'Healthy peach tree with no disease symptoms.',
         'symptoms': 'None',
         'causes': 'N/A',
         'treatment': 'Continue regular care',
         'prevention': 'Proper pruning, balanced fertilization, adequate watering',
         'pesticides': 'None needed', 'severity': 'low'},
        
        {'disease_name': 'Pepper_bell_Bacterial_spot', 'plant_type': 'Pepper', 'category': 'bacterial',
         'description': 'Bacterial spot causing dark water-soaked lesions on leaves and fruit.',
         'symptoms': 'Dark brown spots with yellow halos, raised lesions on fruit',
         'causes': 'Xanthomonas campestris pv. vesicatoria bacteria',
         'treatment': 'Apply copper bactericides, improve plant spacing',
         'prevention': 'Use disease-free seeds, rotate crops, avoid overhead irrigation',
         'pesticides': 'Copper bactericides, Streptomycin', 'severity': 'moderate'},
        
        {'disease_name': 'Pepper_bell_healthy', 'plant_type': 'Pepper', 'category': 'healthy',
         'description': 'Healthy bell pepper plant with no disease symptoms.',
         'symptoms': 'None',
         'causes': 'N/A',
         'treatment': 'Continue regular care',
         'prevention': 'Adequate spacing, proper staking, consistent watering',
         'pesticides': 'None needed', 'severity': 'low'},
        
        {'disease_name': 'Potato_Early_blight', 'plant_type': 'Potato', 'category': 'fungal',
         'description': 'Common fungal disease causing target-like lesions on potato leaves.',
         'symptoms': 'Dark brown spots with concentric rings (target-like)',
         'causes': 'Alternaria solani fungus, favored by warm humid conditions',
         'treatment': 'Apply foliar fungicides, remove infected foliage',
         'prevention': 'Rotate crops (2-3 years), plant certified seed',
         'pesticides': 'Chlorothalonil, Mancozeb, Azoxystrobin', 'severity': 'moderate'},
        
        {'disease_name': 'Potato_Late_blight', 'plant_type': 'Potato', 'category': 'fungal',
         'description': 'Devastating disease that caused Irish Potato Famine.',
         'symptoms': 'Water-soaked gray-green spots, white mold on leaf undersides',
         'causes': 'Phytophthora infestans oomycete, spreads rapidly in cool wet weather',
         'treatment': 'Apply systemic fungicides immediately, destroy infected plants',
         'prevention': 'Plant resistant varieties, avoid overhead irrigation',
         'pesticides': 'Mancozeb + Cymoxanil, Metalaxyl, Fluopicolide', 'severity': 'critical'},
        
        {'disease_name': 'Potato_healthy', 'plant_type': 'Potato', 'category': 'healthy',
         'description': 'Healthy potato plant with no disease symptoms.',
         'symptoms': 'None',
         'causes': 'N/A',
         'treatment': 'Continue regular care',
         'prevention': 'Crop rotation, proper hilling, adequate drainage',
         'pesticides': 'None needed', 'severity': 'low'},
        
        {'disease_name': 'Raspberry_healthy', 'plant_type': 'Raspberry', 'category': 'healthy',
         'description': 'Healthy raspberry plant with no disease symptoms.',
         'symptoms': 'None',
         'causes': 'N/A',
         'treatment': 'Continue regular care',
         'prevention': 'Annual pruning, good air circulation, remove old canes',
         'pesticides': 'None needed', 'severity': 'low'},
        
        {'disease_name': 'Soybean_healthy', 'plant_type': 'Soybean', 'category': 'healthy',
         'description': 'Healthy soybean plant with no disease symptoms.',
         'symptoms': 'None',
         'causes': 'N/A',
         'treatment': 'Continue regular care',
         'prevention': 'Crop rotation, balanced fertilization, proper plant density',
         'pesticides': 'None needed', 'severity': 'low'},
        
        {'disease_name': 'Squash_Powdery_mildew', 'plant_type': 'Squash', 'category': 'fungal',
         'description': 'Common fungal disease causing white powdery coating on squash leaves.',
         'symptoms': 'White powdery coating on leaves, reduced photosynthesis',
         'causes': 'Podosphaeria xanthii fungus, thrives in warm dry conditions',
         'treatment': 'Apply sulfur-based or organic fungicides',
         'prevention': 'Plant resistant varieties, improve air circulation',
         'pesticides': 'Sulfur, Neem oil, Potassium bicarbonate', 'severity': 'moderate'},
        
        {'disease_name': 'Strawberry_Leaf_scorch', 'plant_type': 'Strawberry', 'category': 'fungal',
         'description': 'Leaf scorch causes purple to red spots on strawberry leaves.',
         'symptoms': 'Purple to red spots, browning leaf margins, premature leaf death',
         'causes': 'Diplocarpon earlianum fungus',
         'treatment': 'Remove infected leaves, apply fungicides',
         'prevention': 'Use resistant varieties, remove plant debris',
         'pesticides': 'Captan, Pyaclostrobin, Myclobutanil', 'severity': 'moderate'},
        
        {'disease_name': 'Strawberry_healthy', 'plant_type': 'Strawberry', 'category': 'healthy',
         'description': 'Healthy strawberry plant with no disease symptoms.',
         'symptoms': 'None',
         'causes': 'N/A',
         'treatment': 'Continue regular care',
         'prevention': 'Mulching, proper watering, regular feeding',
         'pesticides': 'None needed', 'severity': 'low'},
        
        {'disease_name': 'Tomato_Bacterial_spot', 'plant_type': 'Tomato', 'category': 'bacterial',
         'description': 'Bacterial spot causing dark lesions on leaves and fruit.',
         'symptoms': 'Small dark brown spots with yellow halos, raised lesions on fruit',
         'causes': 'Xanthomonas vesicatoria bacteria, spread by splashing water',
         'treatment': 'Apply copper bactericides, remove infected plants',
         'prevention': 'Use disease-free seeds, rotate crops, avoid overhead irrigation',
         'pesticides': 'Copper bactericides (copper hydroxide, copper sulfate)', 'severity': 'moderate'},
        
        {'disease_name': 'Tomato_Early_blight', 'plant_type': 'Tomato', 'category': 'fungal',
         'description': 'Fungal disease causing dark concentric spots on lower leaves.',
         'symptoms': 'Dark brown spots with target-like concentric rings',
         'causes': 'Alternaria solani fungus, survives in soil and plant debris',
         'treatment': 'Apply fungicides, remove infected leaves, improve air circulation',
         'prevention': 'Rotate crops (2-3 years), stake plants, mulch',
         'pesticides': 'Chlorothalonil, Mancozeb, Azoxystrobin', 'severity': 'moderate'},
        
        {'disease_name': 'Tomato_Late_blight', 'plant_type': 'Tomato', 'category': 'fungal',
         'description': 'Devastating disease causing water-soaked lesions and rapid plant death.',
         'symptoms': 'Water-soaked gray-green spots, white mold, rapid plant collapse',
         'causes': 'Phytophthora infestans oomycete, favored by cool wet conditions',
         'treatment': 'Remove infected plants immediately, apply systemic fungicides',
         'prevention': 'Plant certified disease-free transplants, avoid overhead irrigation',
         'pesticides': 'Mancozeb + Cymoxanil, Metalaxyl-m', 'severity': 'critical'},
        
        {'disease_name': 'Tomato_Leaf_Mold', 'plant_type': 'Tomato', 'category': 'fungal',
         'description': 'Fungal disease causing yellow spots and mold on tomato leaf undersides.',
         'symptoms': 'Yellow spots on upper leaf surface, olive-green to brown mold below',
         'causes': 'Passalora fuliginea fungus, thrives in high humidity',
         'treatment': 'Apply fungicides, reduce humidity, improve air circulation',
         'prevention': 'Stake plants, prune lower leaves, maintain spacing',
         'pesticides': 'Chlorothalonil, Copper fungicides', 'severity': 'moderate'},
        
        {'disease_name': 'Tomato_Septoria_leaf_spot', 'plant_type': 'Tomato', 'category': 'fungal',
         'description': 'Septoria leaf spot causes numerous small circular spots.',
         'symptoms': 'Small circular spots with dark borders, black pycnidia in center',
         'causes': 'Septoria lycopersici fungus, spread by splashing water',
         'treatment': 'Apply fungicides, remove infected leaves, improve air circulation',
         'prevention': 'Rotate crops, avoid overhead irrigation, remove plant debris',
         'pesticides': 'Chlorothalonil, Mancozeb, Copper fungicides', 'severity': 'moderate'},
        
        {'disease_name': 'Tomato_Spider_mites', 'plant_type': 'Tomato', 'category': 'pest',
         'description': 'Spider mites are tiny arachnids causing stippling and webbing.',
         'symptoms': 'Fine stippling on leaves, webbing, leaf bronzing and drop',
         'causes': 'Tetranychus urticae (two-spotted spider mite)',
         'treatment': 'Apply miticides, spray with water to dislodge mites',
         'prevention': 'Maintain humidity, avoid excessive nitrogen',
         'pesticides': 'Spinosad, Abamectin, Bifenazate', 'severity': 'moderate'},
        
        {'disease_name': 'Tomato_Target_Spot', 'plant_type': 'Tomato', 'category': 'fungal',
         'description': 'Target spot causes large brown spots with concentric rings.',
         'symptoms': 'Large brown spots with concentric rings, sunken lesions on fruit',
         'causes': 'Corynespora cassiicola fungus',
         'treatment': 'Apply fungicides, remove infected plant parts',
         'prevention': 'Rotate crops, plant resistant varieties, proper spacing',
         'pesticides': 'Azoxystrobin, Pyraclostrobin, Difenoconazole', 'severity': 'moderate'},
        
        {'disease_name': 'Tomato_Yellow_Leaf_Curl_Virus', 'plant_type': 'Tomato', 'category': 'viral',
         'description': 'TYLCV causes severe stunting, yellowing, and curling of leaves.',
         'symptoms': 'Upward leaf curling, yellowing, severe stunting, flower drop',
         'causes': 'Tomato yellow leaf curl virus (TYLCV), transmitted by whitefly',
         'treatment': 'No cure - remove infected plants, control whitefly vector',
         'prevention': 'Plant resistant varieties, use yellow sticky traps',
         'pesticides': 'Imidacloprid, Thiamethoxam for whitefly control', 'severity': 'critical'},
        
        {'disease_name': 'Tomato_mosaic_virus', 'plant_type': 'Tomato', 'category': 'viral',
         'description': 'Tomato mosaic virus causes mottled light and dark green patterns.',
         'symptoms': 'Mottled mosaic pattern, leaf curling, reduced fruit set',
         'causes': 'Tobacco mosaic virus (TMV) or Tomato mosaic virus (ToMV)',
         'treatment': 'No cure - remove and destroy infected plants',
         'prevention': 'Use virus-resistant varieties, disinfect tools',
         'pesticides': 'None effective - focus on prevention and sanitation', 'severity': 'high'},
        
        {'disease_name': 'Tomato_healthy', 'plant_type': 'Tomato', 'category': 'healthy',
         'description': 'Healthy tomato plant with no disease symptoms.',
         'symptoms': 'None',
         'causes': 'N/A',
         'treatment': 'Continue regular care',
         'prevention': 'Proper staking, consistent watering, balanced fertilization',
         'pesticides': 'None needed', 'severity': 'low'},
    ]
    
    for disease_data in diseases:
        disease = Disease(**disease_data)
        db.session.add(disease)
    
    db.session.commit()
    print(f"Seeded {len(diseases)} diseases")
