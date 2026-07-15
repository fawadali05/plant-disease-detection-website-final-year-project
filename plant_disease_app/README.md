# Plant Disease Detection System

A complete, deployable web application for plant disease detection using deep learning (CNN). This is a Final Year Project developed by **Fawad Ali** and **Syed Asad Ullah**.

## 📋 Project Overview

This web-based system allows users to:
- Upload a leaf image
- Get AI-powered disease diagnosis
- View confidence scores and treatment recommendations
- Download PDF reports
- Track detection history

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Backend | Python, Flask |
| ML Model | TensorFlow/Keras (ResNet50/MobileNetV2) |
| Frontend | HTML5, CSS3, Bootstrap 5, JavaScript |
| Database | SQLite (dev) / MySQL (prod) |
| ORM | SQLAlchemy |
| Authentication | Flask-Login |

## 📁 Project Structure

```
plant_disease_app/
├── app/
│   ├── __init__.py          # App factory
│   ├── forms.py              # WTForms
│   ├── models/                # Database models
│   ├── routes/               # Flask blueprints
│   │   ├── main.py           # Home, About, Contact
│   │   ├── auth.py           # Login, Register
│   │   ├── user.py           # Dashboard, Detection
│   │   ├── admin.py          # Admin panel
│   │   └── api.py            # REST API
│   ├── static/
│   │   ├── css/style.css     # Custom styles
│   │   ├── js/               # JavaScript files
│   │   └── uploads/          # Uploaded images
│   └── templates/            # HTML templates
├── ml_model/                 # ML model files
│   └── disease_predictor.py
├── models/                   # Database models
├── instance/                 # SQLite database
├── config.py                 # Configuration
├── requirements.txt          # Dependencies
└── run.py                    # Entry point
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip

### Step 1: Clone/Download the Project
```bash
cd plant_disease_app
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment
Create a `.env` file in the project root:
```env
FLASK_APP=run.py
FLASK_ENV=development
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///plant_disease.db

# Admin credentials (change these!)
ADMIN_EMAIL=admin@plantdoc.com
ADMIN_PASSWORD=admin123

# ML Model path
MODEL_PATH=ml_model/plant_disease_model.h5
```

### Step 5: Download Pre-trained Model (Optional)

For best results, download a pre-trained model:

1. **Option A: Hugging Face**
   - Search for "plant disease classification" models
   - Download and place in `ml_model/`

2. **Option B: Kaggle**
   - Download from [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
   - Train your own model or use pre-trained weights

3. **Option C: Use Demo Mode**
   - The app will use a base ResNet50 model with ImageNet weights
   - Not trained on plant diseases, but functional

### Step 6: Run the Application
```bash
python run.py
```

Visit `http://localhost:5000` in your browser.

## 📖 Usage Guide

### For Users
1. Register an account
2. Log in
3. Click "Detect Disease" on the dashboard
4. Upload a clear leaf image
5. View the diagnosis and treatment recommendations
6. Download the PDF report

### For Administrators
1. Log in with admin credentials (set in `.env`)
2. Access the Admin Panel from the navigation menu
3. Manage users, diseases, and view analytics

## 🌐 Deployment

### Option 1: Render (Recommended)

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Set environment variables:
   - `FLASK_ENV=production`
   - `SECRET_KEY=<your-secret-key>`
   - `DATABASE_URL=<your-database-url>`
4. Build command: `pip install -r requirements.txt`
5. Start command: `gunicorn run:app`

### Option 2: Railway

1. Create a new Railway project
2. Add environment variables
3. Deploy from GitHub

### Option 3: PythonAnywhere

1. Upload the project files
2. Create a virtual environment
3. Install dependencies
4. Configure WSGI file
5. Set up the database

### Option 4: Heroku

1. Create `Procfile`:
   ```
   web: gunicorn run:app
   ```
2. Create `runtime.txt`:
   ```
   python-3.11.0
   ```
3. Deploy via GitHub

## 📊 ML Model Information

### Dataset
- **PlantVillage Dataset**: 38 disease classes across 14 plant species
- ~50,000+ labeled images
- Available at: [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)

### Supported Plants
- Apple, Blueberry, Cherry, Corn
- Grape, Orange, Peach, Pepper
- Potato, Raspberry, Soybean, Squash
- Strawberry, Tomato

### Supported Disease Categories
- Fungal
- Bacterial
- Viral
- Pest-related

## 🔧 API Documentation

### Public Endpoints
```
GET  /api/health              - Health check
GET  /api/diseases            - List all diseases
GET  /api/diseases/<id>       - Get disease details
GET  /api/stats/public         - Public statistics
```

### Protected Endpoints (Requires Authentication)
```
POST /api/detect              - Upload image for detection
GET  /api/user/history        - Get user's detection history
```

## 📝 Database Schema

### Users Table
| Field | Type | Description |
|-------|------|-------------|
| id | Integer | Primary key |
| name | String | Full name |
| email | String | Unique email |
| password_hash | String | Hashed password |
| role | String | 'user' or 'admin' |
| created_at | DateTime | Account creation date |

### Detections Table
| Field | Type | Description |
|-------|------|-------------|
| id | Integer | Primary key |
| user_id | Integer | Foreign key to users |
| image_path | String | Path to uploaded image |
| disease_name | String | Detected disease |
| plant_type | String | Affected plant |
| confidence | Float | Confidence score (0-100) |
| detected_at | DateTime | Detection timestamp |

### Diseases Table
| Field | Type | Description |
|-------|------|-------------|
| id | Integer | Primary key |
| disease_name | String | Disease name |
| plant_type | String | Affected plant |
| category | String | Fungal/Bacterial/Viral |
| description | Text | Disease description |
| symptoms | Text | Symptoms |
| causes | Text | Causes |
| treatment | Text | Treatment recommendations |
| prevention | Text | Prevention tips |
| pesticides | Text | Recommended pesticides |

## 📄 PDF Report Features

The system generates comprehensive PDF reports including:
- Detection ID and timestamp
- Uploaded image
- Diagnosis results with confidence
- Disease information
- Treatment recommendations
- Prevention tips
- Recommended pesticides

## 🔐 Security Features

- Password hashing with bcrypt
- Session management
- CSRF protection
- Input validation
- SQL injection prevention

## 📈 Analytics Features

- Total detections tracking
- Disease distribution charts
- User activity tracking
- Daily/Monthly usage statistics
- CSV export functionality

## 🤝 Contributing

This is a Final Year Project. Contributions are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Contact

- **Fawad Ali** - Project Lead
- **Syed Asad Ullah** - Developer

## 📄 License

This project is for educational purposes as part of a Final Year Project.

## 🙏 Acknowledgments

- PlantVillage Dataset creators
- TensorFlow/Keras team
- Bootstrap team
- Flask community

---

**Project Year**: 2024
**Supervisor**: [Your Supervisor's Name]
**Institution**: [Your University Name]
