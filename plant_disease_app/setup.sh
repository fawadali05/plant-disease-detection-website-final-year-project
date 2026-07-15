#!/bin/bash
# Plant Disease Detection System - Setup Script

echo "========================================"
echo "Plant Disease Detection System Setup"
echo "========================================"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOF
FLASK_APP=run.py
FLASK_ENV=development
SECRET_KEY=dev-secret-key-change-in-production-$(date +%s)
DATABASE_URL=sqlite:///plant_disease.db

# Admin credentials (change these for production!)
ADMIN_EMAIL=admin@plantdoc.com
ADMIN_PASSWORD=admin123

# ML Model path
MODEL_PATH=ml_model/plant_disease_model.h5
EOF
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p app/static/uploads
mkdir -p app/static/images
mkdir -p ml_model

echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "To run the application:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run the app: python run.py"
echo ""
echo "Admin login:"
echo "  Email: admin@plantdoc.com"
echo "  Password: admin123"
echo ""
echo "========================================"
