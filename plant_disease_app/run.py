"""
Plant Disease Detection System - Main Entry Point
Run this file to start the Flask application
"""

import os
from app import create_app

# Get configuration from environment
config_name = os.environ.get('FLASK_ENV', 'development')

# Create the Flask application
app = create_app(config_name)

if __name__ == '__main__':
    # Run the application
    print("=" * 60)
    print("Plant Disease Detection System")
    print("=" * 60)
    print(f"Running in {config_name} mode")
    print(f"Admin credentials: Check your .env file")
    print("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=(config_name == 'development')
    )
