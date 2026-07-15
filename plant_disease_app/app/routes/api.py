"""
API Routes - RESTful API for mobile apps and integrations
"""

import os
from flask import jsonify, Blueprint, request, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from datetime import datetime
from models import db, Detection, Disease, User
from sqlalchemy import desc
import base64

api_bp = Blueprint('api', __name__)


def allowed_file(filename):
    """Check if file extension is allowed."""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@api_bp.route('/health')
def health_check():
    """API health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Plant Disease Detection API',
        'version': '1.0.0',
        'timestamp': datetime.utcnow().isoformat()
    })


@api_bp.route('/diseases')
def get_diseases():
    """Get all diseases (public endpoint)."""
    diseases = Disease.query.all()
    
    return jsonify({
        'success': True,
        'count': len(diseases),
        'data': [{
            'id': d.id,
            'name': d.disease_name,
            'plant_type': d.plant_type,
            'category': d.category,
            'description': d.description,
            'treatment': d.treatment,
            'prevention': d.prevention,
            'severity': d.severity
        } for d in diseases]
    })


@api_bp.route('/diseases/<int:disease_id>')
def get_disease(disease_id):
    """Get a specific disease by ID."""
    disease = Disease.query.get(disease_id)
    
    if not disease:
        return jsonify({'success': False, 'error': 'Disease not found'}), 404
    
    return jsonify({
        'success': True,
        'data': {
            'id': disease.id,
            'name': disease.disease_name,
            'plant_type': disease.plant_type,
            'category': disease.category,
            'description': disease.description,
            'symptoms': disease.symptoms,
            'causes': disease.causes,
            'treatment': disease.treatment,
            'prevention': disease.prevention,
            'pesticides': disease.pesticides,
            'recovery_time': disease.recovery_time,
            'severity': disease.severity
        }
    })


@api_bp.route('/detect', methods=['POST'])
def api_detect():
    """API endpoint for disease detection."""
    # Check authentication
    if not current_user.is_authenticated:
        return jsonify({'success': False, 'error': 'Authentication required'}), 401
    
    # Check if file is present
    if 'image' not in request.files and 'image_base64' not in request.form:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    
    try:
        # Handle file upload or base64
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No file selected'}), 400
            
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_filename = f"{timestamp}_{filename}"
            upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
            
            os.makedirs(os.path.dirname(upload_path), exist_ok=True)
            file.save(upload_path)
        else:
            # Handle base64 image
            image_data = request.form.get('image_base64')
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            import base64
            image_bytes = base64.b64decode(image_data)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_filename = f"{timestamp}_api_image.jpg"
            upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
            
            os.makedirs(os.path.dirname(upload_path), exist_ok=True)
            with open(upload_path, 'wb') as f:
                f.write(image_bytes)
        
        # Run prediction
        from ml_model.disease_predictor import get_model
        model = get_model()
        predictions = model.predict(upload_path)
        
        top_prediction = predictions[0]
        
        # Get disease info
        disease_info = Disease.query.filter(
            Disease.disease_name.ilike(f"%{top_prediction['original_class']}%")
        ).first()
        
        # Save detection
        detection = Detection(
            user_id=current_user.id,
            image_path=unique_filename,
            disease_name=top_prediction['disease_name'],
            plant_type=top_prediction['plant_type'],
            confidence=top_prediction['confidence']
        )
        db.session.add(detection)
        db.session.commit()
        
        # Prepare response
        response = {
            'success': True,
            'detection_id': detection.id,
            'results': {
                'disease_name': top_prediction['disease_name'],
                'plant_type': top_prediction['plant_type'],
                'confidence': top_prediction['confidence'],
                'category': top_prediction['category'],
                'all_predictions': [{
                    'disease': p['disease_name'],
                    'confidence': p['confidence']
                } for p in predictions]
            }
        }
        
        if disease_info:
            response['disease_info'] = {
                'description': disease_info.description,
                'treatment': disease_info.treatment,
                'prevention': disease_info.prevention,
                'pesticides': disease_info.pesticides
            }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/user/history')
@login_required
def user_history():
    """Get detection history for authenticated user."""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    detections = Detection.query.filter_by(user_id=current_user.id)\
        .order_by(desc(Detection.detected_at))\
        .paginate(page=page, per_page=per_page, error_out=False)
    
    return jsonify({
        'success': True,
        'page': page,
        'per_page': per_page,
        'total': detections.total,
        'pages': detections.pages,
        'data': [{
            'id': d.id,
            'disease_name': d.disease_name,
            'plant_type': d.plant_type,
            'confidence': d.confidence,
            'detected_at': d.detected_at.isoformat()
        } for d in detections.items]
    })


@api_bp.route('/stats/public')
def public_stats():
    """Get public statistics."""
    total_detections = Detection.query.count()
    total_diseases = Disease.query.count()
    
    return jsonify({
        'success': True,
        'total_detections': total_detections,
        'total_diseases': total_diseases
    })
