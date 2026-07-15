"""
User Routes - Dashboard, Upload, Detection History, Reports
"""

import os
import io
from datetime import datetime, timedelta
from flask import render_template, Blueprint, request, flash, redirect, url_for, send_file, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from sqlalchemy import desc, func
from models import db, User, Detection, Disease
from functools import wraps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import base64

user_bp = Blueprint('user', __name__)


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def admin_required(f):
    """Decorator to require admin role."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'admin':
            flash('You do not have permission to access this page.', 'danger')
            return redirect(url_for('main.home'))
        return f(*args, **kwargs)
    return decorated_function


@user_bp.route('/dashboard')
@login_required
def dashboard():
    """User dashboard with detection history."""
    # Get user's recent detections
    recent_detections = Detection.query.filter_by(user_id=current_user.id)\
        .order_by(desc(Detection.detected_at)).limit(10).all()
    
    # Get statistics
    total_detections = Detection.query.filter_by(user_id=current_user.id).count()
    
    # Get disease distribution
    disease_stats = db.session.query(
        Detection.disease_name,
        func.count(Detection.id).label('count')
    ).filter_by(user_id=current_user.id)\
        .group_by(Detection.disease_name)\
        .order_by(desc('count')).limit(5).all()
    
    # Get detections over time (last 30 days)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    recent_count = Detection.query.filter(
        Detection.user_id == current_user.id,
        Detection.detected_at >= thirty_days_ago
    ).count()
    
    return render_template(
        'user/dashboard.html',
        title='Dashboard',
        recent_detections=recent_detections,
        total_detections=total_detections,
        disease_stats=disease_stats,
        recent_count=recent_count
    )


@user_bp.route('/detect', methods=['GET', 'POST'])
@login_required
def detect():
    """Upload and detect plant disease."""
    if request.method == 'POST':
        # Check if file is present
        if 'file' not in request.files:
            flash('No file selected. Please choose an image to upload.', 'warning')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected. Please choose an image to upload.', 'warning')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                # Save uploaded file
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                unique_filename = f"{timestamp}_{filename}"
                upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(upload_path), exist_ok=True)
                
                # Save file
                file.save(upload_path)
                
                # Process image with EfficientNet ML model
                from ml_model.efficientnet_predictor import get_classifier, get_disease_full_info
                
                classifier = get_classifier()
                predictions = classifier.predict(upload_path)
                
                # Get top prediction
                top_prediction = predictions[0]
                
                # Get disease info from database
                disease_info = Disease.query.filter(
                    Disease.disease_name.ilike(f"%{top_prediction['class_name']}%")
                ).first()
                
                # Get full disease info
                full_info = get_disease_full_info(top_prediction['class_name'])
                
                # Save detection record
                detection = Detection(
                    user_id=current_user.id,
                    image_path=unique_filename,
                    disease_name=top_prediction['disease_name'],
                    plant_type=top_prediction['plant_type'],
                    confidence=top_prediction['confidence']
                )
                db.session.add(detection)
                db.session.commit()
                
                # Prepare results
                results = {
                    'detection_id': detection.id,
                    'disease_name': top_prediction['disease_name'],
                    'plant_type': top_prediction['plant_type'],
                    'confidence': top_prediction['confidence'],
                    'image_path': unique_filename,
                    'all_predictions': predictions,
                    'disease_info': disease_info,
                    'full_info': full_info
                }
                
                return render_template(
                    'user/result.html',
                    title='Detection Results',
                    results=results
                )
                
            except Exception as e:
                flash(f'Error processing image: {str(e)}', 'danger')
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload a JPG, PNG, or GIF image.', 'warning')
            return redirect(request.url)
    
    return render_template(
        'user/detect.html',
        title='Detect Disease'
    )


@user_bp.route('/history')
@login_required
def history():
    """View detection history."""
    # Get pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = 12
    
    # Query detections with pagination
    pagination = Detection.query.filter_by(user_id=current_user.id)\
        .order_by(desc(Detection.detected_at))\
        .paginate(page=page, per_page=per_page, error_out=False)
    
    detections = pagination.items
    
    return render_template(
        'user/history.html',
        title='Detection History',
        detections=detections,
        pagination=pagination
    )


@user_bp.route('/detection/<int:detection_id>')
@login_required
def detection_detail(detection_id):
    """View detailed detection result."""
    detection = Detection.query.filter_by(
        id=detection_id,
        user_id=current_user.id
    ).first_or_404()
    
    # Get disease info
    disease_info = Disease.query.filter(
        Disease.disease_name.ilike(f"%{detection.disease_name}%")
    ).first()
    
    return render_template(
        'user/detection_detail.html',
        title=f'Detection #{detection.id}',
        detection=detection,
        disease_info=disease_info
    )


@user_bp.route('/download-report/<int:detection_id>')
@login_required
def download_report(detection_id):
    """Download PDF report for a detection."""
    detection = Detection.query.filter_by(
        id=detection_id,
        user_id=current_user.id
    ).first_or_404()
    
    # Generate PDF
    pdf_buffer = generate_pdf_report(detection_id)
    
    filename = f"plant_disease_report_{detection.id}.pdf"
    return send_file(
        pdf_buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=filename
    )


def generate_pdf_report(detection_id):
    """Generate PDF report for a detection."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.units import inch
    from io import BytesIO
    
    # Get detection
    detection = Detection.query.get(detection_id)
    disease_info = Disease.query.filter(
        Disease.disease_name.ilike(f"%{detection.disease_name}%")
    ).first()
    
    # Create PDF buffer
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        spaceAfter=20,
        textColor=colors.HexColor('#2E7D32')
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=15,
        spaceAfter=10,
        textColor=colors.HexColor('#1976D2')
    )
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=8
    )
    
    # Build content
    story = []
    
    # Title
    story.append(Paragraph("Plant Disease Detection Report", title_style))
    story.append(Spacer(1, 10))
    
    # Report Info
    report_date = detection.detected_at.strftime('%B %d, %Y at %I:%M %p')
    story.append(Paragraph(f"<b>Report Date:</b> {report_date}", normal_style))
    story.append(Paragraph(f"<b>Report ID:</b> DET-{detection.id:06d}", normal_style))
    story.append(Spacer(1, 15))
    
    # Diagnosis Section
    story.append(Paragraph("Diagnosis Results", heading_style))
    
    diagnosis_data = [
        ['Disease Name:', detection.disease_name],
        ['Plant Type:', detection.plant_type],
        ['Confidence:', f"{detection.confidence:.1f}%"],
        ['Status:', 'Completed' if detection.status == 'completed' else 'Under Review']
    ]
    
    diagnosis_table = Table(diagnosis_data, colWidths=[2*inch, 4*inch])
    diagnosis_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E8F5E9')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    story.append(diagnosis_table)
    story.append(Spacer(1, 15))
    
    # Disease Information
    if disease_info:
        story.append(Paragraph("Disease Information", heading_style))
        
        info_data = [
            ['Category:', disease_info.category.title() if disease_info.category else 'N/A'],
            ['Severity:', disease_info.severity.title() if disease_info.severity else 'N/A'],
        ]
        
        if disease_info.description:
            story.append(Paragraph(f"<b>Description:</b><br/>{disease_info.description}", normal_style))
        
        if disease_info.symptoms:
            story.append(Paragraph(f"<b>Symptoms:</b><br/>{disease_info.symptoms}", normal_style))
        
        if disease_info.causes:
            story.append(Paragraph(f"<b>Causes:</b><br/>{disease_info.causes}", normal_style))
        story.append(Spacer(1, 15))
        
        # Treatment Section
        story.append(Paragraph("Treatment Recommendations", heading_style))
        if disease_info.treatment:
            story.append(Paragraph(disease_info.treatment, normal_style))
        story.append(Spacer(1, 10))
        
        # Prevention Section
        if disease_info.prevention:
            story.append(Paragraph("Prevention Tips", heading_style))
            story.append(Paragraph(disease_info.prevention, normal_style))
            story.append(Spacer(1, 10))
        
        # Recommended Pesticides
        if disease_info.pesticides:
            story.append(Paragraph("Recommended Pesticides", heading_style))
            story.append(Paragraph(disease_info.pesticides, normal_style))
    
    # Footer
    story.append(Spacer(1, 30))
    story.append(Paragraph("_" * 80, normal_style))
    story.append(Paragraph(
        "This report was generated by the Plant Disease Detection System. "
        "For professional agricultural advice, please consult with a certified agricultural expert.",
        ParagraphStyle('Footer', parent=normal_style, fontSize=8, textColor=colors.grey)
    ))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    return buffer


@user_bp.route('/export-history')
@login_required
def export_history():
    """Export detection history as CSV."""
    detections = Detection.query.filter_by(user_id=current_user.id)\
        .order_by(Detection.detected_at.desc()).all()
    
    # Create CSV
    csv_data = "ID,Date,Disease Name,Plant Type,Confidence (%)\n"
    for d in detections:
        csv_data += f"{d.id},{d.detected_at.strftime('%Y-%m-%d %H:%M')},{d.disease_name},{d.plant_type},{d.confidence:.1f}\n"
    
    buffer = io.BytesIO()
    buffer.write(csv_data.encode('utf-8'))
    buffer.seek(0)
    
    return send_file(
        buffer,
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'detection_history_{datetime.now().strftime("%Y%m%d")}.csv'
    )


@user_bp.route('/delete-detection/<int:detection_id>', methods=['POST'])
@login_required
def delete_detection(detection_id):
    """Delete a detection record."""
    detection = Detection.query.filter_by(
        id=detection_id,
        user_id=current_user.id
    ).first_or_404()
    
    # Delete associated image
    image_path = os.path.join(current_app.config['UPLOAD_FOLDER'], detection.image_path)
    if os.path.exists(image_path):
        os.remove(image_path)
    
    db.session.delete(detection)
    db.session.commit()
    
    flash('Detection record deleted successfully.', 'success')
    return redirect(url_for('user.history'))
