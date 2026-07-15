"""
Main Routes - Home, About, Contact, How It Works
"""

from flask import render_template, Blueprint, request, flash, redirect, url_for
from flask_login import current_user
from models import db, ContactMessage, Disease, Detection
from app.forms import ContactForm
import os

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def home():
    """Home page / Landing page."""
    # Get statistics
    total_detections = Detection.query.count()
    total_diseases = Disease.query.count()
    disease_categories = db.session.query(
        Disease.category, db.func.count(Disease.id)
    ).group_by(Disease.category).all()
    
    # Sample diseases for display
    sample_diseases = Disease.query.limit(6).all()
    
    return render_template(
        'main/home.html',
        title='Plant Disease Detection System',
        total_detections=total_detections,
        total_diseases=total_diseases,
        disease_categories=disease_categories,
        sample_diseases=sample_diseases
    )


@main_bp.route('/about')
def about():
    """About page."""
    return render_template(
        'main/about.html',
        title='About Us'
    )


@main_bp.route('/how-it-works')
def how_it_works():
    """How it works page."""
    steps = [
        {
            'number': 1,
            'title': 'Upload Image',
            'description': 'Take a photo or upload an image of a plant leaf showing symptoms.',
            'icon': 'bi-camera'
        },
        {
            'number': 2,
            'title': 'AI Analysis',
            'description': 'Our deep learning model analyzes the image to identify disease patterns.',
            'icon': 'bi-cpu'
        },
        {
            'number': 3,
            'title': 'Get Diagnosis',
            'description': 'Receive instant diagnosis with confidence score and disease details.',
            'icon': 'bi-clipboard-check'
        },
        {
            'number': 4,
            'title': 'Treatment Plan',
            'description': 'Get recommended treatments, prevention tips, and recovery guidance.',
            'icon': 'bi-heart-pulse'
        }
    ]
    
    return render_template(
        'main/how_it_works.html',
        title='How It Works',
        steps=steps
    )


@main_bp.route('/contact', methods=['GET', 'POST'])
def contact():
    """Contact form page."""
    form = ContactForm()
    
    if form.validate_on_submit():
        # Save contact message
        message = ContactMessage(
            name=form.name.data,
            email=form.email.data,
            subject=form.subject.data,
            message=form.message.data
        )
        db.session.add(message)
        db.session.commit()
        
        flash('Thank you for your message! We will get back to you soon.', 'success')
        return redirect(url_for('main.contact'))
    
    return render_template(
        'main/contact.html',
        title='Contact Us',
        form=form
    )


@main_bp.route('/diseases')
def diseases_list():
    """List all diseases in the database."""
    # Get filter parameters
    category = request.args.get('category', '')
    plant_type = request.args.get('plant', '')
    search = request.args.get('search', '')
    
    # Build query
    query = Disease.query
    
    if category:
        query = query.filter(Disease.category == category)
    if plant_type:
        query = query.filter(Disease.plant_type == plant_type)
    if search:
        query = query.filter(
            db.or_(
                Disease.disease_name.ilike(f'%{search}%'),
                Disease.description.ilike(f'%{search}%')
            )
        )
    
    diseases = query.order_by(Disease.disease_name).all()
    
    # Get unique plant types for filter
    plant_types = db.session.query(Disease.plant_type).distinct().all()
    plant_types = [p[0] for p in plant_types]
    
    # Get categories with counts
    categories = db.session.query(
        Disease.category, db.func.count(Disease.id)
    ).group_by(Disease.category).all()
    
    return render_template(
        'main/diseases.html',
        title='Disease Database',
        diseases=diseases,
        plant_types=plant_types,
        categories=categories,
        selected_category=category,
        selected_plant=plant_type,
        search_query=search
    )


@main_bp.route('/diseases/<int:disease_id>')
def disease_detail(disease_id):
    """View disease details."""
    disease = Disease.query.get_or_404(disease_id)
    return render_template(
        'main/disease_detail.html',
        title=disease.disease_name,
        disease=disease
    )


@main_bp.route('/download-report/<int:detection_id>')
def download_report(detection_id):
    """Download detection report (redirects to user route)."""
    from flask_login import login_required
    from flask import send_file
    from app.routes.user import generate_pdf_report
    
    @login_required
    def _download():
        return generate_pdf_report(detection_id)
    
    return _download()


@main_bp.route('/privacy')
def privacy():
    """Privacy policy page."""
    return render_template(
        'main/privacy.html',
        title='Privacy Policy'
    )


@main_bp.route('/terms')
def terms():
    """Terms of service page."""
    return render_template(
        'main/terms.html',
        title='Terms of Service'
    )


@main_bp.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    return render_template('errors/404.html', title='Page Not Found'), 404


@main_bp.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    db.session.rollback()
    return render_template('errors/500.html', title='Server Error'), 500
