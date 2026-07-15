"""
Admin Routes - Dashboard, User Management, Disease Management, Analytics
"""

import os
import io
from datetime import datetime, timedelta
from flask import render_template, Blueprint, request, flash, redirect, url_for, send_file, current_app, abort
from flask_login import login_required, current_user
from sqlalchemy import desc, func, and_
from functools import wraps
from models import db, User, Detection, Disease, AdminLog, ContactMessage
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO

admin_bp = Blueprint('admin', __name__)


def admin_required(f):
    """Decorator to require admin role."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('auth.login'))
        if current_user.role != 'admin':
            flash('You do not have permission to access this page.', 'danger')
            return redirect(url_for('main.home'))
        return f(*args, **kwargs)
    return decorated_function


def log_admin_action(action, target=None, details=None):
    """Log an admin action."""
    log = AdminLog(
        admin_id=current_user.id,
        action=action,
        target=target,
        details=details,
        ip_address=request.remote_addr
    )
    db.session.add(log)
    db.session.commit()


@admin_bp.route('/')
@admin_required
def dashboard():
    """Admin dashboard with overview statistics."""
    # Basic counts
    total_users = User.query.filter_by(role='user').count()
    total_detections = Detection.query.count()
    total_diseases = Disease.query.count()
    unread_messages = ContactMessage.query.filter_by(is_read=False).count()
    
    # Recent detections
    recent_detections = Detection.query.order_by(desc(Detection.detected_at)).limit(10).all()
    
    # Statistics
    today = datetime.utcnow().date()
    week_ago = today - timedelta(days=7)
    
    detections_today = Detection.query.filter(
        func.date(Detection.detected_at) == today
    ).count()
    
    detections_week = Detection.query.filter(
        func.date(Detection.detected_at) >= week_ago
    ).count()
    
    # Most common diseases
    common_diseases = db.session.query(
        Detection.disease_name,
        func.count(Detection.id).label('count')
    ).group_by(Detection.disease_name)\
        .order_by(desc('count')).limit(10).all()
    
    # Disease category distribution
    category_stats = db.session.query(
        Disease.category,
        func.count(Detection.id).label('count')
    ).join(Detection, Disease.disease_name == Detection.disease_name)\
        .group_by(Disease.category).all()
    
    # Active users (last 30 days)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    active_users = User.query.join(Detection).filter(
        Detection.detected_at >= thirty_days_ago
    ).distinct().count()
    
    return render_template(
        'admin/dashboard.html',
        title='Admin Dashboard',
        total_users=total_users,
        total_detections=total_detections,
        total_diseases=total_diseases,
        unread_messages=unread_messages,
        recent_detections=recent_detections,
        detections_today=detections_today,
        detections_week=detections_week,
        active_users=active_users,
        common_diseases=common_diseases,
        category_stats=category_stats
    )


@admin_bp.route('/users')
@admin_required
def users():
    """Manage users."""
    page = request.args.get('page', 1, type=int)
    search = request.args.get('search', '')
    
    query = User.query.filter_by(role='user')
    
    if search:
        query = query.filter(
            db.or_(
                User.name.ilike(f'%{search}%'),
                User.email.ilike(f'%{search}%')
            )
        )
    
    pagination = query.order_by(desc(User.created_at))\
        .paginate(page=page, per_page=20, error_out=False)
    
    return render_template(
        'admin/users.html',
        title='Manage Users',
        users=pagination.items,
        pagination=pagination,
        search=search
    )


@admin_bp.route('/users/<int:user_id>/toggle-status', methods=['POST'])
@admin_required
def toggle_user_status(user_id):
    """Toggle user active status."""
    user = User.query.get_or_404(user_id)
    
    if user.id == current_user.id:
        flash('You cannot deactivate your own account.', 'danger')
        return redirect(url_for('admin.users'))
    
    user.is_active = not user.is_active
    db.session.commit()
    
    status = 'activated' if user.is_active else 'deactivated'
    log_admin_action(f'Toggle user status', f'User {user.email}', f'Status changed to {status}')
    
    flash(f'User {user.email} has been {status}.', 'success')
    return redirect(url_for('admin.users'))


@admin_bp.route('/users/<int:user_id>/delete', methods=['POST'])
@admin_required
def delete_user(user_id):
    """Delete a user and all their data."""
    user = User.query.get_or_404(user_id)
    
    if user.id == current_user.id:
        flash('You cannot delete your own account.', 'danger')
        return redirect(url_for('admin.users'))
    
    log_admin_action('Delete user', f'User {user.email}', f'User {user.name} deleted')
    
    db.session.delete(user)
    db.session.commit()
    
    flash('User deleted successfully.', 'success')
    return redirect(url_for('admin.users'))


@admin_bp.route('/detections')
@admin_required
def detections():
    """View all detections."""
    page = request.args.get('page', 1, type=int)
    search = request.args.get('search', '')
    disease_filter = request.args.get('disease', '')
    
    query = Detection.query
    
    if search:
        query = query.join(User).filter(
            db.or_(
                User.name.ilike(f'%{search}%'),
                User.email.ilike(f'%{search}%')
            )
        )
    
    if disease_filter:
        query = query.filter(Detection.disease_name == disease_filter)
    
    pagination = query.order_by(desc(Detection.detected_at))\
        .paginate(page=page, per_page=25, error_out=False)
    
    # Get all disease names for filter dropdown
    diseases = db.session.query(Detection.disease_name).distinct().all()
    diseases = [d[0] for d in diseases]
    
    return render_template(
        'admin/detections.html',
        title='Detection Records',
        detections=pagination.items,
        pagination=pagination,
        diseases=diseases,
        search=search,
        disease_filter=disease_filter
    )


@admin_bp.route('/diseases')
@admin_required
def diseases():
    """Manage disease database."""
    page = request.args.get('page', 1, type=int)
    search = request.args.get('search', '')
    
    query = Disease.query
    
    if search:
        query = query.filter(
            db.or_(
                Disease.disease_name.ilike(f'%{search}%'),
                Disease.plant_type.ilike(f'%{search}%')
            )
        )
    
    pagination = query.order_by(Disease.plant_type, Disease.disease_name)\
        .paginate(page=page, per_page=20, error_out=False)
    
    return render_template(
        'admin/diseases.html',
        title='Disease Database',
        diseases=pagination.items,
        pagination=pagination,
        search=search
    )


@admin_bp.route('/diseases/add', methods=['GET', 'POST'])
@admin_required
def add_disease():
    """Add a new disease."""
    from app.forms import DiseaseForm
    
    form = DiseaseForm()
    
    if form.validate_on_submit():
        disease = Disease(
            disease_name=form.disease_name.data,
            plant_type=form.plant_type.data,
            category=form.category.data,
            description=form.description.data,
            symptoms=form.symptoms.data,
            causes=form.causes.data,
            treatment=form.treatment.data,
            prevention=form.prevention.data,
            pesticides=form.pesticides.data,
            recovery_time=form.recovery_time.data,
            severity=form.severity.data
        )
        
        db.session.add(disease)
        db.session.commit()
        
        log_admin_action('Add disease', disease.disease_name, f'Added by {current_user.email}')
        
        flash(f'Disease "{disease.disease_name}" added successfully!', 'success')
        return redirect(url_for('admin.diseases'))
    
    return render_template(
        'admin/disease_form.html',
        title='Add Disease',
        form=form,
        action='Add'
    )


@admin_bp.route('/diseases/<int:disease_id>/edit', methods=['GET', 'POST'])
@admin_required
def edit_disease(disease_id):
    """Edit a disease."""
    from app.forms import DiseaseForm
    
    disease = Disease.query.get_or_404(disease_id)
    form = DiseaseForm(obj=disease)
    
    if form.validate_on_submit():
        disease.disease_name = form.disease_name.data
        disease.plant_type = form.plant_type.data
        disease.category = form.category.data
        disease.description = form.description.data
        disease.symptoms = form.symptoms.data
        disease.causes = form.causes.data
        disease.treatment = form.treatment.data
        disease.prevention = form.prevention.data
        disease.pesticides = form.pesticides.data
        disease.recovery_time = form.recovery_time.data
        disease.severity = form.severity.data
        disease.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        log_admin_action('Edit disease', disease.disease_name, f'Edited by {current_user.email}')
        
        flash(f'Disease "{disease.disease_name}" updated successfully!', 'success')
        return redirect(url_for('admin.diseases'))
    
    return render_template(
        'admin/disease_form.html',
        title='Edit Disease',
        form=form,
        disease=disease,
        action='Edit'
    )


@admin_bp.route('/diseases/<int:disease_id>/delete', methods=['POST'])
@admin_required
def delete_disease(disease_id):
    """Delete a disease."""
    disease = Disease.query.get_or_404(disease_id)
    
    log_admin_action('Delete disease', disease.disease_name, f'Deleted by {current_user.email}')
    
    db.session.delete(disease)
    db.session.commit()
    
    flash('Disease deleted successfully.', 'success')
    return redirect(url_for('admin.diseases'))


@admin_bp.route('/messages')
@admin_required
def messages():
    """View contact messages."""
    page = request.args.get('page', 1, type=int)
    unread_only = request.args.get('unread', 'false') == 'true'
    
    query = ContactMessage.query
    
    if unread_only:
        query = query.filter_by(is_read=False)
    
    pagination = query.order_by(desc(ContactMessage.created_at))\
        .paginate(page=page, per_page=20, error_out=False)
    
    unread_count = ContactMessage.query.filter_by(is_read=False).count()
    
    return render_template(
        'admin/messages.html',
        title='Contact Messages',
        messages=pagination.items,
        pagination=pagination,
        unread_only=unread_only,
        unread_count=unread_count
    )


@admin_bp.route('/messages/<int:message_id>/read')
@admin_required
def mark_message_read(message_id):
    """Mark a message as read."""
    message = ContactMessage.query.get_or_404(message_id)
    message.is_read = True
    db.session.commit()
    
    return redirect(url_for('admin.view_message', message_id=message_id))


@admin_bp.route('/messages/<int:message_id>')
@admin_required
def view_message(message_id):
    """View a single message."""
    message = ContactMessage.query.get_or_404(message_id)
    
    if not message.is_read:
        message.is_read = True
        db.session.commit()
    
    return render_template(
        'admin/view_message.html',
        title='View Message',
        message=message
    )


@admin_bp.route('/messages/<int:message_id>/delete', methods=['POST'])
@admin_required
def delete_message(message_id):
    """Delete a message."""
    message = ContactMessage.query.get_or_404(message_id)
    
    db.session.delete(message)
    db.session.commit()
    
    flash('Message deleted successfully.', 'success')
    return redirect(url_for('admin.messages'))


@admin_bp.route('/analytics')
@admin_required
def analytics():
    """View analytics and statistics."""
    # Time-based stats (last 30 days)
    dates = []
    counts = []
    for i in range(30, -1, -1):
        date = datetime.utcnow().date() - timedelta(days=i)
        count = Detection.query.filter(func.date(Detection.detected_at) == date).count()
        dates.append(date.strftime('%Y-%m-%d'))
        counts.append(count)
    
    # Disease distribution
    disease_stats = db.session.query(
        Detection.disease_name,
        func.count(Detection.id).label('count')
    ).group_by(Detection.disease_name)\
        .order_by(desc('count')).limit(15).all()
    
    # Plant type distribution
    plant_stats = db.session.query(
        Detection.plant_type,
        func.count(Detection.id).label('count')
    ).group_by(Detection.plant_type)\
        .order_by(desc('count')).all()
    
    # User growth
    user_dates = []
    user_counts = []
    cumulative = 0
    for i in range(30, -1, -1):
        date = datetime.utcnow().date() - timedelta(days=i)
        new_users = User.query.filter(func.date(User.created_at) == date).count()
        cumulative += new_users
        user_dates.append(date.strftime('%Y-%m-%d'))
        user_counts.append(cumulative)
    
    return render_template(
        'admin/analytics.html',
        title='Analytics',
        dates=dates,
        counts=counts,
        disease_stats=disease_stats,
        plant_stats=plant_stats,
        user_dates=user_dates,
        user_counts=user_counts
    )


@admin_bp.route('/export-report')
@admin_required
def export_report():
    """Export usage report."""
    # Get date range
    start_date = request.args.get('start', (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%d'))
    end_date = request.args.get('end', datetime.utcnow().strftime('%Y-%m-%d'))
    
    # Query detections
    detections = Detection.query.filter(
        func.date(Detection.detected_at) >= start_date,
        func.date(Detection.detected_at) <= end_date
    ).all()
    
    # Create CSV
    csv_data = "Detection ID,User Email,User Name,Disease,Plant Type,Confidence,Date\n"
    for d in detections:
        csv_data += f"{d.id},{d.user.email},{d.user.name},{d.disease_name},{d.plant_type},{d.confidence:.1f},{d.detected_at.strftime('%Y-%m-%d %H:%M')}\n"
    
    buffer = io.BytesIO()
    buffer.write(csv_data.encode('utf-8'))
    buffer.seek(0)
    
    filename = f"usage_report_{start_date}_to_{end_date}.csv"
    return send_file(
        buffer,
        mimetype='text/csv',
        as_attachment=True,
        download_name=filename
    )


@admin_bp.route('/logs')
@admin_required
def admin_logs():
    """View admin activity logs."""
    page = request.args.get('page', 1, type=int)
    
    pagination = AdminLog.query.order_by(desc(AdminLog.created_at))\
        .paginate(page=page, per_page=50, error_out=False)
    
    return render_template(
        'admin/logs.html',
        title='Admin Logs',
        logs=pagination.items,
        pagination=pagination
    )


@admin_bp.route('/settings')
@admin_required
def settings():
    """Admin settings page."""
    return render_template(
        'admin/settings.html',
        title='Settings'
    )
