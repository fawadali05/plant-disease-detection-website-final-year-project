"""
WTForms for the application
"""

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, TextAreaField, SelectField, FileField
from wtforms.validators import DataRequired, Email, Length, EqualTo, Optional, ValidationError
from wtforms import IntegerField, FloatField
from models import User


class RegistrationForm(FlaskForm):
    """User registration form."""
    name = StringField('Full Name', validators=[
        DataRequired(message='Name is required'),
        Length(min=2, max=100, message='Name must be between 2 and 100 characters')
    ])
    email = StringField('Email', validators=[
        DataRequired(message='Email is required'),
        Email(message='Please enter a valid email address')
    ])
    password = PasswordField('Password', validators=[
        DataRequired(message='Password is required'),
        Length(min=6, message='Password must be at least 6 characters')
    ])
    confirm_password = PasswordField('Confirm Password', validators=[
        DataRequired(message='Please confirm your password'),
        EqualTo('password', message='Passwords must match')
    ])
    
    def validate_email(self, field):
        """Check if email is already registered."""
        if User.query.filter_by(email=field.data.lower()).first():
            raise ValidationError('This email is already registered. Please use a different one.')


class LoginForm(FlaskForm):
    """User login form."""
    email = StringField('Email', validators=[
        DataRequired(message='Email is required'),
        Email(message='Please enter a valid email address')
    ])
    password = PasswordField('Password', validators=[
        DataRequired(message='Password is required')
    ])
    remember = SelectField('Remember Me', choices=[
        ('yes', 'Yes'),
        ('no', 'No')
    ], default='no')


class ProfileForm(FlaskForm):
    """User profile edit form."""
    name = StringField('Full Name', validators=[
        DataRequired(message='Name is required'),
        Length(min=2, max=100)
    ])
    email = StringField('Email', validators=[
        DataRequired(message='Email is required'),
        Email()
    ])


class ChangePasswordForm(FlaskForm):
    """Change password form."""
    current_password = PasswordField('Current Password', validators=[
        DataRequired(message='Current password is required')
    ])
    new_password = PasswordField('New Password', validators=[
        DataRequired(message='New password is required'),
        Length(min=6, message='Password must be at least 6 characters')
    ])
    confirm_password = PasswordField('Confirm New Password', validators=[
        DataRequired(),
        EqualTo('new_password', message='Passwords must match')
    ])


class ContactForm(FlaskForm):
    """Contact form."""
    name = StringField('Name', validators=[
        DataRequired(message='Name is required'),
        Length(min=2, max=100)
    ])
    email = StringField('Email', validators=[
        DataRequired(message='Email is required'),
        Email(message='Please enter a valid email address')
    ])
    subject = StringField('Subject', validators=[
        DataRequired(message='Subject is required'),
        Length(min=3, max=200)
    ])
    message = TextAreaField('Message', validators=[
        DataRequired(message='Message is required'),
        Length(min=10, max=2000)
    ])


class DiseaseForm(FlaskForm):
    """Form for adding/editing diseases."""
    disease_name = StringField('Disease Name', validators=[
        DataRequired(message='Disease name is required'),
        Length(min=2, max=100)
    ])
    plant_type = StringField('Plant Type', validators=[
        DataRequired(message='Plant type is required'),
        Length(min=2, max=50)
    ])
    category = SelectField('Category', choices=[
        ('fungal', 'Fungal'),
        ('bacterial', 'Bacterial'),
        ('viral', 'Viral'),
        ('pest', 'Pest/Insect'),
        ('healthy', 'Healthy'),
        ('other', 'Other')
    ])
    description = TextAreaField('Description', validators=[
        DataRequired(message='Description is required')
    ])
    symptoms = TextAreaField('Symptoms')
    causes = TextAreaField('Causes')
    treatment = TextAreaField('Treatment', validators=[
        DataRequired(message='Treatment is required')
    ])
    prevention = TextAreaField('Prevention')
    pesticides = TextAreaField('Recommended Pesticides')
    recovery_time = StringField('Recovery Time')
    severity = SelectField('Severity', choices=[
        ('low', 'Low'),
        ('moderate', 'Moderate'),
        ('high', 'High'),
        ('critical', 'Critical')
    ], default='moderate')
