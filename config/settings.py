import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key')
    
    # Upload settings
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'srt'}
    
    # API settings
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    
    # Rate limiting
    REQUESTS_PER_MINUTE = 12
    
    # Logging
    LOG_FOLDER = 'logs'
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Languages
    SUPPORTED_LANGUAGES = {
        'zh': 'Chinese',
        'en': 'English',
        'ja': 'Japanese',
        'ko': 'Korean',
        'vi': 'Vietnamese'
    }