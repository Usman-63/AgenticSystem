"""
Shared validation utilities for file uploads and API endpoints
"""
import os
import re
from typing import Tuple
from fastapi import UploadFile

# File upload constraints
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.md', '.docx', '.doc'}

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal and other security issues"""
    # Remove path components
    filename = os.path.basename(filename)
    # Remove any non-alphanumeric characters except dots, dashes, and underscores
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255-len(ext)] + ext
    return filename

def validate_file(file: UploadFile) -> Tuple[bool, str]:
    """Validate file size, type, and filename"""
    # Check file extension
    if not file.filename:
        return False, "Filename is required"
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"File type {ext} not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
    
    # Filename will be sanitized when saving
    return True, ""

def validate_endpoint_path(path: str) -> Tuple[bool, str]:
    """Validate API endpoint path format"""
    if not path:
        return False, "Path is required"
    if not path.startswith('/'):
        return False, "Path must start with '/'"
    # Allow alphanumeric, slashes, dashes, underscores, and path parameters {param}
    if not re.match(r'^/[\w{}/-]*$', path):
        return False, "Path contains invalid characters"
    return True, ""

