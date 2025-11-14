"""
HuggingFace Spaces entry point
This file is used when deploying to HuggingFace Spaces
"""

from main import app

# HuggingFace Spaces expects the app to be available
__all__ = ["app"]

