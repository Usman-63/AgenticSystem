"""
Text processing utilities for cleaning and formatting text.
"""
import re


def clean_text_for_tts(text: str) -> str:
    """
    Clean markdown formatting from text before TTS synthesis.
    Removes asterisks, bold markers, and other markdown that TTS would read aloud.
    
    Examples:
        "**bold** text" -> "bold text"
        "* Your first name" -> "Your first name"
        "[link](url)" -> "link"
    
    Args:
        text: The text to clean
        
    Returns:
        Cleaned text without markdown formatting
    """
    if not text:
        return text
    
    # Remove markdown bold/italic markers (**text**, *text*, __text__, _text_)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *italic*
    text = re.sub(r'__([^_]+)__', r'\1', text)      # __bold__
    text = re.sub(r'_([^_]+)_', r'\1', text)        # _italic_
    
    # Remove standalone asterisks (bullet points, etc.)
    text = re.sub(r'^\s*\*\s+', '', text, flags=re.MULTILINE)  # * at start of line
    text = re.sub(r'\s*\*\s+', ' ', text)  # * in middle of text
    
    # Remove markdown headers (# ## ###)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    
    # Remove markdown links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Remove markdown code blocks ```code``` -> code
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'\1', text)  # `code`
    
    # Clean up multiple spaces and newlines
    text = re.sub(r'\n+', ' ', text)  # Multiple newlines to single space
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
    
    return text.strip()

