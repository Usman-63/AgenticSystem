import os
import logging
import threading
import time
from typing import Optional
from dotenv import load_dotenv
from together import Together

logger = logging.getLogger(__name__)

load_dotenv()
MODEL = os.getenv("TOGETHER_MODEL", "Qwen/QwQ-32B")
TOGETHER_TIMEOUT = int(os.getenv("TOGETHER_TIMEOUT", "60"))  # Default 60 seconds

# Cached Together client (singleton pattern)
_together_client: Optional[Together] = None
_client_lock = threading.Lock()

def get_together_client() -> Together:
    """Get or create cached Together client instance"""
    global _together_client
    if _together_client is None:
        with _client_lock:
            # Double-check after acquiring lock
            if _together_client is None:
                api_key = os.getenv("TOGETHER_API_KEY")
                if not api_key:
                    raise RuntimeError("Missing TOGETHER_API_KEY in .env")
                logger.info("Creating Together AI client")
                _together_client = Together(api_key=api_key, timeout=TOGETHER_TIMEOUT)
                logger.info("Together AI client created")
    return _together_client

def call_llm(messages, max_retries: int = 2, retry_delay: float = 1.0):
    """
    Call Together AI LLM with retry logic and error handling.
    
    Args:
        messages: List of message dicts for the LLM
        max_retries: Maximum number of retries on failure (default: 2)
        retry_delay: Initial delay between retries in seconds (default: 1.0, exponential backoff)
    
    Returns:
        LLM response text
    
    Raises:
        RuntimeError: If all retries fail or API key is missing
    """
    client = get_together_client()
    
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            start_time = time.time()
            res = client.chat.completions.create(
                model=MODEL,
                messages=messages,
            )
            elapsed = time.time() - start_time
            
            if not res or not res.choices:
                logger.error("LLM call failed: no response")
                raise RuntimeError("LLM call failed: no response")
            
            content = res.choices[0].message.content
            logger.debug(f"LLM call succeeded in {elapsed:.2f}s (attempt {attempt + 1})")
            return content
            
        except Exception as e:
            last_error = e
            error_msg = str(e)
            
            # Check if it's a network/DNS error
            if "getaddrinfo failed" in error_msg or "NameResolutionError" in error_msg:
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries + 1}): DNS/Network error - {error_msg}")
            elif "timeout" in error_msg.lower():
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries + 1}): Timeout - {error_msg}")
            else:
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries + 1}): {error_msg}")
            
            # Don't retry on last attempt
            if attempt < max_retries:
                delay = retry_delay * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying LLM call in {delay:.1f}s...")
                time.sleep(delay)
            else:
                logger.error(f"LLM call failed after {max_retries + 1} attempts")
                raise RuntimeError(f"LLM call failed after {max_retries + 1} attempts: {error_msg}") from last_error
    
    # Should never reach here, but just in case
    raise RuntimeError(f"LLM call failed: {last_error}") from last_error
