"""
External API Client
Handles calls to the developer API service (deployed separately)
"""

import os
import httpx
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Base URL for the external developer API service
# Set this via environment variable or default to localhost for development
EXTERNAL_API_BASE_URL = os.getenv("EXTERNAL_API_BASE_URL", "http://localhost:8001")

async def call_external_api(method: str, path: str, payload: Optional[Dict] = None) -> Dict:
    """
    Call an external API endpoint
    
    Args:
        method: HTTP method (GET, POST, etc.)
        path: API path (e.g., "/customer/submit")
        payload: Optional payload for POST/PUT requests
    
    Returns:
        Response data as dictionary
    """
    url = f"{EXTERNAL_API_BASE_URL}{path}"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            if method.upper() == "GET":
                response = await client.get(url)
            elif method.upper() == "POST":
                response = await client.post(url, json=payload or {})
            elif method.upper() == "PUT":
                response = await client.put(url, json=payload or {})
            elif method.upper() == "DELETE":
                response = await client.delete(url)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.error(f"External API call failed: {method} {url} - {e}")
        return {"error": str(e), "ok": False}
    except Exception as e:
        logger.error(f"Unexpected error calling external API: {e}")
        return {"error": str(e), "ok": False}

