from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

class ApiEndpoint(BaseModel):
    method: str
    path: str
    description: str
    payload: Optional[Dict[str, Any]] = None

class ScriptConfig(BaseModel):
    """Minimal config - only stores what frontend provides"""
    rag_context: Dict[str, Any] = Field(default_factory=lambda: {"enabled": False, "documents": [], "description": ""})
    api_endpoints: Optional[List[ApiEndpoint]] = Field(default_factory=list)
