"""
Developer API Service
This service mimics the APIs that developers would implement for their companies.
Deploy this separately (e.g., on HuggingFace Space) and configure the main agentic system to call it.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List
from datetime import datetime
import logging
import uuid

app = FastAPI(title="Developer API Service", version="1.0.0")
logger = logging.getLogger(__name__)

# In-memory storage for customer submissions
customer_submissions: Dict[str, Dict] = {}

# Enable CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health and utility endpoints
@app.get("/health")
async def health():
    """Health check endpoint"""
    return JSONResponse({"status": "ok"})

@app.get("/ping")
async def ping():
    """Ping endpoint for testing"""
    return JSONResponse({"pong": True, "ts": datetime.utcnow().isoformat() + "Z"})

@app.get("/version")
async def version():
    """Version information"""
    return JSONResponse({"version": "1.0.0", "build": "dev", "service": "developer-api"})

@app.post("/echo")
async def echo(payload: Dict):
    """Echo endpoint - returns the payload for testing"""
    return JSONResponse({"echo": payload, "received_at": datetime.utcnow().isoformat() + "Z"})

# Authentication endpoints
@app.post("/auth/login")
async def auth_login(payload: Dict):
    """Dummy login endpoint"""
    username = str(payload.get("username") or "user")
    token = f"token-{username}-dummy"
    return JSONResponse({"token": token, "user": {"username": username}})

# Order management endpoints
@app.get("/orders/{order_id}")
async def get_order(order_id: str):
    """Get order by ID"""
    return JSONResponse({
        "order_id": order_id,
        "status": "processing",
        "items": ["Widget A", "Widget B"],
        "total": 19.98
    })

@app.post("/orders/preview")
async def order_preview(payload: Dict):
    """Preview order with pricing"""
    items: List[str] = payload.get("items") or []
    if not isinstance(items, list):
        return JSONResponse({"ok": False, "error": "items must be a list"}, status_code=400)
    price_each = 9.99
    total = round(price_each * len(items), 2)
    return JSONResponse({
        "ok": True,
        "items": items,
        "price_each": price_each,
        "total": total
    })

# Customer management endpoints
@app.post("/customer/submit")
async def customer_submit(payload: Dict):
    """Submit customer order/information - stores data and returns ID"""
    logger.info("Received customer submission keys=%s", list(payload.keys()))
    
    # Generate unique ID
    submission_id = str(uuid.uuid4())
    
    # Store submission with metadata
    submission_data = {
        "id": submission_id,
        "data": payload,
        "submitted_at": datetime.utcnow().isoformat() + "Z"
    }
    customer_submissions[submission_id] = submission_data
    
    logger.info("Stored submission with ID: %s", submission_id)
    return JSONResponse({
        "ok": True,
        "id": submission_id,
        "received": payload,
        "received_at": submission_data["submitted_at"]
    })

@app.get("/customer/{submission_id}")
async def get_customer_submission(submission_id: str):
    """Get customer submission by ID"""
    if submission_id not in customer_submissions:
        raise HTTPException(status_code=404, detail="Submission not found")
    
    return JSONResponse({
        "ok": True,
        **customer_submissions[submission_id]
    })

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return JSONResponse({
        "service": "Developer API",
        "description": "Mock API endpoints for testing agentic systems",
        "endpoints": {
            "health": "/health",
            "ping": "/ping",
            "version": "/version",
            "echo": "POST /echo",
            "auth": "POST /auth/login",
            "orders": "GET /orders/{order_id}, POST /orders/preview",
            "customer": "POST /customer/submit, GET /customer/{submission_id}"
        }
    })

