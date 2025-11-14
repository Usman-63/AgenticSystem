from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import os
import json
import logging
from typing import List, Dict, Optional
from app.models import ScriptConfig, ApiEndpoint
from app.rag import ingest_documents

router = APIRouter()
logger = logging.getLogger(__name__)

DEFAULT_COMPANY = "default"

def _get_script_path() -> str:
    return os.path.join(os.getcwd(), "configs", "script.json")

def _get_raw_script_path() -> str:
    return os.path.join(os.getcwd(), "simpleScript.txt")

@router.post("/api/setup/script")
async def save_script(payload: Dict):
    """Save script - saves immediately when user enters it"""
    try:
        script_data = payload.get("script", {})
        raw_script = payload.get("raw_script", "")
        
        if not raw_script.strip():
            return JSONResponse({"ok": False, "error": "Raw script is required"}, status_code=400)
        
        # Load existing config or create new one
        script_path = _get_script_path()
        os.makedirs(os.path.dirname(script_path), exist_ok=True)
        
        if os.path.exists(script_path):
            with open(script_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        else:
            cfg = script_data if script_data else {"rag_context": {"enabled": False, "documents": [], "description": ""}, "api_endpoints": []}
        
        # Merge with existing config (preserve files and endpoints if already saved)
        if "rag_context" in script_data:
            cfg["rag_context"] = script_data["rag_context"]
        if "api_endpoints" not in cfg:
            cfg["api_endpoints"] = []
        
        # Validate and save
        script_config = ScriptConfig(**cfg)
        with open(script_path, "w", encoding="utf-8") as f:
            json.dump(script_config.model_dump(), f, ensure_ascii=False, indent=2)
        
        # Save raw script to simpleScript.txt
        raw_path = _get_raw_script_path()
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(raw_script)
        
        logger.info("Script saved")
        return JSONResponse({"ok": True, "message": "Script saved"})
    except Exception as e:
        logger.error(f"Failed to save script: {e}")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

@router.post("/api/setup/files")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload files during setup"""
    try:
        saved_paths = []
        base = os.path.join("storage", f"company_{DEFAULT_COMPANY}", "raw")
        os.makedirs(base, exist_ok=True)
        
        for f in files:
            path = os.path.join(base, f.filename)
            with open(path, "wb") as out:
                out.write(await f.read())
            saved_paths.append(path)
        
        # Ingest documents into vector store
        doc_ids, chunks = ingest_documents(DEFAULT_COMPANY, saved_paths)
        
        # Update script.json with document names - files are saved immediately when uploaded
        script_path = _get_script_path()
        os.makedirs(os.path.dirname(script_path), exist_ok=True)
        
        if os.path.exists(script_path):
            with open(script_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        else:
            cfg = {"rag_context": {"enabled": True, "documents": [], "description": ""}, "api_endpoints": []}
        
        if "rag_context" not in cfg:
            cfg["rag_context"] = {"enabled": True, "documents": [], "description": ""}
        
        docs = cfg["rag_context"].get("documents", [])
        names = [os.path.basename(p) for p in saved_paths]
        for n in names:
            if n not in docs:
                docs.append(n)
        cfg["rag_context"]["documents"] = docs
        
        with open(script_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Uploaded {len(saved_paths)} files, {chunks} chunks added")
        return JSONResponse({
            "ok": True,
            "files": names,
            "chunks_added": chunks
        })
    except Exception as e:
        logger.error(f"Failed to upload files: {e}")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

@router.post("/api/setup/endpoints")
async def save_endpoints(payload: Dict):
    """Save API endpoints configuration"""
    try:
        endpoints_data = payload.get("endpoints", [])
        endpoints = [ApiEndpoint(**ep) for ep in endpoints_data]
        
        # Update script.json
        script_path = _get_script_path()
        os.makedirs(os.path.dirname(script_path), exist_ok=True)
        
        if os.path.exists(script_path):
            with open(script_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        else:
            # Create minimal config if it doesn't exist
            cfg = {
                "rag_context": {"enabled": False, "documents": [], "description": ""},
                "api_endpoints": []
            }
        
        cfg["api_endpoints"] = [ep.model_dump() for ep in endpoints]
        
        with open(script_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(endpoints)} API endpoints")
        return JSONResponse({"ok": True, "endpoints": len(endpoints)})
    except Exception as e:
        logger.error(f"Failed to save endpoints: {e}")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@router.get("/api/setup/status")
async def get_setup_status():
    """Check if setup has been completed"""
    script_path = _get_script_path()
    exists = os.path.exists(script_path)
    return JSONResponse({"configured": exists})

@router.get("/api/setup/raw-script")
async def get_raw_script():
    """Get the raw script content"""
    raw_path = _get_raw_script_path()
    if os.path.exists(raw_path):
        with open(raw_path, "r", encoding="utf-8") as f:
            content = f.read()
        from fastapi.responses import Response
        return Response(content=content, media_type="text/plain")
    return JSONResponse({"error": "Raw script not found"}, status_code=404)

