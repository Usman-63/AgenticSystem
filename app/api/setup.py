from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import json
import logging
from typing import List, Dict, Optional
from app.models import ScriptConfig, ApiEndpoint
from app.rag import ingest_documents
from app.utils.validation import validate_file, sanitize_filename, validate_endpoint_path, MAX_FILE_SIZE

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
        # But let backend determine RAG enabled status based on actual document count
        if "rag_context" in script_data:
            # Load existing documents to determine enabled status
            existing_docs = cfg.get("rag_context", {}).get("documents", [])
            doc_count = len(existing_docs) if isinstance(existing_docs, list) else 0
            rag_enabled = doc_count > 0
            
            cfg["rag_context"] = script_data["rag_context"]
            cfg["rag_context"]["enabled"] = rag_enabled  # Backend decides
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
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        saved_paths = []
        base = os.path.join("storage", f"company_{DEFAULT_COMPANY}", "raw")
        os.makedirs(base, exist_ok=True)
        
        for f in files:
            # Validate file
            is_valid, error_msg = validate_file(f)
            if not is_valid:
                raise HTTPException(status_code=400, detail=f"Invalid file {f.filename}: {error_msg}")
            
            # Read and check size
            content = await f.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail=f"File {f.filename} exceeds maximum size of {MAX_FILE_SIZE / (1024*1024):.0f}MB")
            
            # Sanitize filename
            safe_filename = sanitize_filename(f.filename)
            path = os.path.join(base, safe_filename)
            
            # Save file
            with open(path, "wb") as out:
                out.write(content)
            saved_paths.append(path)
        
        # Ingest documents into vector store - returns doc_id_map (filename -> doc_id)
        doc_id_map, chunks = ingest_documents(DEFAULT_COMPANY, saved_paths)
        
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
        
        # Get existing documents (handle both old format [strings] and new format [objects])
        existing_docs = cfg["rag_context"].get("documents", [])
        existing_docs_dict = {}
        for doc in existing_docs:
            if isinstance(doc, str):
                existing_docs_dict[doc] = {"filename": doc, "doc_id": None}
            elif isinstance(doc, dict):
                existing_docs_dict[doc.get("filename", "")] = doc
        
        # Add new documents
        names = [os.path.basename(p) for p in saved_paths]
        for filename, doc_id in doc_id_map.items():
            if filename not in existing_docs_dict:
                existing_docs_dict[filename] = {"filename": filename, "doc_id": doc_id}
            else:
                # Update doc_id if it was None (migrating from old format)
                if existing_docs_dict[filename].get("doc_id") is None:
                    existing_docs_dict[filename]["doc_id"] = doc_id
        
        # Convert back to list, preserving order
        updated_docs = list(existing_docs_dict.values())
        
        # Backend decides if RAG should be enabled based on document count
        rag_enabled = len(updated_docs) > 0
        cfg["rag_context"]["documents"] = updated_docs
        cfg["rag_context"]["enabled"] = rag_enabled
        
        with open(script_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Uploaded {len(saved_paths)} files, {chunks} chunks added, RAG enabled: {rag_enabled}")
        return JSONResponse({
            "ok": True,
            "files": [{"filename": name, "doc_id": doc_id_map.get(name)} for name in names],
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
        if not isinstance(endpoints_data, list):
            raise HTTPException(status_code=400, detail="endpoints must be a list")
        
        # Validate each endpoint
        valid_methods = {'GET', 'POST', 'PUT', 'DELETE', 'PATCH'}
        endpoints = []
        for idx, ep_data in enumerate(endpoints_data):
            if not isinstance(ep_data, dict):
                raise HTTPException(status_code=400, detail=f"Endpoint at index {idx} must be an object")
            
            method = ep_data.get("method", "").upper()
            if method not in valid_methods:
                raise HTTPException(status_code=400, detail=f"Invalid method '{method}' at index {idx}. Must be one of: {', '.join(valid_methods)}")
            
            path = ep_data.get("path", "")
            is_valid, error_msg = validate_endpoint_path(path)
            if not is_valid:
                raise HTTPException(status_code=400, detail=f"Invalid path at index {idx}: {error_msg}")
            
            # Validate payload is a dict if provided
            payload_data = ep_data.get("payload")
            if payload_data is not None and not isinstance(payload_data, dict):
                raise HTTPException(status_code=400, detail=f"Payload at index {idx} must be an object or null")
            
            endpoints.append(ApiEndpoint(**ep_data))
        
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

