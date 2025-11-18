from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os, json
import logging
from typing import List, Dict
from app.rag import ingest_documents, search_with_threshold, delete_document
from app.utils.validation import validate_file, sanitize_filename, MAX_FILE_SIZE

router = APIRouter()
logger = logging.getLogger(__name__)

DEFAULT_COMPANY = "default"

@router.post("/api/kb/upload")
async def upload(files: List[UploadFile] = File(...)):
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
        
        # Read file content to check size
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
    
    # Ingest documents - returns doc_id_map (filename -> doc_id)
    doc_id_map, chunks = ingest_documents(DEFAULT_COMPANY, saved_paths)
    
    # Update configs/script.json rag_context.documents with uploaded files
    # Documents are stored as objects: {"filename": "...", "doc_id": "..."}
    try:
        cfg_path = os.path.join("configs", "script.json")
        os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
        
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        else:
            cfg = {"rag_context": {"enabled": True, "documents": [], "description": ""}, "api_endpoints": []}
        
        # Get existing documents (handle both old format [strings] and new format [objects])
        existing_docs = cfg.get("rag_context", {}).get("documents") or []
        # Convert old format to new format if needed
        existing_docs_dict = {}
        for doc in existing_docs:
            if isinstance(doc, str):
                # Old format: just filename string
                existing_docs_dict[doc] = {"filename": doc, "doc_id": None}
            elif isinstance(doc, dict):
                # New format: object with filename and doc_id
                existing_docs_dict[doc.get("filename", "")] = doc
        
        # Add new documents
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
        
        if "rag_context" not in cfg:
            cfg["rag_context"] = {"enabled": rag_enabled, "documents": updated_docs, "description": ""}
        else:
            cfg["rag_context"]["documents"] = updated_docs
            cfg["rag_context"]["enabled"] = rag_enabled  # Update enabled status
        
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Updated script.json with {len(updated_docs)} documents, RAG enabled: {rag_enabled}")
    except Exception as e:
        logger.error(f"Failed to update script.json: {e}", exc_info=True)
        # Fallback: return just the new documents
        updated_docs = [{"filename": filename, "doc_id": doc_id} for filename, doc_id in doc_id_map.items()]
    
    return JSONResponse({
        "doc_id_map": doc_id_map,
        "chunks_added": chunks,
        "documents": updated_docs
    })

@router.delete("/api/kb/document/{filename}")
async def delete_document_endpoint(filename: str):
    """Delete a document by filename"""
    try:
        # Load script.json to get doc_id
        cfg_path = os.path.join("configs", "script.json")
        if not os.path.exists(cfg_path):
            return JSONResponse({"ok": False, "error": "script.json not found"}, status_code=404)
        
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        
        documents = cfg.get("rag_context", {}).get("documents", [])
        
        # Find the document and its doc_id
        doc_to_delete = None
        doc_id = None
        for doc in documents:
            if isinstance(doc, str):
                # Old format
                if doc == filename:
                    doc_to_delete = doc
                    # For old format, we need to find doc_id from vector store
                    # This is a fallback - ideally all docs should have doc_id
                    break
            elif isinstance(doc, dict):
                # New format
                if doc.get("filename") == filename:
                    doc_to_delete = doc
                    doc_id = doc.get("doc_id")
                    break
        
        if doc_to_delete is None:
            return JSONResponse({"ok": False, "error": f"Document '{filename}' not found"}, status_code=404)
        
        # If doc_id is None (old format), try to find it from vector store
        if doc_id is None:
            # We can't easily find doc_id from filename alone in Chroma
            # So we'll delete by source_path instead
            source_path = os.path.join("storage", f"company_{DEFAULT_COMPANY}", "raw", filename)
            logger.warning(f"Document '{filename}' has no doc_id, attempting to delete by source_path")
            # For now, we'll still try to delete from vector store using filename metadata
            # This is a limitation - ideally all documents should have doc_id
            return JSONResponse({
                "ok": False,
                "error": "Document has no doc_id. Please re-upload the document to get a doc_id."
            }, status_code=400)
        
        # Delete from vector store
        success = delete_document(DEFAULT_COMPANY, doc_id)
        if not success:
            return JSONResponse({"ok": False, "error": "Failed to delete from vector store"}, status_code=500)
        
        # Remove from script.json
        documents = [d for d in documents if (
            (isinstance(d, str) and d != filename) or
            (isinstance(d, dict) and d.get("filename") != filename)
        )]
        cfg["rag_context"]["documents"] = documents
        
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        
        # Delete the raw file
        file_path = os.path.join("storage", f"company_{DEFAULT_COMPANY}", "raw", filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Deleted raw file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete raw file {file_path}: {e}")
        
        logger.info(f"Successfully deleted document '{filename}' (doc_id={doc_id})")
        return JSONResponse({"ok": True, "filename": filename, "doc_id": doc_id})
        
    except Exception as e:
        logger.error(f"Error deleting document '{filename}': {e}", exc_info=True)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@router.get("/api/kb/documents")
async def list_documents():
    """Get list of all uploaded documents"""
    try:
        cfg_path = os.path.join("configs", "script.json")
        if not os.path.exists(cfg_path):
            return JSONResponse({"documents": []})
        
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        
        documents = cfg.get("rag_context", {}).get("documents", [])
        # Ensure all documents are in the new format (objects with filename and doc_id)
        formatted_docs = []
        for doc in documents:
            if isinstance(doc, str):
                # Old format: convert to new format
                formatted_docs.append({"filename": doc, "doc_id": None})
            elif isinstance(doc, dict):
                # New format: use as-is
                formatted_docs.append(doc)
        
        return JSONResponse({"documents": formatted_docs})
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        return JSONResponse({"documents": [], "error": str(e)}, status_code=500)

