from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import os, json
from typing import List, Dict
from app.rag import ingest_documents, search_with_threshold
from app.services.together_client import call_llm

router = APIRouter()

DEFAULT_COMPANY = "default"

@router.post("/api/kb/upload")
async def upload(files: List[UploadFile] = File(...)):
    saved_paths = []
    base = os.path.join("storage", f"company_{DEFAULT_COMPANY}", "raw")
    os.makedirs(base, exist_ok=True)
    for f in files:
        path = os.path.join(base, f.filename)
        with open(path, "wb") as out:
            out.write(await f.read())
        saved_paths.append(path)
    doc_ids, chunks = ingest_documents(DEFAULT_COMPANY, saved_paths)
    # Update configs/script.json rag_context.documents with uploaded filenames
    try:
        cfg_path = os.path.join("configs", "script.json")
        os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
        
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        else:
            cfg = {"rag_context": {"enabled": True, "documents": [], "description": ""}, "api_endpoints": []}
        
        docs = cfg.get("rag_context", {}).get("documents") or []
        names = [os.path.basename(p) for p in saved_paths]
        # ensure uniqueness while preserving order
        for n in names:
            if n not in docs:
                docs.append(n)
        if "rag_context" not in cfg:
            cfg["rag_context"] = {"enabled": True, "documents": docs, "description": ""}
        else:
            cfg["rag_context"]["documents"] = docs
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        updated_docs = docs
    except Exception as e:
        logger.error(f"Failed to update script.json: {e}")
        updated_docs = [os.path.basename(p) for p in saved_paths]
    return JSONResponse({"doc_ids": doc_ids, "chunks_added": chunks, "documents": updated_docs})

@router.post("/api/kb/query")
async def kb_query(payload: Dict):
    query = payload.get("query", "")
    persona = payload.get("persona", "")
    tone = payload.get("tone", "")
    org_name = payload.get("org_name", "")
    org_about = payload.get("about_organization") or payload.get("org_desc", "")
    hits = search_with_threshold(DEFAULT_COMPANY, query)
    docs = [d for (d, s) in hits]
    rag_answer = "\n".join([d.page_content for d in docs])
    format_prompt = (
        f"You are {persona} from {org_name}. "
        f"Use a {tone} voice. "
        f"Organization about: {org_about}. "
        f"The user asked: '{query}'. "
        f"The knowledge base found: '{rag_answer}'. "
        f"Formulate a friendly response."
    )
    msgs = [{"role":"system","content": format_prompt}, {"role":"user","content": query}]
    reply = call_llm(msgs)
    sources = [{"source_path": d.metadata.get("source_path"), "score": s, "preview": (d.page_content[:200] if d.page_content else "")} for (d, s) in hits]
    return JSONResponse({"reply": reply, "kb": {"query": query, "sources": sources}})
