from fastapi import APIRouter, Request
import logging
import threading
from fastapi.responses import JSONResponse
from typing import Dict, Optional
import os, json
from app.models import ScriptConfig
from app.script import sanitize_reply, parse_search_kb_tag, parse_api_call_tag
from app.rag import search_with_threshold
from app.services.together_client import call_llm
from app.services.external_api_client import call_external_api

router = APIRouter()
logger = logging.getLogger(__name__)

# Cached script config with file modification time checking
_script_cache: Optional[ScriptConfig] = None
_script_cache_time: float = 0
_script_cache_path: Optional[str] = None
_script_cache_lock = threading.Lock()

def _load_default_script() -> ScriptConfig:
    """Load script config with caching based on file modification time"""
    global _script_cache, _script_cache_time, _script_cache_path
    
    path = os.path.join(os.getcwd(), "configs", "script.json")
    
    # Check if file exists and get modification time
    if not os.path.exists(path):
        # Return empty config if setup hasn't been completed yet
        _script_cache = ScriptConfig()
        _script_cache_time = 0
        _script_cache_path = path
        return _script_cache
    
    mtime = os.path.getmtime(path)
    
    # Check if cache is valid
    if (_script_cache is None or 
        _script_cache_path != path or 
        mtime > _script_cache_time):
        
        with _script_cache_lock:
            # Double-check after acquiring lock
            if (_script_cache is None or 
                _script_cache_path != path or 
                mtime > _script_cache_time):
                
                logger.debug(f"Loading script config from {path} (mtime: {mtime})")
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                _script_cache = ScriptConfig(**data)
                _script_cache_time = mtime
                _script_cache_path = path
                logger.debug("Script config cached")
    
    return _script_cache

# removed CID-specific endpoints for testing simplicity
@router.post("/scripted_chat")
async def scripted_chat_default(req: Request):
    body = await req.json()
    payload = body if isinstance(body, dict) else {}
    
    logger.info("SCRIPTED_CHAT REQUEST RECEIVED")
    logger.debug("Body type: %s", type(body))
    logger.debug("Body keys: %s", list(body.keys()) if isinstance(body, dict) else "Not a dict")
    logger.debug("Payload type: %s", type(payload))
    logger.debug("Payload keys: %s", list(payload.keys()) if isinstance(payload, dict) else "Not a dict")
    logger.debug("Raw payload: %s", json.dumps(payload, indent=2))
    logger.debug("History key exists: %s", "history" in payload if isinstance(payload, dict) else False)
    logger.debug("History value directly: %s", payload.get("history") if isinstance(payload, dict) else "N/A")
    
    fresh = _load_default_script()
    script = fresh.model_dump()
    turn = 0
    try:
        turn = int(payload.get("turn", 0))
    except Exception:
        turn = 0
    logger.debug("Turn number: %d", turn)
    logger.debug("User content: %s", payload.get("content", ""))
    # Build system prompt from raw script + documents list, and nothing else
    messages = []
    raw_path = os.path.join(os.getcwd(), "simpleScript.txt")
    raw_script = ""
    if os.path.exists(raw_path):
        with open(raw_path, "r", encoding="utf-8") as f:
            raw_script = f.read()
        logger.debug("Raw script loaded: %d characters", len(raw_script))
    else:
        logger.warning("Raw script not found at: %s", raw_path)
    docs = script.get("rag_context", {}).get("documents") or []
    logger.debug("RAG documents: %s", docs)
    doc_text = ""
    if docs:
        # Handle both old format (strings) and new format (objects with filename and doc_id)
        doc_names = []
        for d in docs:
            if isinstance(d, str):
                doc_names.append(d)
            elif isinstance(d, dict):
                doc_names.append(d.get("filename", ""))
        doc_lines = ["\nDocuments available:"] + [f"- {d}" for d in doc_names if d]
        doc_text = "\n" + "\n".join(doc_lines)
    endpoints = script.get("api_endpoints") or []
    logger.debug("API endpoints count: %d", len(endpoints) if isinstance(endpoints, list) else 0)
    ep_text = ""
    if isinstance(endpoints, list) and endpoints:
        ep_lines = ["\nAPI endpoints available:"]
        for e in endpoints:
            method = e.get('method', 'GET')
            path = e.get('path', '')
            desc = e.get('description', '')
            ep_lines.append(f"- {method} {path}: {desc}")
            endpoint_payload_schema = e.get('payload')
            if endpoint_payload_schema and isinstance(endpoint_payload_schema, dict) and endpoint_payload_schema:
                # Show compact payload structure as JSON for the LLM
                payload_json = json.dumps(endpoint_payload_schema, separators=(',', ':'))
                ep_lines.append(f"  Payload schema: {payload_json}")
        ep_text = "\n" + "\n".join(ep_lines)
    # Get configurable prompt components from script config, with defaults
    intro = "You are a HUMAN assistant. Greet the user once, then proceed with concise, clear answers."
    grounding = script.get("grounding_rules") or (
        "\nGrounding Rules:\n"
        "- Do not Respond to user with AI, Speak with emotions"
        "- Only use information from the RAW SCRIPT and the listed documents.\n"
        "- Do not invent facts; if not covered, respond: 'I don't have that information yet.' and ask a brief clarifying question.\n"
        "- When you need the knowledge base, respond only with [SEARCH_KB: 'reformulated question'].\n"
        "- After using the KB, answer briefly.\n"
        "- No special formatting; keep responses under two short paragraphs.\n"
    )
    kb_instr = script.get("kb_instructions") or (
        "\nKnowledge Base Search Instructions:\n"
        "- If the user's question is not covered by the script, respond only with: [SEARCH_KB: 'reformulated question']\n"
        "- Example: [SEARCH_KB: 'refund policy']\n"
        "- Do not include any other text with [SEARCH_KB]."
    )
    api_instr = script.get("api_instructions") or (
        "\nAPI Call Instructions:\n"
        "- To use an API, respond only with: [API_CALL: 'METHOD /path', {payload}]\n"
        "- Examples:\n"
        "  [API_CALL: 'GET /api/ping']\n"
        "- Do not include other text with [API_CALL]."
    )
    system_only = intro + doc_text + ep_text + grounding + kb_instr + api_instr + "\n--RAW\n" + raw_script
    logger.debug("System prompt length: %d characters", len(system_only))
    
    messages.append({"role": "system", "content": system_only})
    hist = payload.get("history", [])
    logger.debug("History from payload: %d messages", len(hist) if isinstance(hist, list) else 0)
    
    if isinstance(hist, list):
        logger.debug("Processing %d history messages", len(hist))
        for i, h in enumerate(hist[-20:]):  # Process up to 20 (will be limited by backend)
            r = h.get("role")
            c = h.get("content")
            if r in ("user", "assistant") and c:
                messages.append({"role": r, "content": c})
            else:
                logger.debug("  -> SKIPPED (role=%s, has_content=%s)", r, bool(c))
    else:
        logger.warning("History is not a list! Type: %s, Value: %s", type(hist), hist)
    
    user_content = payload.get("content", "")
    messages.append({"role": "user", "content": user_content})
    
    # Enforce history limit on backend (max 20 messages in history, excluding system and current user)
    # The frontend may send more, but we enforce the limit here
    MAX_HISTORY = 20
    if len(messages) > MAX_HISTORY + 2:  # +2 for system and current user message
        # Keep system message, current user message, and last MAX_HISTORY messages
        system_msg = messages[0]
        user_msg = messages[-1]
        history_msgs = messages[1:-1][-MAX_HISTORY:]
        messages = [system_msg] + history_msgs + [user_msg]
        logger.debug(f"History limit enforced: reduced to {len(messages)} messages (max history: {MAX_HISTORY})")
    
    logger.debug("Total messages to LLM: %d", len(messages))
    logger.debug("Message structure: system=%d, history=%d, user=1", 1, len(messages) - 2)
    
    logger.debug("Calling LLM...")
    raw = call_llm(messages)
    logger.debug("LLM raw response length: %d characters", len(raw) if raw else 0)
    logger.debug("LLM raw response: %s", raw[:500] if raw else "None")
    
    reply = sanitize_reply(raw)
    logger.info("Sanitized reply: %s", reply[:500] if reply else "None")
    api_call = parse_api_call_tag(reply)
    logger.debug("API call parse result: %s", api_call if api_call else "None")
    
    if api_call:
        method = api_call.get("method") or "GET"
        path = api_call.get("path") or ""
        api_payload = api_call.get("payload") or {}
        logger.info(">>> API CALL DETECTED: %s %s", method, path)
        logger.debug("API payload: %s", json.dumps(api_payload) if api_payload else "None")
        
        # Remove /api prefix if present (external API doesn't need it)
        if path.startswith("/api/"):
            path = path[4:]  # Remove "/api" prefix
        
        # Call external developer API service
        result = await call_external_api(method, path, api_payload)
        logger.debug("External API result: %s", json.dumps(result)[:200] if result else "None")
        format_prompt = (
            f"The API call was: {method} {path}. "
            f"The API returned: {json.dumps(result)}. "
            f"Formulate a friendly, human response based on the API result."
        )
        fm = [{"role": "system", "content": format_prompt}, {"role": "user", "content": api_payload if isinstance(api_payload, str) else json.dumps(api_payload)}]
        final_reply = sanitize_reply(call_llm(fm))
        logger.debug("API response: %s", final_reply[:200] if final_reply else "None")
        return JSONResponse({"reply": final_reply, "api": {"method": method, "path": path, "result": result}})
    
    search_query = parse_search_kb_tag(reply)
    logger.debug("KB search parse result: %s", search_query if search_query else "None")
    
    if search_query:
        logger.info(">>> KB SEARCH DETECTED: %s", search_query)
        hits = search_with_threshold("default", search_query)
        logger.debug("KB search returned %d hits", len(hits))
        docs = [d for (d, s) in hits]
        rag_answer = "\n".join([d.page_content for d in docs])
        logger.debug("KB answer length: %d characters", len(rag_answer))
        format_prompt = (
            f"The user asked: '{search_query}'. "
            f"The knowledge base found: '{rag_answer}'. "
            f"IMPORTANT: The information above came from the knowledge base, NOT from what the user said. "
            f"The user did NOT mention or provide this information. "
            f"Formulate a friendly, human response that presents this information as something you found or looked up, "
            f"not as something the user told you. Use phrases like 'I found', 'According to our records', "
            f"'Our knowledge base shows', or 'I can see that' instead of 'you have', 'you mentioned', or 'you said'. "
            f"Never attribute knowledge base information to the user."
        )
        fm = [{"role":"system","content": format_prompt}, {"role":"user","content": user_content }]
        logger.debug("Calling LLM for KB response formatting...")
        final_reply = sanitize_reply(call_llm(fm))
        # Format sources with backend-side formatting
        sources = []
        for doc, score in hits:
            source_path = doc.metadata.get("source_path", "")
            filename = os.path.basename(source_path) if source_path else "(unknown)"
            preview = doc.page_content[:200] if doc.page_content else ""
            # Format score to 4 decimal places on backend
            formatted_score = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
            sources.append({
                "source_path": source_path,
                "filename": filename,
                "score": formatted_score,
                "score_raw": score,
                "preview": preview
            })
        logger.debug("KB response: %s", final_reply[:200] if final_reply else "None")
        return JSONResponse({"reply": final_reply, "kb": {"query": search_query, "sources": sources}})
    
    logger.debug(">>> DIRECT REPLY (no API/KB)")
    logger.debug("Final reply: %s", reply[:500] if reply else "None")
    return JSONResponse({"reply": reply})

@router.get("/state")
async def load_default():
    return JSONResponse({})

@router.get("/state/history")
async def full_history_default():
    return JSONResponse([])


@router.get("/state/script")
async def get_script_default():
    fresh = _load_default_script()
    return JSONResponse(fresh.model_dump())

@router.post("/state/script/reload")
async def reload_script_default():
    fresh = _load_default_script()
    return JSONResponse({"ok": True, "script": fresh.model_dump()})

@router.get("/state/company")
async def get_company_default():
    return JSONResponse({"company_id": "default"})

@router.post("/state/company")
async def set_company_default(payload: Dict):
    return JSONResponse({"company_id": "default"})

@router.get("/ping")
async def ping():
    """Health check / ping endpoint"""
    return JSONResponse({"status": "ok", "message": "pong"})
