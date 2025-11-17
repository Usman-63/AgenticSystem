from fastapi import APIRouter, Request
import logging
from fastapi.responses import JSONResponse
from typing import Dict
import os, json
from app.models import ScriptConfig
from app.script import sanitize_reply, parse_search_kb_tag, parse_api_call_tag
from app.rag import search_with_threshold
from app.services.together_client import call_llm
from app.services.external_api_client import call_external_api

router = APIRouter()
logger = logging.getLogger(__name__)

def _load_default_script() -> ScriptConfig:
    path = os.path.join(os.getcwd(), "configs", "script.json")
    if not os.path.exists(path):
        # Return empty config if setup hasn't been completed yet
        return ScriptConfig()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ScriptConfig(**data)

# removed CID-specific endpoints for testing simplicity
@router.post("/scripted_chat")
async def scripted_chat_default(req: Request):
    body = await req.json()
    payload = body if isinstance(body, dict) else {}
    
    logger.info("=" * 80)
    logger.info("SCRIPTED_CHAT REQUEST RECEIVED")
    logger.info("=" * 80)
    logger.info("Body type: %s", type(body))
    logger.info("Body keys: %s", list(body.keys()) if isinstance(body, dict) else "Not a dict")
    logger.info("Payload type: %s", type(payload))
    logger.info("Payload keys: %s", list(payload.keys()) if isinstance(payload, dict) else "Not a dict")
    logger.info("Raw payload: %s", json.dumps(payload, indent=2))
    logger.info("History key exists: %s", "history" in payload if isinstance(payload, dict) else False)
    logger.info("History value directly: %s", payload.get("history") if isinstance(payload, dict) else "N/A")
    
    fresh = _load_default_script()
    script = fresh.model_dump()
    turn = 0
    try:
        turn = int(payload.get("turn", 0))
    except Exception:
        turn = 0
    logger.info("Turn number: %d", turn)
    logger.info("User content: %s", payload.get("content", ""))
    # Build system prompt from raw script + documents list, and nothing else
    messages = []
    raw_path = os.path.join(os.getcwd(), "simpleScript.txt")
    raw_script = ""
    if os.path.exists(raw_path):
        with open(raw_path, "r", encoding="utf-8") as f:
            raw_script = f.read()
        logger.info("Raw script loaded: %d characters", len(raw_script))
        logger.debug("Raw script preview (first 200 chars): %s", raw_script[:200])
    else:
        logger.warning("Raw script not found at: %s", raw_path)
    docs = script.get("rag_context", {}).get("documents") or []
    logger.info("RAG documents: %s", docs)
    doc_text = ""
    if docs:
        doc_lines = ["\nDocuments available:"] + [f"- {d}" for d in docs]
        doc_text = "\n" + "\n".join(doc_lines)
    endpoints = script.get("api_endpoints") or []
    logger.info("API endpoints count: %d", len(endpoints) if isinstance(endpoints, list) else 0)
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
    intro = script.get("intro_text") or "You are a HUMAN assistant. Greet the user once, then proceed with concise, clear answers."
    grounding = script.get("grounding_rules") or (
        "\nGrounding Rules:\n"
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
    logger.info("System prompt length: %d characters", len(system_only))
    logger.debug("System prompt preview (first 500 chars):\n%s", system_only[:500])
    
    messages.append({"role": "system", "content": system_only})
    hist = payload.get("history", [])
    logger.info("History from payload: %s", hist)
    logger.info("History type: %s", type(hist))
    logger.info("History provided: %d messages", len(hist) if isinstance(hist, list) else 0)
    
    if isinstance(hist, list):
        logger.info("Processing %d history messages", len(hist))
        for i, h in enumerate(hist[-8:]):
            logger.info("History[%d]: %s", i, h)
            r = h.get("role")
            c = h.get("content")
            logger.info("  Role: %s, Content: %s", r, c[:100] if c and len(c) > 100 else c)
            if r in ("user", "assistant") and c:
                messages.append({"role": r, "content": c})
                logger.info("  -> Added to messages")
            else:
                logger.info("  -> SKIPPED (role=%s, has_content=%s)", r, bool(c))
    else:
        logger.warning("History is not a list! Type: %s, Value: %s", type(hist), hist)
    
    user_content = payload.get("content", "")
    messages.append({"role": "user", "content": user_content})
    logger.info("Total messages to LLM: %d", len(messages))
    logger.info("Message structure: system=%d, history=%d, user=1", 1, len(messages) - 2)
    
    logger.info("Calling LLM...")
    raw = call_llm(messages)
    logger.info("LLM raw response length: %d characters", len(raw) if raw else 0)
    logger.info("LLM raw response: %s", raw[:500] if raw else "None")
    
    reply = sanitize_reply(raw)
    logger.info("Sanitized reply: %s", reply[:500] if reply else "None")
    api_call = parse_api_call_tag(reply)
    logger.info("API call parse result: %s", api_call if api_call else "None")
    
    if api_call:
        method = api_call.get("method") or "GET"
        path = api_call.get("path") or ""
        api_payload = api_call.get("payload") or {}
        logger.info(">>> API CALL DETECTED: %s %s", method, path)
        logger.info("API payload: %s", json.dumps(api_payload) if api_payload else "None")
        
        # Remove /api prefix if present (external API doesn't need it)
        if path.startswith("/api/"):
            path = path[4:]  # Remove "/api" prefix
        
        # Call external developer API service
        result = await call_external_api(method, path, api_payload)
        logger.info("External API result: %s", json.dumps(result)[:200] if result else "None")
        format_prompt = (
            f"The API call was: {method} {path}. "
            f"The API returned: {json.dumps(result)}. "
            f"Formulate a friendly, human response based on the API result."
        )
        fm = [{"role": "system", "content": format_prompt}, {"role": "user", "content": api_payload if isinstance(api_payload, str) else json.dumps(api_payload)}]
        final_reply = sanitize_reply(call_llm(fm))
        logger.info("API response: %s", final_reply[:200] if final_reply else "None")
        logger.info("=" * 80)
        return JSONResponse({"reply": final_reply, "api": {"method": method, "path": path, "result": result}})
    
    search_query = parse_search_kb_tag(reply)
    logger.info("KB search parse result: %s", search_query if search_query else "None")
    
    if search_query:
        logger.info(">>> KB SEARCH DETECTED: %s", search_query)
        hits = search_with_threshold("default", search_query)
        logger.info("KB search returned %d hits", len(hits))
        docs = [d for (d, s) in hits]
        rag_answer = "\n".join([d.page_content for d in docs])
        logger.info("KB answer length: %d characters", len(rag_answer))
        format_prompt = (
            f"The user asked: '{search_query}'. "
            f"The knowledge base found: '{rag_answer}'. "
            f"Formulate a friendly, human response based on the knowledge base information."
        )
        fm = [{"role":"system","content": format_prompt}, {"role":"user","content": user_content }]
        logger.info("Calling LLM for KB response formatting...")
        final_reply = sanitize_reply(call_llm(fm))
        sources = [{
            "source_path": d.metadata.get("source_path"),
            "score": s,
            "preview": (d.page_content[:200] if d.page_content else "")
        } for (d, s) in hits]
        logger.info("KB response: %s", final_reply[:200] if final_reply else "None")
        logger.info("=" * 80)
        return JSONResponse({"reply": final_reply, "kb": {"query": search_query, "sources": sources}})
    
    logger.info(">>> DIRECT REPLY (no API/KB)")
    logger.info("Final reply: %s", reply[:500] if reply else "None")
    logger.info("=" * 80)
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
