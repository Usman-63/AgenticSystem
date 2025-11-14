from fastapi import APIRouter
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
async def scripted_chat_default(payload: Dict):
    fresh = _load_default_script()
    script = fresh.model_dump()
    turn = 0
    try:
        turn = int(payload.get("turn", 0))
    except Exception:
        turn = 0
    logger.info("Scripted chat request, turn=%d", turn)
    # Build system prompt from raw script + documents list, and nothing else
    messages = []
    raw_path = os.path.join(os.getcwd(), "simpleScript.txt")
    raw_script = ""
    if os.path.exists(raw_path):
        with open(raw_path, "r", encoding="utf-8") as f:
            raw_script = f.read()
    else:
        logger.warning("Raw script not found")
    docs = script.get("rag_context", {}).get("documents") or []
    doc_text = ""
    if docs:
        doc_lines = ["\nDocuments available:"] + [f"- {d}" for d in docs]
        doc_text = "\n" + "\n".join(doc_lines)
    endpoints = script.get("api_endpoints") or []
    ep_text = ""
    if isinstance(endpoints, list) and endpoints:
        ep_lines = ["\nAPI endpoints available:"]
        for e in endpoints:
            method = e.get('method', 'GET')
            path = e.get('path', '')
            desc = e.get('description', '')
            ep_lines.append(f"- {method} {path}: {desc}")
            payload = e.get('payload')
            if payload and isinstance(payload, dict) and payload:
                # Show full payload structure as JSON for the LLM
                payload_json = json.dumps(payload, indent=2)
                ep_lines.append(f"  Payload schema:\n{payload_json}")
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
    messages.append({"role": "system", "content": system_only})
    hist = payload.get("history", [])
    if isinstance(hist, list):
        for h in hist[-8:]:
            r = h.get("role")
            c = h.get("content")
            if r in ("user", "assistant") and c:
                messages.append({"role": r, "content": c})
    messages.append({"role": "user", "content": payload.get("content", "")})
    raw = call_llm(messages)
    reply = sanitize_reply(raw)
    api_call = parse_api_call_tag(reply)
    if api_call:
        method = api_call.get("method") or "GET"
        path = api_call.get("path") or ""
        payload = api_call.get("payload") or {}
        logger.info("API call detected: %s %s", method, path)
        
        # Remove /api prefix if present (external API doesn't need it)
        if path.startswith("/api/"):
            path = path[4:]  # Remove "/api" prefix
        
        # Call external developer API service
        result = await call_external_api(method, path, payload)
        format_prompt = (
            f"The API call was: {method} {path}. "
            f"The API returned: {json.dumps(result)}. "
            f"Formulate a friendly, human response based on the API result."
        )
        fm = [{"role": "system", "content": format_prompt}, {"role": "user", "content": payload if isinstance(payload, str) else json.dumps(payload)}]
        final_reply = sanitize_reply(call_llm(fm))
        return JSONResponse({"reply": final_reply, "api": {"method": method, "path": path, "result": result}})
    search_query = parse_search_kb_tag(reply)
    if search_query:
        logger.info("KB search requested: %s", search_query)
        hits = search_with_threshold("default", search_query)
        docs = [d for (d, s) in hits]
        rag_answer = "\n".join([d.page_content for d in docs])
        format_prompt = (
            f"The user asked: '{search_query}'. "
            f"The knowledge base found: '{rag_answer}'. "
            f"Formulate a friendly, human response based on the knowledge base information."
        )
        fm = [{"role":"system","content": format_prompt}, {"role":"user","content": payload.get("content","") }]
        final_reply = sanitize_reply(call_llm(fm))
        sources = [{
            "source_path": d.metadata.get("source_path"),
            "score": s,
            "preview": (d.page_content[:200] if d.page_content else "")
        } for (d, s) in hits]
        return JSONResponse({"reply": final_reply, "kb": {"query": search_query, "sources": sources}})
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
