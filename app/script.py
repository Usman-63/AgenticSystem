import json
import re
import logging
from typing import Dict, Optional

def sanitize_reply(text: str) -> str:
    t = re.sub(r"<think>[\s\S]*?</think>\s*", "", text, flags=re.MULTILINE)
    return t.strip()

def parse_search_kb_tag(text: str) -> Optional[str]:
    m = re.search(r"\[SEARCH_KB:\s*'([\s\S]*?)'\s*\]", text)
    if not m:
        return None
    return m.group(1)

def parse_api_call_tag(text: str) -> Optional[Dict]:
    m = re.search(r"\[API_CALL:\s*'([A-Z]+)\s+([^']+)'\s*(?:,\s*(\{[\s\S]*?\}))?\s*\]", text)
    if not m:
        return None
    method = m.group(1).strip()
    path = m.group(2).strip()
    payload = {}
    if m.group(3):
        try:
            payload = json.loads(m.group(3))
        except Exception:
            payload = {}
    return {"method": method, "path": path, "payload": payload}
