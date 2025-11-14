import os
import logging
from dotenv import load_dotenv
from together import Together

logger = logging.getLogger(__name__)

load_dotenv()
MODEL = os.getenv("TOGETHER_MODEL", "Qwen/QwQ-32B")

def call_llm(messages):
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing TOGETHER_API_KEY in .env")
    client = Together(api_key=api_key)
    res = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    if not res or not res.choices:
        logger.error("LLM call failed: no response")
        raise RuntimeError("LLM call failed")
    return res.choices[0].message.content
