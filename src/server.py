from fastapi import FastAPI, Request
import logging
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
from app.services.together_client import call_llm
from app.api.state import router as state_router
from app.api.rag import router as rag_router
from app.api.setup import router as setup_router

app = FastAPI()
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')

@app.post("/api/chat")
async def chat(req: Request):
    body = await req.json()
    messages = body.get("messages", [])
    reply = call_llm(messages)
    return JSONResponse({"reply": reply})

app.include_router(state_router, prefix="/api")
app.include_router(rag_router)
app.include_router(setup_router)
app.mount("/static", StaticFiles(directory="public"), name="static")

@app.get("/")
def index():
    return FileResponse(os.path.join("public", "setup.html"))

@app.get("/chat.html")
def chat_page():
    return FileResponse(os.path.join("public", "chat.html"))

@app.get("/setup.html")
def setup_page():
    return FileResponse(os.path.join("public", "setup.html"))

 
