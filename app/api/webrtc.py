"""
WebRTC signaling and data channel handling for real-time voice communication.
Uses WebSocket for signaling and WebRTC DataChannel for audio streaming.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import json
import logging
import asyncio
from voice.service.turn_manager import TurnManager
from voice.service.voice_session import VoiceSessionManager
import os

router = APIRouter()
logger = logging.getLogger(__name__)

BASE_DIR = os.path.join("storage", "voice")
os.makedirs(BASE_DIR, exist_ok=True)
SESS = VoiceSessionManager(BASE_DIR)

# Config
WHISPER_BIN = os.getenv("WHISPER_BIN", "")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "")
PIPER_VOICE = os.getenv("PIPER_VOICE", "")
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
TURN = TurnManager(BASE_DIR, FFMPEG_BIN, WHISPER_BIN, WHISPER_MODEL, PIPER_VOICE)

# Store active WebRTC connections
active_connections: dict[str, WebSocket] = {}

@router.websocket("/voice/webrtc/{session_id}")
async def webrtc_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for WebRTC signaling and audio data streaming.
    Handles:
    1. WebRTC offer/answer exchange (signaling)
    2. Audio chunk streaming via DataChannel messages
    3. Real-time processing with buffer-based silence detection
    """
    await websocket.accept()
    active_connections[session_id] = websocket
    logger.info("WebRTC WebSocket connected: session_id=%s", session_id)
    
    # Initialize session
    TURN.start(session_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                msg_type = message.get("type")
                
                if msg_type == "offer":
                    # WebRTC offer received - for now, we'll use DataChannel directly
                    # In a full WebRTC implementation, you'd handle SDP exchange here
                    await websocket.send_text(json.dumps({
                        "type": "answer",
                        "session_id": session_id,
                        "status": "ready"
                    }))
                    logger.info("WebRTC offer received for session_id=%s", session_id)
                
                elif msg_type == "audio_chunk":
                    # Audio data received via DataChannel (base64 encoded)
                    import base64
                    chunk_data = base64.b64decode(message.get("data", ""))
                    respond = message.get("respond", False)
                    
                    # Process chunk with buffer-based silence detection
                    res = TURN.push_chunk(session_id, chunk_data, respond=respond)
                    
                    # Send response back
                    response_msg = {
                        "type": "processing_result",
                        **res
                    }
                    logger.info("webrtc_websocket: sending processing_result: finalized=%s, transcript='%s', reply='%s'", 
                               res.get("finalized"), res.get("transcript", ""), res.get("reply", ""))
                    await websocket.send_text(json.dumps(response_msg))
                    
                    # If finalized, send transcript and reply
                    if res.get("finalized"):
                        if res.get("audio_path"):
                            # Send audio file path for playback
                            await websocket.send_text(json.dumps({
                                "type": "audio_ready",
                                "audio_path": f"/api/voice/audio/{session_id}"
                            }))
                
                elif msg_type == "ping":
                    # Keep-alive
                    await websocket.send_text(json.dumps({"type": "pong"}))
                
                else:
                    logger.warning("Unknown message type: %s", msg_type)
                    
            except json.JSONDecodeError:
                logger.error("Invalid JSON received: %s", data[:100])
            except Exception as e:
                logger.error("Error processing message: %s", str(e), exc_info=True)
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "error": str(e)
                }))
                
    except WebSocketDisconnect:
        logger.info("WebRTC WebSocket disconnected: session_id=%s", session_id)
    except Exception as e:
        logger.error("WebRTC WebSocket error: %s", str(e), exc_info=True)
    finally:
        active_connections.pop(session_id, None)
        logger.info("WebRTC connection closed: session_id=%s", session_id)

@router.post("/voice/webrtc/start")
async def webrtc_start():
    """Start a new WebRTC session"""
    sid = SESS.start()
    logger.info("webrtc_start sid=%s", sid)
    TURN.start(sid)
    return JSONResponse({"ok": True, "session_id": sid})

