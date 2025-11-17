"""
WebRTC signaling and data channel handling for real-time voice communication.
Uses WebSocket for signaling and WebRTC DataChannel for audio streaming.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import json
import logging
import asyncio
from voice.service.shared_session import SESS, TURN
import os
from uvicorn.protocols.utils import ClientDisconnected

router = APIRouter()
logger = logging.getLogger(__name__)

# Store active WebRTC connections
active_connections: dict[str, WebSocket] = {}

async def safe_send_text(websocket: WebSocket, message: dict) -> bool:
    """
    Safely send a text message over WebSocket, handling connection errors gracefully.
    Returns True if sent successfully, False if connection is closed.
    """
    try:
        await websocket.send_text(json.dumps(message))
        return True
    except (WebSocketDisconnect, ClientDisconnected, RuntimeError) as e:
        # Connection is closed or closing, this is expected when client disconnects
        logger.debug("WebSocket send failed (connection closed): %s", str(e))
        return False
    except Exception as e:
        logger.error("Unexpected error sending WebSocket message: %s", str(e), exc_info=True)
        return False

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
    
    # Initialize session in both TURN and SESS (ensure they're in sync)
    TURN.start(session_id)
    # Ensure session exists in SESS as well
    if not SESS.get(session_id):
        from voice.service.voice_session import VoiceSession
        s = VoiceSession(session_id, SESS.base)
        SESS.sessions[session_id] = s
        logger.info("webrtc_websocket: created session in SESS for session_id=%s", session_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                msg_type = message.get("type")
                
                if msg_type == "offer":
                    # WebRTC offer received - for now, we'll use DataChannel directly
                    # In a full WebRTC implementation, you'd handle SDP exchange here
                    if not await safe_send_text(websocket, {
                        "type": "answer",
                        "session_id": session_id,
                        "status": "ready"
                    }):
                        break  # Connection closed, exit loop
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
                    if not await safe_send_text(websocket, response_msg):
                        break  # Connection closed, exit loop
                    
                    # If finalized, send transcript and reply
                    if res.get("finalized"):
                        if res.get("audio_path") and os.path.exists(res["audio_path"]):
                            # Ensure session exists and set audio path in shared session
                            s = SESS.get(session_id)
                            if not s:
                                # Session might not exist yet, create it (but this shouldn't happen since TURN.start was called)
                                logger.warning("webrtc_websocket: session %s not found in SESS, creating it", session_id)
                                # Note: SESS.start() creates a new session_id, so we can't use it here
                                # Instead, we'll create the session manually
                                from voice.service.voice_session import VoiceSession
                                s = VoiceSession(session_id, SESS.base)
                                SESS.sessions[session_id] = s
                            if s:
                                s.reply_audio_path = res["audio_path"]
                                logger.info("webrtc_websocket: set reply_audio_path=%s for session_id=%s", res["audio_path"], session_id)
                            # Only notify client if audio file exists
                            # Include segment number in path to prevent browser caching old audio
                            import time
                            audio_url = f"/api/voice/audio/{session_id}?t={int(time.time() * 1000)}"
                            if not await safe_send_text(websocket, {
                                "type": "audio_ready",
                                "audio_path": audio_url,
                                "audio_file": res["audio_path"]  # Send actual file path for server lookup
                            }):
                                break  # Connection closed, exit loop
                        else:
                            logger.warning("Audio path not available or file doesn't exist: %s", res.get("audio_path"))
                
                elif msg_type == "ping":
                    # Keep-alive
                    if not await safe_send_text(websocket, {"type": "pong"}):
                        break  # Connection closed, exit loop
                
                elif msg_type == "playback_complete":
                    # Client notifies that playback has finished - clear processing flag
                    TURN.clear_processing_flag(session_id)
                    logger.info("webrtc_websocket: playback_complete received for session_id=%s", session_id)
                
                else:
                    logger.warning("Unknown message type: %s", msg_type)
                    
            except json.JSONDecodeError:
                logger.error("Invalid JSON received: %s", data[:100])
            except Exception as e:
                logger.error("Error processing message: %s", str(e), exc_info=True)
                # Try to send error, but don't break if connection is closed
                if not await safe_send_text(websocket, {
                    "type": "error",
                    "error": str(e)
                }):
                    break  # Connection closed, exit loop
                
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

