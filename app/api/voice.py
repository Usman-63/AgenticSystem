from fastapi import APIRouter, UploadFile, File, WebSocket
from fastapi.responses import JSONResponse, FileResponse
import os
import uuid
from typing import Dict
from voice.service.voice_session import VoiceSessionManager, synth_silence_wav
from voice.asr.whisper_runner import transcribe_wav
from voice.tts.piper_runner import synthesize_wav_api
from voice.vad.silero_runner import get_speech_segments
from voice.service.turn_manager import TurnManager
from app.services.together_client import call_llm
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

BASE_DIR = os.path.join("storage", "voice")
os.makedirs(BASE_DIR, exist_ok=True)
SESS = VoiceSessionManager(BASE_DIR)

# Config (could be moved to configs/voice.json)
WHISPER_BIN = os.getenv("WHISPER_BIN", "")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "")
PIPER_VOICE = os.getenv("PIPER_VOICE", "")
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
TURN = TurnManager(BASE_DIR, FFMPEG_BIN, WHISPER_BIN, WHISPER_MODEL, PIPER_VOICE)

@router.post("/voice/start")
async def voice_start():
    sid = SESS.start()
    logger.info("voice_start sid=%s", sid)
    TURN.start(sid)
    return JSONResponse({"ok": True, "session_id": sid})

@router.post("/voice/upload")
async def voice_upload(session_id: str, audio: UploadFile = File(...)):
    s = SESS.get(session_id)
    if not s:
        return JSONResponse({"ok": False, "error": "invalid session"}, status_code=400)
    ext = os.path.splitext(audio.filename or "")[1] or ".webm"
    in_path = os.path.join(s.dir, f"input{ext}")
    logger.info("voice_upload sid=%s filename=%s ext=%s", session_id, audio.filename, ext)
    with open(in_path, "wb") as f:
        payload = await audio.read()
        f.write(payload)
    try:
        size = os.path.getsize(in_path)
    except Exception:
        size = 0
    logger.info("voice_upload saved path=%s size=%d", in_path, size)
    # Convert to 16kHz mono WAV for Whisper
    out_wav = os.path.join(s.dir, "input.wav")
    try:
        import subprocess
        cmd = [FFMPEG_BIN, "-y", "-i", in_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", out_wav]
        logger.info("voice_upload ffmpeg cmd=%s", " ".join(cmd))
        res = subprocess.run(cmd, capture_output=True)
        logger.info("voice_upload ffmpeg rc=%d stderr=%s", res.returncode, (res.stderr[:200] if res and res.stderr else b""))
        if res.returncode == 0 and os.path.exists(out_wav) and os.path.getsize(out_wav) > 0:
            SESS.set_input(session_id, out_wav)
            logger.info("voice_upload converted out_wav=%s size=%d", out_wav, os.path.getsize(out_wav))
            return JSONResponse({"ok": True, "path": out_wav})
    except Exception:
        logger.warning("voice_upload ffmpeg failed")
    # Fallback to original path if conversion failed
    SESS.set_input(session_id, in_path)
    logger.info("voice_upload fallback to original path=%s", in_path)
    return JSONResponse({"ok": True, "path": in_path, "converted": False})

@router.post("/voice/stop")
async def voice_stop(payload: Dict):
    sid = payload.get("session_id")
    s = SESS.get(sid)
    if not s or not s.input_path:
        return JSONResponse({"ok": False, "error": "no input"}, status_code=400)
    logger.info("voice_stop sid=%s input_path=%s", sid, s.input_path)
    transcript = transcribe_wav(WHISPER_BIN, WHISPER_MODEL, s.input_path) or ""
    logger.info("voice_stop transcript_len=%d", len(transcript))
    if not transcript:
        transcript = ""
    SESS.set_transcript(sid, transcript)
    # Call agent with transcript
    msgs = [{"role": "user", "content": transcript or "(no audio recognized)"}]
    reply = call_llm(msgs)
    logger.info("voice_stop reply_len=%d", len(reply or ""))
    SESS.set_reply_text(sid, reply)
    # TTS
    out_wav = os.path.join(s.dir, "reply.wav")
    ok = False
    if PIPER_VOICE:
        logger.info("voice_stop tts start voice=%s out=%s", PIPER_VOICE, out_wav)
        ok = synthesize_wav_api(PIPER_VOICE, reply, out_wav)
    if not ok:
        synth_silence_wav(out_wav, seconds=0.5)
        logger.info("voice_stop tts failed, synthesized silence out=%s", out_wav)
    else:
        try:
            logger.info("voice_stop tts ok size=%d", os.path.getsize(out_wav))
        except Exception:
            logger.info("voice_stop tts ok size=?")
    SESS.set_reply_audio(sid, out_wav)
    return JSONResponse({"ok": True, "session_id": sid, "transcript": transcript, "reply": reply, "audio_path": out_wav})

@router.get("/voice/audio/{session_id}")
async def voice_audio(session_id: str):
    s = SESS.get(session_id)
    if not s or not s.reply_audio_path or not os.path.exists(s.reply_audio_path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path=s.reply_audio_path, media_type="audio/wav")

@router.post("/voice/asr")
async def voice_asr(payload: Dict):
    sid = payload.get("session_id")
    s = SESS.get(sid)
    if not s or not s.input_path:
        return JSONResponse({"ok": False, "error": "no input"}, status_code=400)
    txt = transcribe_wav(WHISPER_BIN, WHISPER_MODEL, s.input_path) or ""
    return JSONResponse({"ok": True, "session_id": sid, "transcript": txt})

@router.post("/voice/vad")
async def voice_vad(payload: Dict):
    sid = payload.get("session_id")
    s = SESS.get(sid)
    if not s or not s.input_path:
        logger.warning("voice_vad: no session or input_path sid=%s", sid)
        return JSONResponse({"ok": False, "error": "no input"}, status_code=400)
    if not os.path.exists(s.input_path):
        logger.warning("voice_vad: input_path does not exist sid=%s path=%s", sid, s.input_path)
        return JSONResponse({"ok": False, "error": "input file not found"}, status_code=400)
    thr = float(payload.get("threshold", 0.5))
    mspeech = int(payload.get("min_speech_ms", 250))
    msil = int(payload.get("min_silence_ms", 200))
    logger.info("voice_vad: sid=%s path=%s threshold=%.2f min_speech=%d min_silence=%d", 
                sid, s.input_path, thr, mspeech, msil)
    try:
        segments = get_speech_segments(s.input_path, sampling_rate=16000, threshold=thr, min_speech_ms=mspeech, min_silence_ms=msil)
        logger.info("voice_vad: returned %d segments", len(segments) if segments else 0)
        if segments:
            logger.debug("voice_vad: first segment: %s", segments[0] if segments else None)
        return JSONResponse({"ok": True, "session_id": sid, "segments": segments or []})
    except Exception as e:
        logger.error("voice_vad: exception: %s", str(e), exc_info=True)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@router.post("/voice/frames")
async def voice_frames(session_id: str, respond: bool = False, chunk: UploadFile = File(...)):
    try:
        data = await chunk.read()
        logger.info("voice_frames: sid=%s respond=%s chunk_size=%d", session_id, respond, len(data))
        res = TURN.push_chunk(session_id, data, respond=respond)
        if not res:
            logger.error("voice_frames: push_chunk returned None for sid=%s", session_id)
            return JSONResponse({"ok": False, "error": "push_chunk returned None"}, status_code=500)
        if res.get("audio_path"):
            s = SESS.get(session_id)
            if s:
                s.reply_audio_path = res["audio_path"]
        logger.debug("voice_frames: response finalized=%s state=%s", res.get("finalized"), res.get("state"))
        return JSONResponse(res)
    except Exception as e:
        logger.error("voice_frames: exception: %s", str(e), exc_info=True)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)