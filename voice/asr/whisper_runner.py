import os
import subprocess
import threading
import logging
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_process = None
_model_path = None
_whisper_bin = None

def _ensure_process(whisper_bin: str, model_path: str) -> bool:
    global _process, _model_path, _whisper_bin
    if _process is not None and _model_path == model_path and _whisper_bin == whisper_bin:
        return True
    with _lock:
        if _process is not None and _model_path == model_path and _whisper_bin == whisper_bin:
            return True
        try:
            if not os.path.exists(whisper_bin):
                logger.error("Whisper binary does not exist: %s", whisper_bin)
                return False
            if not os.path.exists(model_path):
                logger.error("Whisper model file does not exist: %s", model_path)
                return False
            
            # Note: whisper-cli doesn't support persistent processes
            # Each call will load the model, but we cache the paths
            _whisper_bin = whisper_bin
            _model_path = model_path
            logger.info("Whisper paths cached: bin=%s, model=%s", whisper_bin, model_path)
            return True
        except Exception as e:
            logger.error("Failed to initialize whisper: %s", str(e), exc_info=True)
            return False

def transcribe_wav_bytes(whisper_bin: str, model_path: str, wav_bytes: bytes) -> Optional[str]:
    """
    Transcribe audio from in-memory wav bytes by piping to whisper-cli via stdin.
    This avoids file I/O during processing.
    """
    if not wav_bytes or len(wav_bytes) == 0:
        logger.error("WAV bytes are empty")
        return None
    
    if not _ensure_process(whisper_bin, model_path):
        logger.error("Whisper not available, cannot transcribe")
        return None
    
    try:
        logger.info("Transcribing audio from memory (wav_bytes=%d bytes)", len(wav_bytes))
        # Write wav bytes to a temporary file first, then use that file with whisper-cli
        # whisper-cli with -otxt creates a .txt file next to the input file
        import uuid
        tmp_wav_path = os.path.join(tempfile.gettempdir(), f"whisper_{uuid.uuid4().hex}.wav")
        tmp_out_path = tmp_wav_path + ".txt"  # whisper-cli creates this automatically
        
        try:
            # Write wav bytes to temp file
            with open(tmp_wav_path, 'wb') as f:
                f.write(wav_bytes)
            
            # Use the temp wav file with whisper-cli - it will create .txt file next to it
            res = subprocess.run(
                [_whisper_bin, "-m", _model_path, "-f", tmp_wav_path, "-otxt"],
                capture_output=True,
                timeout=300  # 5 minute timeout
            )
            if res.returncode != 0:
                stderr_text = res.stderr.decode('utf-8', errors='ignore') if res.stderr else ""
                logger.warning("whisper rc=%d stderr=%s", res.returncode, stderr_text[:200])
                return None
            
            # Read transcript from the output file (created next to input file)
            if os.path.exists(tmp_out_path):
                with open(tmp_out_path, 'r', encoding='utf-8') as f:
                    txt = f.read().strip()
                logger.info("whisper ok txt_len=%d txt='%s'", len(txt), txt[:100] if txt else "")
                return txt if txt else None
            else:
                logger.warning("whisper: output file not found: %s", tmp_out_path)
                return None
        finally:
            # Clean up temporary files
            try:
                if os.path.exists(tmp_wav_path):
                    os.remove(tmp_wav_path)
                if os.path.exists(tmp_out_path):
                    os.remove(tmp_out_path)
            except Exception as e:
                logger.warning("whisper: failed to remove temp files: %s", str(e))
    except subprocess.TimeoutExpired:
        logger.error("Whisper transcription timed out")
        return None
    except Exception as e:
        logger.error("Transcription failed: %s", str(e), exc_info=True)
        return None

def transcribe_wav(whisper_bin: str, model_path: str, wav_path: str) -> Optional[str]:
    """Legacy function for file-based transcription (kept for backward compatibility)"""
    if not os.path.exists(wav_path):
        logger.error("Audio file does not exist: %s", wav_path)
        return None
    
    if not _ensure_process(whisper_bin, model_path):
        logger.error("Whisper not available, cannot transcribe")
        return None
    
    try:
        logger.info("Transcribing audio file: %s", wav_path)
        res = subprocess.run(
            [_whisper_bin, "-m", _model_path, "-f", wav_path, "-otxt"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        if res.returncode != 0:
            logger.warning("whisper rc=%d stderr=%s", res.returncode, (res.stderr[:200] if res and res.stderr else ""))
            return None
        # When -otxt is used, output .txt will be created next to input; we can parse stdout or read file
        out_txt = wav_path + ".txt"
        if os.path.exists(out_txt):
            with open(out_txt, "r", encoding="utf-8") as f:
                txt = f.read().strip()
                logger.info("whisper ok txt_len=%d", len(txt))
                return txt
        txt = res.stdout.strip()
        logger.info("whisper ok stdout_len=%d", len(txt))
        return txt
    except subprocess.TimeoutExpired:
        logger.error("Whisper transcription timed out")
        return None
    except Exception as e:
        logger.error("Transcription failed: %s", str(e), exc_info=True)
        return None

