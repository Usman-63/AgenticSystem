import os
import threading
import logging
import tempfile
import io
from typing import Optional

logger = logging.getLogger(__name__)

# Thread-safe singleton model instance
_lock = threading.Lock()
_model = None
_model_path = None
_device = None
_compute_type = None

def _get_model(model_path: str, device: str = "cpu", compute_type: str = "int8") -> Optional[object]:
    """
    Get or create a singleton WhisperModel instance.
    Thread-safe lazy initialization.
    """
    global _model, _model_path, _device, _compute_type
    
    # Check if we need to reload the model
    if _model is not None and _model_path == model_path and _device == device and _compute_type == compute_type:
        return _model
    
    with _lock:
        # Double-check after acquiring lock
        if _model is not None and _model_path == model_path and _device == device and _compute_type == compute_type:
            return _model
        
        try:
            from faster_whisper import WhisperModel
            
            logger.info("Loading Whisper model: path=%s, device=%s, compute_type=%s", model_path, device, compute_type)
            _model = WhisperModel(model_path, device=device, compute_type=compute_type)
            _model_path = model_path
            _device = device
            _compute_type = compute_type
            logger.info("Whisper model loaded successfully")
            return _model
        except ImportError:
            logger.error("faster-whisper not installed. Install with: pip install faster-whisper")
            return None
        except Exception as e:
            logger.error("Failed to load Whisper model: %s", str(e), exc_info=True)
            return None

def transcribe_wav_bytes(model_path: str, wav_bytes: bytes, device: str = "cpu", compute_type: str = "int8", 
                        vad_filter: bool = True, whisper_bin: str = None) -> Optional[str]:
    """
    Transcribe audio from in-memory wav bytes using faster-whisper.
    
    Args:
        model_path: Model name (e.g., "base", "large-v3") or local path
        wav_bytes: WAV audio data in memory
        device: "cpu" or "cuda"
        compute_type: "int8", "float16", "int8_float16", etc.
        vad_filter: Enable VAD filtering to remove silence (default: True)
        whisper_bin: Deprecated, kept for backward compatibility (ignored)
    
    Returns:
        Transcribed text or None on error
    """
    if whisper_bin:
        logger.warning("whisper_bin parameter is deprecated and ignored (using faster-whisper)")
    
    if not wav_bytes or len(wav_bytes) == 0:
        logger.error("WAV bytes are empty")
        return None
    
    model = _get_model(model_path, device, compute_type)
    if model is None:
        logger.error("Whisper model not available, cannot transcribe")
        return None
    
    try:
        logger.info("Transcribing audio from memory (wav_bytes=%d bytes)", len(wav_bytes))
        
        # Write wav bytes to temporary file (faster-whisper needs a file path)
        import uuid
        tmp_wav_path = os.path.join(tempfile.gettempdir(), f"whisper_{uuid.uuid4().hex}.wav")
        
        try:
            with open(tmp_wav_path, 'wb') as f:
                f.write(wav_bytes)
            
            # Transcribe with VAD filtering enabled by default
            segments, info = model.transcribe(
                tmp_wav_path,
                vad_filter=vad_filter,
                vad_parameters=dict(min_silence_duration_ms=500) if vad_filter else None,
                beam_size=5
            )
            
            # Collect all segment texts
            texts = []
            for segment in segments:
                if segment.text.strip():
                    texts.append(segment.text.strip())
            
            txt = " ".join(texts).strip()
            logger.info("whisper ok txt_len=%d txt='%s'", len(txt), txt[:100] if txt else "")
            return txt if txt else None
            
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(tmp_wav_path):
                    os.remove(tmp_wav_path)
            except Exception as e:
                logger.warning("Failed to remove temp file: %s", str(e))
                
    except Exception as e:
        logger.error("Transcription failed: %s", str(e), exc_info=True)
        return None

def transcribe_wav(model_path: str, wav_path: str, device: str = "cpu", compute_type: str = "int8",
                  vad_filter: bool = True, whisper_bin: str = None) -> Optional[str]:
    """
    Transcribe audio from a WAV file using faster-whisper.
    
    Args:
        model_path: Model name (e.g., "base", "large-v3") or local path
        wav_path: Path to WAV audio file
        device: "cpu" or "cuda"
        compute_type: "int8", "float16", "int8_float16", etc.
        vad_filter: Enable VAD filtering to remove silence (default: True)
        whisper_bin: Deprecated, kept for backward compatibility (ignored)
    
    Returns:
        Transcribed text or None on error
    """
    if whisper_bin:
        logger.warning("whisper_bin parameter is deprecated and ignored (using faster-whisper)")
    
    if not os.path.exists(wav_path):
        logger.error("Audio file does not exist: %s", wav_path)
        return None
    
    model = _get_model(model_path, device, compute_type)
    if model is None:
        logger.error("Whisper model not available, cannot transcribe")
        return None
    
    try:
        logger.info("Transcribing audio file: %s", wav_path)
        
        # Transcribe with VAD filtering enabled by default
        segments, info = model.transcribe(
            wav_path,
            vad_filter=vad_filter,
            vad_parameters=dict(min_silence_duration_ms=500) if vad_filter else None,
            beam_size=5
        )
        
        # Collect all segment texts
        texts = []
        for segment in segments:
            if segment.text.strip():
                texts.append(segment.text.strip())
        
        txt = " ".join(texts).strip()
        logger.info("whisper ok txt_len=%d", len(txt))
        return txt if txt else None
        
    except Exception as e:
        logger.error("Transcription failed: %s", str(e), exc_info=True)
        return None
