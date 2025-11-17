import os
import threading
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

_vad_lock = threading.Lock()
_vad_model = None
_vad_utils = None

def _ensure_vad() -> bool:
    global _vad_model, _vad_utils
    if _vad_model is not None and _vad_utils is not None:
        return True
    with _vad_lock:
        if _vad_model is not None and _vad_utils is not None:
            return True
        try:
            import torch
            logger.info("Loading Silero VAD model...")
            _vad_model, _vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                    model='silero_vad',
                                                    force_reload=False,
                                                    trust_repo=True)
            logger.info("Silero VAD model loaded successfully")
            return True
        except Exception as e:
            logger.error("Failed to load Silero VAD model: %s", str(e), exc_info=True)
            return False

def get_speech_segments_from_audio(audio_array, sampling_rate: int = 16000,
                                   threshold: float = 0.5,
                                   min_speech_ms: int = 250,
                                   min_silence_ms: int = 200) -> List[Dict]:
    """
    Run VAD on in-memory audio array (numpy array).
    This is more efficient than reading from disk.
    """
    if not _ensure_vad():
        logger.error("VAD model not available")
        return []
    
    try:
        import numpy as np
        # Extract get_speech_timestamps function from _vad_utils tuple
        get_speech_timestamps = None
        if isinstance(_vad_utils, tuple):
            for item in _vad_utils:
                if getattr(item, '__name__', '') == 'get_speech_timestamps':
                    get_speech_timestamps = item
                    break
            if get_speech_timestamps is None and len(_vad_utils) > 0:
                get_speech_timestamps = _vad_utils[0]
        elif isinstance(_vad_utils, dict):
            get_speech_timestamps = _vad_utils.get('get_speech_timestamps')
        else:
            logger.error("Unexpected _vad_utils type: %s", type(_vad_utils))
            return []
        
        if get_speech_timestamps is None:
            logger.error("Could not find get_speech_timestamps function")
            return []
        
        # Ensure audio is numpy array and correct format
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array, dtype=np.float32)
        
        # Ensure it's float32 and normalized
        if audio_array.dtype != np.float32:
            if audio_array.dtype == np.int16:
                audio_array = audio_array.astype(np.float32) / 32768.0
            elif audio_array.dtype == np.int32:
                audio_array = audio_array.astype(np.float32) / 2147483648.0
            else:
                audio_array = audio_array.astype(np.float32)
        
        # Convert to mono if stereo
        if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
            audio_array = np.mean(audio_array, axis=1)
        
        logger.debug("Running VAD on in-memory audio, length: %d samples", len(audio_array))
        ts = get_speech_timestamps(
            audio_array,
            _vad_model,
            sampling_rate=sampling_rate,
            threshold=threshold,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms,
        )
        logger.debug("VAD detected %d timestamp segments", len(ts) if ts else 0)
        out: List[Dict] = []
        for t in ts or []:
            start_val = t.get('start', 0) if isinstance(t, dict) else (t.start if hasattr(t, 'start') else 0)
            end_val = t.get('end', 0) if isinstance(t, dict) else (t.end if hasattr(t, 'end') else 0)
            out.append({
                'start': start_val / sampling_rate,
                'end': end_val / sampling_rate,
            })
        return out
    except Exception as e:
        logger.error("Error in get_speech_segments_from_audio: %s", str(e), exc_info=True)
        return []

def get_speech_segments(wav_path: str, sampling_rate: int = 16000,
                        threshold: float = 0.5,
                        min_speech_ms: int = 250,
                        min_silence_ms: int = 200) -> List[Dict]:
    if not _ensure_vad():
        logger.error("VAD model not available")
        return []
    if not os.path.exists(wav_path):
        logger.error("WAV file does not exist: %s", wav_path)
        return []
    try:
        # Extract get_speech_timestamps function from _vad_utils tuple
        get_speech_timestamps = None
        if isinstance(_vad_utils, tuple):
            # Try to find by name first
            for item in _vad_utils:
                if getattr(item, '__name__', '') == 'get_speech_timestamps':
                    get_speech_timestamps = item
                    break
            # Fallback to first element if name-based search didn't work
            if get_speech_timestamps is None and len(_vad_utils) > 0:
                get_speech_timestamps = _vad_utils[0]
        elif isinstance(_vad_utils, dict):
            get_speech_timestamps = _vad_utils.get('get_speech_timestamps')
        else:
            logger.error("Unexpected _vad_utils type: %s", type(_vad_utils))
            return []
        
        if get_speech_timestamps is None:
            logger.error("Could not find get_speech_timestamps function")
            return []
        
        # Use soundfile to read audio (torchcodec has FFmpeg DLL issues on Windows)
        # Audio is already converted to 16kHz mono by ffmpeg in the upload endpoint
        logger.info("Reading audio from %s (sampling_rate=%d)", wav_path, sampling_rate)
        try:
            import soundfile as sf
            import numpy as np
            # Read audio file
            audio, sr = sf.read(wav_path)
            # Convert to mono if stereo (shouldn't happen since ffmpeg converts to mono)
            if len(audio.shape) > 1 and audio.shape[1] > 1:
                audio = np.mean(audio, axis=1)
            # Ensure it's float32 and normalized to [-1, 1]
            if audio.dtype != np.float32:
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
                elif audio.dtype == np.int32:
                    audio = audio.astype(np.float32) / 2147483648.0
                else:
                    audio = audio.astype(np.float32)
            # Audio should already be 16kHz from ffmpeg conversion, but log if different
            if sr != sampling_rate:
                logger.warning("Audio sample rate is %d Hz, expected %d Hz. VAD may not work correctly.", sr, sampling_rate)
        except ImportError as e:
            logger.error("soundfile not available: %s", str(e))
            return []
        except Exception as e:
            logger.error("Error reading audio file: %s", str(e), exc_info=True)
            return []
        logger.info("Audio loaded, length: %d samples", len(audio) if audio is not None else 0)
        ts = get_speech_timestamps(
            audio,
            _vad_model,
            sampling_rate=sampling_rate,
            threshold=threshold,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms,
        )
        logger.info("VAD detected %d timestamp segments", len(ts) if ts else 0)
        out: List[Dict] = []
        for t in ts or []:
            start_val = t.get('start', 0) if isinstance(t, dict) else (t.start if hasattr(t, 'start') else 0)
            end_val = t.get('end', 0) if isinstance(t, dict) else (t.end if hasattr(t, 'end') else 0)
            out.append({
                'start': start_val / sampling_rate,
                'end': end_val / sampling_rate,
            })
        logger.info("Converted to %d segments", len(out))
        return out
    except Exception as e:
        logger.error("Error in get_speech_segments: %s", str(e), exc_info=True)
        return []
