"""
Shared session manager instance for voice and webrtc routers.
This ensures both routers share the same session state.
"""
import os
import logging
from voice.service.voice_session import VoiceSessionManager
from voice.service.turn_manager import TurnManager

logger = logging.getLogger(__name__)

BASE_DIR = os.path.join("storage", "voice")
os.makedirs(BASE_DIR, exist_ok=True)

# Shared instances
SESS = VoiceSessionManager(BASE_DIR)

# Auto-detect GPU availability
def _detect_cuda() -> bool:
    """Detect if CUDA is available for GPU acceleration"""
    # Check if explicitly set in environment
    env_cuda = os.getenv("USE_CUDA", "").lower()
    if env_cuda in ("true", "1", "yes"):
        return True
    if env_cuda in ("false", "0", "no"):
        return False
    
    # Auto-detect CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"[INFO] voice.service.shared_session: CUDA detected: GPU acceleration enabled (device: {device_name})")
            logger.info("CUDA detected: GPU acceleration enabled (device: %s)", device_name)
            return True
    except ImportError:
        pass
    except Exception as e:
        logger.debug("PyTorch CUDA check failed: %s", str(e))
    
    try:
        # Try to import onnxruntime with CUDA provider
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available_providers:
            print("[INFO] voice.service.shared_session: ONNX Runtime CUDA provider detected: GPU acceleration enabled")
            logger.info("ONNX Runtime CUDA provider detected: GPU acceleration enabled")
            return True
    except ImportError:
        pass
    except Exception as e:
        logger.debug("ONNX Runtime CUDA check failed: %s", str(e))
    
    print("[INFO] voice.service.shared_session: No GPU detected: Using CPU")
    logger.info("No GPU detected: Using CPU")
    return False

# Config
# WHISPER_BIN is deprecated but kept for backward compatibility (not used with faster-whisper)
WHISPER_BIN = os.getenv("WHISPER_BIN", "")
if WHISPER_BIN:
    logger.warning("WHISPER_BIN environment variable is deprecated (using faster-whisper, ignoring WHISPER_BIN)")

# WHISPER_MODEL can be:
# - Model name: "tiny", "base", "small", "medium", "large-v3", etc. (auto-downloads from HuggingFace)
# - Local path: path to converted CTranslate2 model directory
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # Default to "base" if not set
if not WHISPER_MODEL:
    WHISPER_MODEL = "base"  # Fallback to base model
    logger.warning("WHISPER_MODEL not set, using default: 'base'")

PIPER_VOICE = os.getenv("PIPER_VOICE", "")
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
USE_CUDA = _detect_cuda()

# Initialize TurnManager (whisper_bin is optional, kept for backward compat)
TURN = TurnManager(
    BASE_DIR, 
    FFMPEG_BIN, 
    WHISPER_MODEL, 
    PIPER_VOICE, 
    use_cuda=USE_CUDA,
    whisper_bin=WHISPER_BIN if WHISPER_BIN else None
)

