# Agentic System

A configurable AI assistant system with RAG (Retrieval Augmented Generation), API calling capabilities, and real-time voice interaction via WebSocket.

## Setup

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install External Tools

#### FFmpeg
Download and install FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html) or use a pre-built distribution.

Set the path in your `.env` file:
```env
FFMPEG_BIN=path/to/ffmpeg.exe  # Windows
# or
FFMPEG_BIN=ffmpeg  # Linux/Mac (if in PATH)
```

#### Faster-Whisper (ASR)
The system uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for speech recognition, which provides GPU acceleration and built-in VAD filtering.

**Option 1: Use model name (auto-downloads from HuggingFace)**
```env
WHISPER_MODEL=base  # Options: tiny, base, small, medium, large-v3, etc.
```

**Option 2: Use local converted model**
If you have a converted CTranslate2 model, specify the path:
```env
WHISPER_MODEL=path/to/whisper-base-ct2
```

**Note:** `WHISPER_BIN` is no longer required (deprecated). The system will automatically use GPU if available, or fall back to CPU.

**GPU Acceleration:**
- GPU is auto-detected if CUDA is available
- Set `USE_CUDA=true` in `.env` to force GPU, or `USE_CUDA=false` to force CPU
- GPU uses `float16` compute type, CPU uses `int8` for optimal performance

#### Piper TTS (Optional)
Download a Piper voice model and set the path:
```env
PIPER_VOICE=path/to/models/en_US-lessac-high.onnx
```

### 3. Environment Variables

Create a `.env` file in the root directory:

```env
TOGETHER_API_KEY=your_together_ai_api_key_here
TOGETHER_MODEL=Qwen/QwQ-32B

# External Developer API URL (deploy separately - see DEVELOPER_API.md)
EXTERNAL_API_BASE_URL=http://localhost:8001
# Or use your deployed URL:
# EXTERNAL_API_BASE_URL=https://your-username-developer-api.hf.space

# Voice processing (see above for setup instructions)
FFMPEG_BIN=ffmpeg
# WHISPER_BIN is deprecated (no longer needed with faster-whisper)
WHISPER_MODEL=base  # Model name (tiny, base, small, medium, large-v3) or local path
PIPER_VOICE=models/en_US-lessac-high.onnx
# Optional: Force GPU/CPU (auto-detected if not set)
# USE_CUDA=true  # Force GPU
# USE_CUDA=false  # Force CPU
```

Get your API key from [Together AI](https://together.ai/)

**Note**: The Developer API service should be deployed separately. See `DEVELOPER_API.md` for details.

### 4. Run the Server

```bash
uvicorn src.server:app --reload --host 0.0.0.0 --port 8000
```

Or using Python directly:

```bash
python -m uvicorn src.server:app --reload
```

The server will start on `http://localhost:8000`

## Usage

1. **Setup**: Visit `http://localhost:8000` to configure your assistant
   - Enter your raw script (conversation flow)
   - Optionally upload knowledge base files
   - Optionally define API endpoints

2. **Chat**: After setup, you'll be redirected to the chat interface at `http://localhost:8000/chat.html`

## Project Structure

- `src/server.py` - FastAPI application entry point
- `app/api/` - API route handlers
- `app/services/` - External service integrations
- `public/` - Frontend HTML files
- `configs/` - Configuration files
- `storage/` - Uploaded documents
- `data/chroma/` - Vector database storage

## Features

- **Script-driven conversations**: Define conversation flow via raw script
- **RAG integration**: Upload documents for knowledge base search
- **API calling**: Assistant can call defined API endpoints
- **Real-time voice interaction**: WebSocket-based voice communication with:
  - Voice Activity Detection (VAD) using Silero VAD
  - Automatic Speech Recognition (ASR) using Whisper
  - Text-to-Speech (TTS) using Piper
  - Buffer-based silence detection for efficient processing
- **Professional UI**: Multi-step setup wizard and clean chat interface

## Voice Features

The system supports real-time voice conversations:

1. **Voice Activity Detection**: Detects when the user is speaking vs. silent
2. **Silence-based processing**: Only processes audio when silence is detected (reduces CPU usage)
3. **In-memory processing**: Efficient audio processing with minimal disk I/O
4. **Segment archival**: All voice segments are saved with transcripts for debugging
5. **WebSocket streaming**: Low-latency audio streaming for real-time interaction


