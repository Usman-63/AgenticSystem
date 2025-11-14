# Agentic System

A configurable AI assistant system with RAG (Retrieval Augmented Generation) and API calling capabilities.

## Setup

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the root directory:

```env
TOGETHER_API_KEY=your_together_ai_api_key_here
TOGETHER_MODEL=Qwen/QwQ-32B

# External Developer API URL (deploy separately - see DEVELOPER_API.md)
EXTERNAL_API_BASE_URL=http://localhost:8001
# Or use your deployed URL:
# EXTERNAL_API_BASE_URL=https://your-username-developer-api.hf.space
```

Get your API key from [Together AI](https://together.ai/)

**Note**: The Developer API service should be deployed separately. See `DEVELOPER_API.md` for details.

### 3. Run the Server

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
- **Professional UI**: Multi-step setup wizard and clean chat interface


