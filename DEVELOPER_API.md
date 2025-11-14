# Developer API Service

The Developer API Service is a **separate service** that provides mock API endpoints. This mimics what developers would implement for their companies.

## Architecture

```
┌─────────────────────┐
│  Agentic System     │
│  (Main Service)     │
└──────────┬──────────┘
           │
           │ HTTP Calls
           │
           ▼
┌─────────────────────┐
│  Developer API      │
│  (External Service) │
│  - /customer/submit │
│  - /orders/preview  │
│  - /ping            │
│  etc.               │
└─────────────────────┘
```

## Setup

### 1. Deploy Developer API

The Developer API is in the `developer_api/` folder. Deploy it separately:

**Option A: HuggingFace Spaces**
1. Create a new HuggingFace Space
2. Upload files from `developer_api/` folder
3. Set the Space to use `app.py` as the entry point

**Option B: Local Development**
```bash
cd developer_api
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

### 2. Configure Main Agentic System

Set the environment variable to point to your deployed Developer API:

```env
EXTERNAL_API_BASE_URL=https://your-username-developer-api.hf.space
```

Or for local development:
```env
EXTERNAL_API_BASE_URL=http://localhost:8001
```

## Endpoints Available

The Developer API provides these endpoints:

- `GET /health` - Health check
- `GET /ping` - Ping endpoint
- `GET /version` - Version info
- `POST /echo` - Echo payload
- `POST /auth/login` - Dummy login
- `GET /orders/{order_id}` - Get order
- `POST /orders/preview` - Preview order
- `POST /customer/submit` - Submit customer order

## How It Works

1. **Developer** deploys the Developer API service (separately)
2. **Agentic System** is configured with the Developer API URL
3. When the LLM decides to make an API call (via `[API_CALL: ...]` tag), the Agentic System calls the external Developer API
4. The Developer API processes the request and returns a response
5. The Agentic System formats the response and returns it to the customer

## Notes

- The Developer API is completely separate from the main Agentic System
- Developers can customize the Developer API endpoints for their specific needs
- The main Agentic System only needs the base URL to call the Developer API

