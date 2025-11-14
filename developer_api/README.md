# Developer API Service

This is a separate service that provides mock API endpoints mimicking what developers would implement for their companies. This service is meant to be deployed separately (e.g., on HuggingFace Space) and called by the main Agentic System.

## Purpose

This service provides testing/dummy endpoints that the Agentic System can call when the LLM decides to make API calls. In a real scenario, developers would implement these endpoints for their own companies.

## Endpoints

### Health & Utility
- `GET /health` - Health check
- `GET /ping` - Ping endpoint
- `GET /version` - Version information
- `POST /echo` - Echo back the payload

### Authentication
- `POST /auth/login` - Dummy login endpoint

### Orders
- `GET /orders/{order_id}` - Get order by ID
- `POST /orders/preview` - Preview order with pricing

### Customer
- `POST /customer/submit` - Submit customer order/information

## Installation

```bash
pip install -r requirements.txt
```

## Running Locally

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

The service will be available at `http://localhost:8001`

## Deployment

This service can be deployed separately on:
- HuggingFace Spaces
- Any cloud platform (AWS, GCP, Azure)
- Docker container

## Integration with Agentic System

The main Agentic System should be configured to call this service's base URL when making API calls. Update the API endpoint configuration in the Agentic System to point to this service.

Example:
- If deployed at `https://your-service.hf.space`
- The Agentic System should call `https://your-service.hf.space/customer/submit` instead of `/api/customer/submit`

