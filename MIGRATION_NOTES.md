# Migration Notes

## Separated Developer API Service

The testing/dummy endpoints have been moved to a separate service in the `developer_api/` folder.

### What Changed

1. **Removed from main service:**
   - `app/api/dummy.py` - All dummy endpoints moved to `developer_api/main.py`
   - `app/api/customer.py` - Customer endpoint moved to `developer_api/main.py`
   - Removed routers from `src/server.py`

2. **Added:**
   - `developer_api/` folder - Standalone service for developer API endpoints
   - `app/services/external_api_client.py` - Client to call external developer API
   - Updated `app/api/state.py` to use external API client instead of local endpoints

3. **Configuration:**
   - Added `EXTERNAL_API_BASE_URL` environment variable
   - Main service now calls external API instead of local endpoints

### Old Files (Can be deleted)

These files are no longer used in the main service:
- `app/api/dummy.py` (moved to `developer_api/main.py`)
- `app/api/customer.py` (moved to `developer_api/main.py`)

### Next Steps

1. Deploy `developer_api/` separately (e.g., HuggingFace Space)
2. Set `EXTERNAL_API_BASE_URL` in `.env` to point to deployed service
3. Optionally delete old files: `app/api/dummy.py` and `app/api/customer.py`

