# Frontend-Backend Connection Configuration

This document outlines how the frontend (Node.js/Browser) connects to the backend (FastAPI/PyTorch) and how to configure the ports and addresses.

## Architecture

*   **Backend**: A FastAPI server running `server.py`.
    *   **Default Mode (`llaisys`)**: Runs on `http://localhost:8000`.
    *   **PyTorch Mode (`--pytorch-model`)**: Runs on `http://localhost:8002` by default.
*   **Frontend Server**: A Node.js Express server running `frontend/server.js`.
    *   Runs on `http://localhost:3000` by default.
    *   Serves static assets (`index.html`, `app.js`).
    *   Provides a `/config` endpoint to tell the browser where the backend is.
*   **Frontend Client**: Browser-based JavaScript (`app.js`).
    *   Fetches configuration from `/config`.
    *   Connects directly to the Backend API.

## Configuration

### Backend

You can override the default port using the `--port` argument:

```bash
# Run PyTorch backend on custom port 9000
python server.py --model path/to/model --pytorch-model --port 9000
```

### Frontend

The frontend determines the backend URL via the `API_URL` environment variable or a `.env` file.

**Default Behavior:**
If no configuration is provided, the frontend assumes the backend is at `http://localhost:8002` (the PyTorch default).

**Configuration Methods:**

1.  **Environment Variables**:
    ```bash
    API_URL=http://localhost:9000 node server.js
    ```

2.  **Configuration File (.env)**:
    Create a `.env` file in the `frontend/` directory (see `.env.example`):
    ```ini
    PORT=3000
    API_URL=http://localhost:8002
    ```

## Troubleshooting Connection Issues

If the frontend cannot connect to the backend ("Connection Error" or stuck "Thinking..."):

1.  **Check Backend Status**:
    *   Ensure the python server is running.
    *   Verify the port it is listening on (look for `Uvicorn running on http://0.0.0.0:XXXX` in the logs).
    *   If using `--pytorch-model`, the default is **8002**.
    *   If using standard mode, the default is **8000**.

2.  **Check Frontend Configuration**:
    *   Open the browser Developer Tools (F12) -> Network Tab.
    *   Refresh the page.
    *   Look for the request to `/config`. Check the response JSON: `{"apiUrl": "..."}`.
    *   Ensure this URL matches your running backend.

3.  **CORS Errors**:
    *   If you see "CORS policy" errors in the browser console, ensure the backend has CORS enabled.
    *   The provided `server.py` is configured to allow all origins (`*`).

4.  **Network Reachability**:
    *   Try accessing the backend directly in your browser: `http://localhost:8002/docs`. You should see the Swagger UI.
    *   If running in Docker or a VM, ensure ports are forwarded correctly.

5.  **Integration Test**:
    *   Run `python3 test/test_connection_config.py` to verify the port selection logic and configuration handling works as expected.
