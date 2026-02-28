# LLAISYS (Let's Learn AI SYStem)

**LLAISYS** is a high-performance, educational Large Language Model (LLM) inference system built from scratch in C++, with Python bindings and a production-grade OpenAI-compatible API server.

It is designed to demonstrate key concepts in modern AI infrastructure:
*   **Custom C++ Tensor Library**: Memory management, operators, and device abstraction.
*   **Efficient Inference**: KV Cache, Prefix Caching (Session Management), and Sampling (Temp/Top-K/Top-P).
*   **Architecture**: Clean separation of Model Weights (immutable) and Session State (mutable).
*   **Application Layer**: FastAPI server with streaming support (SSE) and multi-session handling.

---

## 🏗️ Architecture

The system is composed of three main layers:

1.  **Core (C++)**: 
    *   **Runtime**: Device abstraction (CPU, Nvidia GPU via CUDA).
    *   **Tensor Engine**: ND-Array implementation with stride support.
    *   **Operators**: Optimized kernels (RoPE, RMSNorm, SwiGLU, Attention).
    *   **Model**: Static weights management.
    *   **Session**: Dynamic KV Cache state management.

2.  **Bindings (Python)**:
    *   **ctypes Interface**: Direct FFI calls to the C++ shared library.
    *   **Model Wrapper**: Pythonic API for loading weights (Safetensors) and managing inference.

3.  **Application (Python)**:
    *   **Inference Server**: FastAPI-based server implementing OpenAI's `/v1/chat/completions`.
    *   **Session Manager**: Smart routing of requests to existing sessions based on prefix matching (Prefix Caching).

---

## 🚀 Installation

### Prerequisites
*   **Compiler**: C++17 compatible (GCC, Clang, MSVC).
*   **Build Tool**: [Xmake](https://xmake.io/).
*   **Python**: >= 3.9.
*   **Dependencies**: PyTorch (for loading weights), Transformers, FastAPI, Uvicorn.

### Step-by-Step Guide

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/llaisys.git
    cd llaisys
    ```

2.  **Build C++ Backend**
    ```bash
    xmake
    xmake install
    ```
    This compiles `libllaisys.so` (or `.dylib`/`.dll`) and places it where Python can find it.

3.  **Install Python Package**
    ```bash
    pip install ./python/
    ```

4.  **Install Application Dependencies**
    ```bash
    pip install fastapi uvicorn sse-starlette pydantic transformers huggingface_hub requests prometheus_client opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-fastapi
    ```

---

## ⚙️ Configuration

The system is configured primarily via CLI arguments when starting the server or scripts.

### Backend Configuration
*   **Device**: `cpu` (default) or `nvidia` (requires CUDA build).
*   To enable CUDA, edit `xmake.lua` or configure with `xmake f --nv-gpu=y`.

### Server Configuration
*   `--model`: Path to model directory (Hugging Face format) or HF Hub ID.
*   `--host`: Host to bind (default: `0.0.0.0`).
*   `--port`: Port to bind (default: `8000`).
*   `--max-sessions`: Maximum number of concurrent sessions to keep in memory (LRU eviction).

---

## 📖 Usage

### 1. CLI Chat
Run a simple interactive chat in your terminal:

```bash
python chat.py --model Qwen/Qwen2.5-0.5B-Instruct --temperature 0.7
```

### 2. API Server
Start the production-ready API server:

```bash
python server.py --model Qwen/Qwen2.5-0.5B-Instruct --host 0.0.0.0 --port 8000 --max-sessions 20
```

### 3. Client Example
Use `curl` or any OpenAI-compatible client to interact with the server:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2",
    "messages": [{"role": "user", "content": "Hello, how does KV Cache work?"}],
    "stream": true,
    "temperature": 0.7
  }'
```

### 4. Start Observability Stack

The project includes a comprehensive observability stack powered by Prometheus, Grafana, and Jaeger.

#### Docker Commands

1.  **Start Services**:
    ```bash
    docker-compose up -d
    ```
    This starts Prometheus, Grafana, and Jaeger in the background.

2.  **Stop Services**:
    ```bash
    docker-compose down
    ```

3.  **View Logs**:
    ```bash
    docker-compose logs -f
    ```

#### Accessing Dashboards

*   **Grafana**: [http://localhost:3001](http://localhost:3001)
    *   **Credentials**: `admin` / `admin`
    *   **Dashboard**: Go to *Dashboards* > *LLAISYS Performance Dashboard* to see real-time metrics.
    *   **Key Metrics**:
        *   **Request Rate**: Requests per second (QPS).
        *   **Application Latency**: P95/P99 latency of chat completions (excluding metrics overhead).
        *   **Error Rate**: Percentage of failed requests.
        *   **Metric Endpoint Performance**: Monitoring of the monitoring system itself.

*   **Prometheus**: [http://localhost:9090](http://localhost:9090)
    *   Use this to query raw metrics or debug scraping targets.
    *   **Sample Queries**:
        *   *Total Request Rate (Global)*:
            ```promql
            sum(rate(llaisys_requests_total[1m]))
            ```
        *   *P99 Latency by Endpoint*:
            ```promql
            histogram_quantile(0.99, sum(rate(llaisys_request_latency_seconds_bucket[5m])) by (le, endpoint))
            ```
        *   *Error Rate %*:
            ```promql
            sum(rate(llaisys_requests_total{status=~"5.*"}[5m])) / sum(rate(llaisys_requests_total[5m])) * 100
            ```

*   **Jaeger**: [http://localhost:16686](http://localhost:16686)
    *   Use this to view distributed traces for request debugging.
    *   **Search Tips**:
        *   **Service**: Select `fastapi` (or configured service name) in the Service dropdown.
        *   **Tags**: Filter by specific attributes, e.g., `http.status_code=200` or `http.method="POST"`.
        *   **Min Duration**: Set e.g., `500ms` to find slow requests.

### 5. Python SDK Usage
```python
import llaisys

# Load Model
model = llaisys.models.Qwen2("path/to/model")

# Create Session
session = model.create_session()

# Generate
output_tokens = model.generate(
    inputs=[1, 2, 3],
    max_new_tokens=20,
    session=session
)
```

### 6. Frontend Usage
Run the dedicated frontend server for a Chat UI:
```bash
cd frontend
npm install
node server.js
```
Visit http://localhost:3000 to start chatting.

---

## 🧪 Development

### Running Tests
We have a comprehensive test suite covering Ops, Tensor logic, and Server logic.

```bash
# Run unit tests for C++ operators (via Python wrapper)
python test/test_ops.py

# Run server logic tests (Session Management)
python test/test_server_logic.py
```

### Directory Structure
*   `src/`: C++ source code.
*   `python/`: Python package source.
*   `include/`: C API headers.
*   `test/`: Test scripts.

---

## 🤝 Contribution

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/amazing-feature`).
3.  Commit your changes.
4.  Push to the branch.
5.  Open a Pull Request.

Please ensure all tests pass (`xmake && pip install ./python/ && python test/test_ops.py`) before submitting.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
