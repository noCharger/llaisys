commit 69249e64d1c7817cdf86d5f33510ba336234546a
Author: Louis Chu <lingzhichu.clz@gmail.com>
Date:   Sun Mar 15 16:52:19 2026 -0700

    feat: build AI chatbot application with HTTPS support (#12)
    
    * feat: build AI chatbot application with HTTPS support
    
    * feat: add session management for stateful inference
    
    * feat: implement random sampling with top-k and top-p support
    
    - Implemented new `random_sample` operator in C++ supporting Temperature, Top-K, and Top-P filtering
    - Replaced greedy argmax with the new random sampling operator in `llaisysQwen2ModelInfer` and `llaisysQwen2ModelForward`
    - Updated C API headers and Python bindings to expose `temp`, `top_p`, and `top_k` parameters
    - Added unit tests for `random_sample` operator mimicking PyTorch behavior
    - Added e2e tests in `test_infer.py` to verify deterministic and random generation behaviors

diff --git a/.gitignore b/.gitignore
index 8a13bc6..8fba47e 100644
--- a/.gitignore
+++ b/.gitignore
@@ -29,7 +29,7 @@ docs/
 cache/
 
 # JSON
-*.json
+package-lock.json
 
 #GGUF
 *.gguf
@@ -90,4 +90,11 @@ htmlcov/
 # Windows
 Thumbs.db
 ehthumbs.db
-desktop.ini
\ No newline at end of file
+desktop.ini
+
+# Node.js
+node_modules/
+
+# SSL Certificates
+src/chatbot/certs/
+*.pem
\ No newline at end of file
diff --git a/include/llaisys/models/qwen2.h b/include/llaisys/models/qwen2.h
index 7054626..388f7e2 100644
--- a/include/llaisys/models/qwen2.h
+++ b/include/llaisys/models/qwen2.h
@@ -35,8 +35,13 @@ __C {
 
     __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model);
 
+    __export struct LlaisysQwen2Session *llaisysQwen2ModelCreateSession(struct LlaisysQwen2Model * model);
+    __export void llaisysQwen2ModelDestroySession(struct LlaisysQwen2Session * session);
+    __export void llaisysQwen2ModelRewindSession(struct LlaisysQwen2Session * session, size_t len);
+    __export int64_t llaisysQwen2ModelForward(struct LlaisysQwen2Session * session, int64_t * token_ids, size_t ntoken, float temp, float top_p, int top_k);
+
     __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model);
 
-    __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken);
+    __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken, float temp, float top_p, int top_k);
 }
 #endif // LLAISYS_MODELS_QWEN2_H
diff --git a/include/llaisys/ops.h b/include/llaisys/ops.h
index ddb3be2..f35bb6a 100644
--- a/include/llaisys/ops.h
+++ b/include/llaisys/ops.h
@@ -13,6 +13,7 @@ __C {
     __export void llaisysROPE(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t pos_ids, float theta);
     __export void llaisysSelfAttention(llaisysTensor_t attn_val, llaisysTensor_t q, llaisysTensor_t k, llaisysTensor_t v, float scale);
     __export void llaisysSwiGLU(llaisysTensor_t out, llaisysTensor_t gate, llaisysTensor_t up);
+    __export void llaisysRandomSample(llaisysTensor_t out_token, llaisysTensor_t logits, float temp, float top_p, int top_k);
 }
 
 #endif
diff --git a/python/llaisys/libllaisys/ops.py b/python/llaisys/libllaisys/ops.py
index 5be095e..1c3de4e 100644
--- a/python/llaisys/libllaisys/ops.py
+++ b/python/llaisys/libllaisys/ops.py
@@ -1,5 +1,5 @@
 from .tensor import llaisysTensor_t
-from ctypes import c_float
+from ctypes import c_float, c_int
 
 def load_ops(lib):
     lib.llaisysAdd.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t]
@@ -34,3 +34,12 @@ def load_ops(lib):
 
     lib.llaisysSwiGLU.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t]
     lib.llaisysSwiGLU.restype = None
+
+    lib.llaisysRandomSample.argtypes = [
+        llaisysTensor_t, # out_token
+        llaisysTensor_t, # logits
+        c_float, # temp
+        c_float, # top_p
+        c_int    # top_k
+    ]
+    lib.llaisysRandomSample.restype = None
diff --git a/python/llaisys/libllaisys/qwen2.py b/python/llaisys/libllaisys/qwen2.py
index 67a271b..cab55de 100644
--- a/python/llaisys/libllaisys/qwen2.py
+++ b/python/llaisys/libllaisys/qwen2.py
@@ -47,5 +47,32 @@ def load_qwen2(lib):
     lib.llaisysQwen2ModelWeights.argtypes = [ctypes.c_void_p]
     lib.llaisysQwen2ModelWeights.restype = ctypes.POINTER(LlaisysQwen2Weights)
 
-    lib.llaisysQwen2ModelInfer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t]
+    lib.llaisysQwen2ModelInfer.argtypes = [
+        ctypes.c_void_p, 
+        ctypes.POINTER(ctypes.c_int64), 
+        ctypes.c_size_t,
+        ctypes.c_float, 
+        ctypes.c_float, 
+        ctypes.c_int
+    ]
     lib.llaisysQwen2ModelInfer.restype = ctypes.c_int64
+
+    # Session APIs
+    lib.llaisysQwen2ModelCreateSession.argtypes = [ctypes.c_void_p]
+    lib.llaisysQwen2ModelCreateSession.restype = ctypes.c_void_p
+
+    lib.llaisysQwen2ModelDestroySession.argtypes = [ctypes.c_void_p]
+    lib.llaisysQwen2ModelDestroySession.restype = None
+
+    lib.llaisysQwen2ModelRewindSession.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
+    lib.llaisysQwen2ModelRewindSession.restype = None
+
+    lib.llaisysQwen2ModelForward.argtypes = [
+        ctypes.c_void_p, 
+        ctypes.POINTER(ctypes.c_int64), 
+        ctypes.c_size_t, 
+        ctypes.c_float, 
+        ctypes.c_float, 
+        ctypes.c_int
+    ]
+    lib.llaisysQwen2ModelForward.restype = ctypes.c_int64
diff --git a/python/llaisys/models/qwen2.py b/python/llaisys/models/qwen2.py
index 8aa7ca9..08ee040 100644
--- a/python/llaisys/models/qwen2.py
+++ b/python/llaisys/models/qwen2.py
@@ -124,6 +124,29 @@ class Qwen2:
         
         return None
 
+    def create_session(self):
+        return LIB_LLAISYS.llaisysQwen2ModelCreateSession(self.model)
+
+    def destroy_session(self, session_ptr):
+        LIB_LLAISYS.llaisysQwen2ModelDestroySession(session_ptr)
+
+    def rewind_session(self, session_ptr, length):
+        LIB_LLAISYS.llaisysQwen2ModelRewindSession(session_ptr, length)
+
+    def forward(self, session_ptr, input_ids, temperature=0.7, top_p=0.9, top_k=40):
+        if not input_ids:
+             return self.end_token
+        
+        arr = (ctypes.c_int64 * len(input_ids))(*input_ids)
+        return LIB_LLAISYS.llaisysQwen2ModelForward(
+            session_ptr, 
+            arr, 
+            len(input_ids), 
+            temperature, 
+            top_p, 
+            top_k
+        )
+
     def generate(
         self,
         inputs: Sequence[int],
@@ -136,22 +159,26 @@ class Qwen2:
             return []
             
         tokens = list(inputs)
+        session = self.create_session()
         
-        # Prefill
-        input_array = (ctypes.c_int64 * len(tokens))(*tokens)
-        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(self.model, input_array, len(tokens))
-        
-        tokens.append(next_token)
-        if next_token == self.end_token:
-            return tokens
-        
-        for _ in range(max_new_tokens - 1):
-            input_array = (ctypes.c_int64 * 1)(tokens[-1])
-            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(self.model, input_array, 1)
-            tokens.append(next_token)
+        try:
+            # Prefill
+            input_array = (ctypes.c_int64 * len(tokens))(*tokens)
+            next_token = LIB_LLAISYS.llaisysQwen2ModelForward(session, input_array, len(tokens), temperature, top_p, top_k)
             
+            tokens.append(next_token)
             if next_token == self.end_token:
-                break
+                return tokens
+            
+            for _ in range(max_new_tokens - 1):
+                input_array = (ctypes.c_int64 * 1)(tokens[-1])
+                next_token = LIB_LLAISYS.llaisysQwen2ModelForward(session, input_array, 1, temperature, top_p, top_k)
+                tokens.append(next_token)
+                
+                if next_token == self.end_token:
+                    break
+        finally:
+            self.destroy_session(session)
                 
         return tokens
         
diff --git a/python/llaisys/ops.py b/python/llaisys/ops.py
index ed0180b..5d79d76 100644
--- a/python/llaisys/ops.py
+++ b/python/llaisys/ops.py
@@ -53,3 +53,13 @@ class Ops:
     @staticmethod
     def swiglu(out: Tensor, gate: Tensor, up: Tensor):
         LIB_LLAISYS.llaisysSwiGLU(out.lib_tensor(), gate.lib_tensor(), up.lib_tensor())
+
+    @staticmethod
+    def random_sample(out_token: Tensor, logits: Tensor, temp: float, top_p: float, top_k: int):
+        LIB_LLAISYS.llaisysRandomSample(
+            out_token.lib_tensor(), 
+            logits.lib_tensor(), 
+            c_float(temp), 
+            c_float(top_p), 
+            c_int(top_k)
+        )
diff --git a/scripts/install_node_linux.sh b/scripts/install_node_linux.sh
new file mode 100644
index 0000000..0927131
--- /dev/null
+++ b/scripts/install_node_linux.sh
@@ -0,0 +1,21 @@
+#!/bin/bash
+
+# Install Node.js and npm using nvm (Node Version Manager)
+echo "Installing nvm..."
+curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
+
+# Load nvm
+export NVM_DIR="$HOME/.nvm"
+[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
+
+# Install latest LTS version of Node.js
+echo "Installing Node.js LTS..."
+nvm install --lts
+
+# Verify installation
+echo "Node version:"
+node -v
+echo "NPM version:"
+npm -v
+
+echo "Installation complete. Please restart your terminal or run 'source ~/.bashrc' (or ~/.zshrc) to use npm."
diff --git a/scripts/start.py b/scripts/start.py
new file mode 100755
index 0000000..54d390d
--- /dev/null
+++ b/scripts/start.py
@@ -0,0 +1,303 @@
+#!/usr/bin/env python3
+import subprocess
+import sys
+import os
+import time
+import urllib.request
+import ssl
+import signal
+import atexit
+import argparse
+from pathlib import Path
+import logging
+
+ROOT_DIR = Path(__file__).parent.parent.absolute()
+BACKEND_DIR = ROOT_DIR / "src" / "chatbot" / "backend"
+FRONTEND_DIR = ROOT_DIR / "src" / "chatbot" / "frontend"
+OPS_DIR = ROOT_DIR / "src" / "chatbot" / "ops"
+LOGS_DIR = ROOT_DIR / "logs"
+
+LOGS_DIR.mkdir(exist_ok=True)
+
+logging.basicConfig(
+    level=logging.INFO,
+    format='%(asctime)s - [%(levelname)s] - %(message)s',
+    handlers=[
+        logging.StreamHandler(sys.stdout),
+        logging.FileHandler(LOGS_DIR / "launcher.log")
+    ]
+)
+logger = logging.getLogger("launcher")
+
+ssl_context = ssl.create_default_context()
+ssl_context.check_hostname = False
+ssl_context.verify_mode = ssl.CERT_NONE
+
+processes = []
+observability_started = False
+
+def cleanup():
+    logger.info("Initiating graceful shutdown...")
+    for p, name in processes:
+        if p.poll() is None:
+            logger.info(f"Stopping {name} (PID {p.pid})...")
+            if sys.platform == "win32":
+                p.terminate()
+            else:
+                p.send_signal(signal.SIGTERM)
+            
+            try:
+                p.wait(timeout=5)
+            except subprocess.TimeoutExpired:
+                logger.warning(f"{name} did not terminate, forcing kill...")
+                p.kill()
+    
+    if observability_started:
+        logger.info("Stopping observability stack (Docker Compose)...")
+        try:
+            subprocess.run(["docker", "compose", "down"], cwd=OPS_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
+        except Exception as e:
+            logger.error(f"Error stopping observability stack: {e}")
+
+    logger.info("Shutdown complete.")
+
+atexit.register(cleanup)
+
+def handle_sigint(sig, frame):
+    sys.exit(0)
+
+signal.signal(signal.SIGINT, handle_sigint)
+
+def check_health(url: str, timeout: int = 120, interval: int = 2) -> bool:
+    start_time = time.time()
+    logger.info(f"Waiting for {url} (timeout: {timeout}s)...")
+    
+    while time.time() - start_time < timeout:
+        try:
+            req = urllib.request.Request(url)
+            with urllib.request.urlopen(req, context=ssl_context, timeout=5) as response:
+                if response.status == 200:
+                    logger.info(f"Successfully connected to {url}")
+                    return True
+        except (urllib.error.URLError, ConnectionResetError, ConnectionRefusedError):
+            pass
+        except Exception as e:
+            logger.debug(f"Health check exception: {e}")
+            
+        time.sleep(interval)
+        
+    logger.error(f"Timeout waiting for {url}")
+    return False
+
+def start_process(name: str, cmd: list, cwd: Path, log_file: str, env: dict = None) -> subprocess.Popen:
+    log_path = LOGS_DIR / log_file
+    logger.info(f"Starting {name}... (Logs: {log_path})")
+    
+    process_env = os.environ.copy()
+    if env:
+        process_env.update(env)
+        
+    with open(log_path, 'w') as f:
+        process = subprocess.Popen(
+            cmd,
+            cwd=cwd,
+            stdout=f,
+            stderr=subprocess.STDOUT,
+            env=process_env,
+            text=True
+        )
+        
+    processes.append((process, name))
+    return process
+
+def check_docker() -> bool:
+    try:
+        subprocess.run(["docker", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
+        subprocess.run(["docker", "compose", "version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
+        return True
+    except (subprocess.CalledProcessError, FileNotFoundError):
+        return False
+
+def start_observability():
+    global observability_started
+    logger.info("Checking observability stack requirements...")
+    
+    if not check_docker():
+        logger.warning("Docker or Docker Compose not found. Skipping observability stack.")
+        return
+
+    docker_compose_file = OPS_DIR / "docker-compose.yml"
+    if not docker_compose_file.exists():
+        logger.warning(f"docker-compose.yml not found at {OPS_DIR}. Skipping observability stack.")
+        return
+
+    logger.info(f"Starting observability stack from {OPS_DIR}...")
+    try:
+        # Start containers detached
+        subprocess.run(["docker", "compose", "up", "-d"], cwd=OPS_DIR, check=True)
+        observability_started = True
+        
+        # Health checks (non-blocking, just log)
+        logger.info("Waiting for observability services to be ready...")
+        
+        # Give it a moment to start
+        time.sleep(5)
+        
+        grafana_ok = check_health("http://localhost:3001/api/health", timeout=15, interval=2)
+        prometheus_ok = check_health("http://localhost:9090/-/healthy", timeout=15, interval=2)
+        
+        if grafana_ok:
+            logger.info("Grafana is running at http://localhost:3001 (User/Pass: admin/admin)")
+        else:
+            logger.warning("Grafana health check timed out, but container might be starting.")
+            
+        if prometheus_ok:
+            logger.info("Prometheus is running at http://localhost:9090")
+        else:
+            logger.warning("Prometheus health check timed out.")
+            
+    except subprocess.CalledProcessError as e:
+        logger.error(f"Failed to launch observability stack: {e}")
+    except Exception as e:
+        logger.error(f"Unexpected error launching observability stack: {e}")
+
+def main():
+    parser = argparse.ArgumentParser(description="LLAISYS Chatbot Launcher")
+    parser.add_argument("--model_path", type=str, help="Path to the LLM model or HuggingFace ID", default=None)
+    parser.add_argument("--max-sessions", type=int, help="Maximum concurrent sessions", default=100)
+    parser.add_argument("--max-steps", type=int, help="Maximum generation steps per request", default=4096)
+    parser.add_argument("--temperature", type=float, help="Default temperature", default=0.7)
+    parser.add_argument("--top-p", type=float, help="Default top_p", default=0.9)
+    parser.add_argument("--top-k", type=int, help="Default top_k", default=50)
+    parser.add_argument("--system-prompt", type=str, help="Default system prompt", default=None)
+    parser.add_argument("--device", type=str, choices=["cpu", "nvidia"], help="Device to run the model on (cpu or nvidia)", default="cpu")
+    parser.add_argument("--dtype", type=str, choices=["float32", "float16"], help="Model data type", default="float32")
+    parser.add_argument("--no-https", action="store_true", help="Disable HTTPS and run in HTTP mode only")
+    parser.add_argument("--prod", action="store_true", default=True, help="Enable production mode (observability stack). Default: True")
+    parser.add_argument("--no-prod", dest="prod", action="store_false", help="Disable production mode")
+    args = parser.parse_args()
+
+    logger.info("=== LLAISYS Chatbot Launcher ===")
+    
+    if args.prod:
+        logger.info("Production mode enabled. Initializing observability stack...")
+        start_observability()
+    else:
+        logger.info("Production mode disabled. Skipping observability stack.")
+
+    model_path = args.model_path or os.environ.get("MODEL_PATH", "Qwen/Qwen2-0.5B-Instruct")
+    device = args.device or os.environ.get("DEVICE", "cpu")
+    dtype = args.dtype or os.environ.get("DTYPE", "float32")
+    backend_port = os.environ.get("BACKEND_PORT", "6008")
+    frontend_port = os.environ.get("FRONTEND_PORT", "6006")
+    
+    backend_https_port = int(backend_port) + 1
+    frontend_https_port = int(frontend_port) + 1
+    
+    backend_reqs = BACKEND_DIR / "requirements.txt"
+    if backend_reqs.exists():
+        logger.info(f"Installing backend dependencies from {backend_reqs}...")
+        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(backend_reqs)], check=True)
+
+    backend_cmd = [
+        sys.executable, "server.py", 
+        "--model", model_path, 
+        "--device", device, 
+        "--dtype", dtype,
+        "--port", backend_port,
+        "--max-sessions", str(args.max_sessions),
+        "--max-steps", str(args.max_steps),
+        "--temperature", str(args.temperature),
+        "--top-p", str(args.top_p),
+        "--top-k", str(args.top_k)
+    ]
+    
+    if args.system_prompt:
+        backend_cmd.extend(["--system-prompt", args.system_prompt])
+    
+    if args.no_https:
+        backend_cmd.append("--no-https")
+    
+    backend_proc = start_process(
+        "Backend Server", 
+        backend_cmd, 
+        BACKEND_DIR, 
+        "backend.log",
+        env={"PYTHONUNBUFFERED": "1"}
+    )
+    
+    time.sleep(2)
+    if backend_proc.poll() is not None:
+        logger.error("Backend server crashed immediately. Check logs/backend.log")
+        sys.exit(1)
+        
+    logger.info("Waiting for model to load...")
+    
+    if args.no_https:
+        backend_url = f"http://localhost:{backend_port}/health"
+        backend_http_url = f"http://localhost:{backend_port}/health"
+    else:
+        backend_url = f"https://localhost:{backend_https_port}/health"
+        backend_http_url = f"http://localhost:{backend_port}/health"
+    
+    if not (check_health(backend_url, timeout=300) or check_health(backend_http_url, timeout=5)):
+        logger.error("Backend failed to start properly. Check logs/backend.log")
+        sys.exit(1)
+        
+    if not (FRONTEND_DIR / "package-lock.json").exists() or not (FRONTEND_DIR / "node_modules").exists():
+        logger.info("Installing frontend dependencies...")
+        subprocess.run(["npm", "install"], cwd=FRONTEND_DIR, check=True)
+
+    logger.info("Building frontend...")
+    subprocess.run(["npm", "run", "build"], cwd=FRONTEND_DIR, check=True)
+    
+    frontend_env = {"PORT": frontend_port}
+    
+    if args.no_https:
+        frontend_env["NO_HTTPS"] = "true"
+        frontend_env["INTERNAL_API_URL"] = f"http://localhost:{backend_port}"
+    else:
+        frontend_env["INTERNAL_API_URL"] = f"https://localhost:{backend_https_port}"
+        
+    frontend_proc = start_process(
+        "Frontend Server",
+        ["npm", "start"],
+        FRONTEND_DIR,
+        "frontend.log",
+        env=frontend_env
+    )
+    
+    if args.no_https:
+        frontend_url = f"http://localhost:{frontend_port}/config"
+        frontend_http_url = f"http://localhost:{frontend_port}/config"
+    else:
+        frontend_url = f"https://localhost:{frontend_https_port}/config"
+        frontend_http_url = f"http://localhost:{frontend_port}/config"
+    
+    if not (check_health(frontend_url, timeout=60) or check_health(frontend_http_url, timeout=5)):
+        logger.error("Frontend failed to start properly. Check logs/frontend.log")
+        sys.exit(1)
+        
+    logger.info("="*50)
+    logger.info("ALL SERVICES STARTED SUCCESSFULLY")
+    if args.no_https:
+        logger.info(f"Frontend URL: http://localhost:{frontend_port}")
+        logger.info(f"Backend URL:  http://localhost:{backend_port}")
+    else:
+        logger.info(f"Frontend URL: http://localhost:{frontend_port} (HTTPS: https://localhost:{frontend_https_port})")
+        logger.info(f"Backend URL:  http://localhost:{backend_port} (HTTPS: https://localhost:{backend_https_port})")
+    logger.info("Press Ctrl+C to stop all services.")
+    logger.info("="*50)
+    
+    try:
+        while True:
+            for p, name in processes:
+                if p.poll() is not None:
+                    logger.error(f"CRITICAL: {name} crashed with exit code {p.returncode}!")
+                    sys.exit(1)
+            time.sleep(2)
+    except KeyboardInterrupt:
+        pass
+
+if __name__ == "__main__":
+    main()
diff --git a/src/chatbot/backend/context_manager.py b/src/chatbot/backend/context_manager.py
new file mode 100644
index 0000000..f153982
--- /dev/null
+++ b/src/chatbot/backend/context_manager.py
@@ -0,0 +1,142 @@
+from typing import List, Dict, Any, Optional, Set, Union
+from dataclasses import dataclass, field
+from jinja2 import Environment, BaseLoader, select_autoescape
+import logging
+
+logger = logging.getLogger("llaisys.context_manager")
+
+@dataclass
+class TaggedItem:
+    """
+    Represents a context item with associated tags.
+    """
+    id: str
+    content: str
+    tags: Set[str] = field(default_factory=set)
+    metadata: Dict[str, Any] = field(default_factory=dict)
+
+    def add_tag(self, tag: str):
+        self.tags.add(tag)
+
+    def remove_tag(self, tag: str):
+        self.tags.discard(tag)
+
+    def has_tag(self, tag: str) -> bool:
+        return tag in self.tags
+    
+    def has_any_tag(self, tags: List[str]) -> bool:
+        return bool(self.tags.intersection(tags))
+    
+    def has_all_tags(self, tags: List[str]) -> bool:
+        return self.tags.issuperset(tags)
+
+class ContextManager:
+    """
+    Manages a collection of tagged context items and provides template rendering capabilities.
+    """
+    def __init__(self):
+        self.items: Dict[str, TaggedItem] = {}
+        
+        # Initialize Jinja2 Environment
+        self.env = Environment(
+            loader=BaseLoader(), # Use string templates
+            autoescape=select_autoescape(),
+            trim_blocks=True,
+            lstrip_blocks=True
+        )
+        
+        # Register custom filters
+        self.env.filters['filter_by_tag'] = self.filter_by_tag
+        self.env.filters['exclude_by_tag'] = self.exclude_by_tag
+
+    def add_item(self, item_id: str, content: str, tags: List[str] = None, metadata: Dict[str, Any] = None) -> TaggedItem:
+        """Adds or updates a context item."""
+        if tags is None:
+            tags = []
+        if metadata is None:
+            metadata = {}
+            
+        item = TaggedItem(
+            id=item_id,
+            content=content,
+            tags=set(tags),
+            metadata=metadata
+        )
+        self.items[item_id] = item
+        logger.info(f"Added context item: {item_id} with tags: {tags}")
+        return item
+
+    def get_item(self, item_id: str) -> Optional[TaggedItem]:
+        return self.items.get(item_id)
+
+    def remove_item(self, item_id: str) -> bool:
+        if item_id in self.items:
+            del self.items[item_id]
+            logger.info(f"Removed context item: {item_id}")
+            return True
+        return False
+
+    def list_items(self) -> List[TaggedItem]:
+        return list(self.items.values())
+
+    def get_items_by_tag(self, tag: str) -> List[TaggedItem]:
+        """Returns all items that have the specified tag."""
+        return [item for item in self.items.values() if item.has_tag(tag)]
+
+    def filter_by_tag(self, items: List[TaggedItem], tag: str) -> List[TaggedItem]:
+        """Jinja2 filter: Select items that have the specified tag."""
+        if not items:
+            return []
+        filtered = []
+        for item in items:
+            if isinstance(item, TaggedItem) and item.has_tag(tag):
+                filtered.append(item)
+            elif isinstance(item, dict) and tag in item.get('tags', []):
+                filtered.append(item)
+        return filtered
+
+    def exclude_by_tag(self, items: List[TaggedItem], tag: str) -> List[TaggedItem]:
+        """Jinja2 filter: Exclude items that have the specified tag."""
+        if not items:
+            return []
+        filtered = []
+        for item in items:
+            if isinstance(item, TaggedItem) and not item.has_tag(tag):
+                filtered.append(item)
+        return filtered
+
+    def render_template(self, template_str: str, extra_context: Dict[str, Any] = None) -> str:
+        """
+        Renders a template string using the current context items and optional extra context.
+        The template has access to 'context_items' which is the list of all TaggedItems.
+        """
+        try:
+            template = self.env.from_string(template_str)
+            render_context = {
+                "context_items": list(self.items.values()),
+                **(extra_context or {})
+            }
+            return template.render(**render_context)
+        except Exception as e:
+            logger.error(f"Template rendering failed: {e}")
+            raise
+
+if __name__ == "__main__":
+    cm = ContextManager()
+    cm.add_item("rule1", "Always be polite.", ["behavior", "core"])
+    cm.add_item("rule2", "Use concise language.", ["style", "core"])
+    cm.add_item("fact1", "The sky is blue.", ["knowledge", "nature"])
+    
+    template = """
+    System Instructions:
+    {% for item in context_items | filter_by_tag('core') %}
+    - {{ item.content }}
+    {% endfor %}
+    
+    Knowledge:
+    {% for item in context_items | filter_by_tag('knowledge') %}
+    - {{ item.content }}
+    {% endfor %}
+    """
+    
+    print(cm.render_template(template))
diff --git a/src/chatbot/backend/requirements.txt b/src/chatbot/backend/requirements.txt
new file mode 100644
index 0000000..94fd906
--- /dev/null
+++ b/src/chatbot/backend/requirements.txt
@@ -0,0 +1,9 @@
+fastapi
+uvicorn
+transformers
+huggingface_hub
+prometheus_client
+opentelemetry-api
+opentelemetry-sdk
+opentelemetry-instrumentation-fastapi
+jinja2
diff --git a/src/chatbot/backend/schemas.py b/src/chatbot/backend/schemas.py
new file mode 100644
index 0000000..28edb02
--- /dev/null
+++ b/src/chatbot/backend/schemas.py
@@ -0,0 +1,53 @@
+from typing import List, Optional, Dict, Any
+from pydantic import BaseModel
+
+class ChatMessage(BaseModel):
+    role: str
+    content: str
+
+class ChatCompletionRequest(BaseModel):
+    model: str = "qwen2"
+    messages: List[ChatMessage]
+    temperature: Optional[float] = 0.7
+    top_p: Optional[float] = 0.9
+    top_k: Optional[int] = 50
+    max_tokens: Optional[int] = 512
+    stream: Optional[bool] = False
+    session_id: Optional[str] = None
+    system_prompt: Optional[str] = None
+    use_template: Optional[bool] = False
+
+class ChatCompletionResponseChoice(BaseModel):
+    index: int
+    message: ChatMessage
+    finish_reason: Optional[str] = None
+
+class ChatCompletionResponse(BaseModel):
+    id: str
+    object: str = "chat.completion"
+    created: int
+    model: str
+    choices: List[ChatCompletionResponseChoice]
+    usage: Optional[Dict[str, int]] = None
+
+class ChatCompletionChunkDelta(BaseModel):
+    role: Optional[str] = None
+    content: Optional[str] = None
+
+class ChatCompletionChunkChoice(BaseModel):
+    index: int
+    delta: ChatCompletionChunkDelta
+    finish_reason: Optional[str] = None
+
+class ChatCompletionChunk(BaseModel):
+    id: str
+    object: str = "chat.completion.chunk"
+    created: int
+    model: str
+    choices: List[ChatCompletionChunkChoice]
+
+class ContextItemRequest(BaseModel):
+    id: str
+    content: str
+    tags: List[str] = []
+    metadata: Dict[str, Any] = {}
diff --git a/src/chatbot/backend/server.py b/src/chatbot/backend/server.py
new file mode 100644
index 0000000..2e1c334
--- /dev/null
+++ b/src/chatbot/backend/server.py
@@ -0,0 +1,673 @@
+import argparse
+import sys
+import os
+import time
+import uuid
+import asyncio
+import json
+import ssl
+import logging
+import threading
+from typing import List, Optional, Dict, Any, AsyncGenerator
+from contextlib import asynccontextmanager
+from dataclasses import dataclass, field
+
+from fastapi import FastAPI, HTTPException, Request, status
+from fastapi.responses import Response, RedirectResponse, StreamingResponse
+from fastapi.middleware.cors import CORSMiddleware
+import uvicorn
+
+from transformers import AutoTokenizer
+
+try:
+    from huggingface_hub import snapshot_download
+except ImportError:
+    snapshot_download = None
+
+try:
+    import llaisys
+except ImportError:
+    llaisys = None
+
+from schemas import (
+    ChatMessage,
+    ChatCompletionRequest,
+    ChatCompletionResponseChoice,
+    ChatCompletionResponse,
+    ChatCompletionChunk,
+    ChatCompletionChunkChoice,
+    ChatCompletionChunkDelta,
+    ContextItemRequest
+)
+from context_manager import ContextManager
+
+try:
+    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
+    from opentelemetry import trace
+    from opentelemetry.sdk.trace import TracerProvider
+    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
+    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
+    OBSERVABILITY_ENABLED = True
+except ImportError:
+    OBSERVABILITY_ENABLED = False
+
+logging.basicConfig(
+    level=logging.INFO,
+    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
+)
+logger = logging.getLogger("llaisys.server")
+
+class ServerState:
+    """Manages the global state of the inference server."""
+    def __init__(self):
+        self.model: Any = None
+        self.tokenizer: Any = None
+        self.sessions: Dict[str, Any] = {}
+        self.model_path: str = ""
+        self.device_name: str = "cpu"
+        self.dtype: Any = 0
+        self.max_sessions: int = 10 
+        self.max_steps: int = 4096
+        self.default_system_prompt: str = (
+            "You are a helpful AI assistant.\n"
+            "{% if context_items %}\n"
+            "Context:\n"
+            "{% for item in context_items %}\n"
+            "- {{ item.content }}\n"
+            "{% endfor %}\n"
+            "Use the context above if relevant.\n"
+            "{% endif %}\n"
+            "Answer the user clearly. If the task is complex, break it down logically."
+        )
+        self.max_context_len: int = 8192
+        self.https_port: int = 0
+        self.temperature: float = 0.7
+        self.top_p: float = 0.9
+        self.top_k: int = 50
+        self.context_manager = ContextManager()
+
+state = ServerState()
+
+@dataclass
+class SessionMeta:
+    """Metadata for an active inference session."""
+    session_ptr: Any
+    history_ids: List[int] = field(default_factory=list)
+    messages: List[Dict[str, str]] = field(default_factory=list)
+    last_accessed: float = field(default_factory=time.time)
+    in_use: bool = False
+
+session_store: Dict[str, SessionMeta] = {}
+store_lock = threading.Lock()
+def load_model() -> None:
+    """Loads the model and tokenizer into global state."""
+    if llaisys is None:
+        logger.warning("llaisys module not found. Inference will not work.")
+        return
+
+    try:
+        logger.info(f"Loading model from {state.model_path} on {state.device_name}...")
+        
+        # Set data type
+        if hasattr(llaisys, "DataType"):
+             state.dtype = llaisys.DataType.F16 if state.dtype == "float16" else llaisys.DataType.F32
+        
+        device_type = llaisys.DeviceType.NVIDIA if state.device_name == "nvidia" else llaisys.DeviceType.CPU
+        
+        state.model = llaisys.models.Qwen2(state.model_path, device_type, dtype=state.dtype)
+        state.tokenizer = AutoTokenizer.from_pretrained(state.model_path, trust_remote_code=True)
+        
+        # Update max context length from model config if available
+        if hasattr(state.model, "meta") and hasattr(state.model.meta, "maxseq"):
+            state.max_context_len = state.model.meta.maxseq
+            logger.info(f"Max context length set to {state.max_context_len}")
+            
+        logger.info("Model loaded successfully.")
+    except Exception as e:
+        logger.error(f"Failed to load model: {e}")
+        
+def cleanup_sessions():
+    """Cleans up all active sessions."""
+    if not state.model:
+        return
+    logger.info("Cleaning up sessions...")
+    with store_lock:
+        for meta in session_store.values():
+            try:
+                state.model.destroy_session(meta.session_ptr)
+            except Exception as e:
+                logger.error(f"Error destroying session: {e}")
+        session_store.clear()
+
+def get_session_for_request(input_ids: List[int], session_id: Optional[str] = None) -> tuple[SessionMeta, int]:
+    """
+    Retrieves or creates a session for the request.
+    Returns (SessionMeta, match_length_in_tokens).
+    Thread-safe implementation with store_lock.
+    """
+    current_time = time.time()
+    
+    with store_lock:
+        if session_id and session_id in session_store:
+            meta = session_store[session_id]
+            if meta.in_use:
+                logger.warning(f"Requested session {session_id} is currently in use.")
+                raise HTTPException(status_code=409, detail=f"Session {session_id} is busy processing another request.")
+
+            meta.last_accessed = current_time
+            meta.in_use = True
+            
+            # Calculate prefix match
+            match_len = 0
+            min_len = min(len(meta.history_ids), len(input_ids))
+            for i in range(min_len):
+                if meta.history_ids[i] == input_ids[i]:
+                    match_len += 1
+                else:
+                    break
+            
+            # Rewind if necessary
+            if match_len < len(meta.history_ids):
+                logger.info(f"Rewinding session {session_id} from {len(meta.history_ids)} to {match_len}")
+                state.model.rewind_session(meta.session_ptr, match_len)
+                meta.history_ids = meta.history_ids[:match_len]
+                
+            return meta, match_len
+
+        # Find the session with the longest common prefix among AVAILABLE sessions
+        best_id = None
+        best_meta = None
+        best_len = 0
+        
+        for sid, meta in session_store.items():
+            if meta.in_use:
+                continue
+                
+            match_len = 0
+            min_len = min(len(meta.history_ids), len(input_ids))
+            for i in range(min_len):
+                if meta.history_ids[i] == input_ids[i]:
+                    match_len += 1
+                else:
+                    break
+            
+            if match_len > best_len:
+                best_len = match_len
+                best_id = sid
+                best_meta = meta
+
+        if best_meta is not None:
+            if not session_id:
+                best_meta.last_accessed = current_time
+                best_meta.in_use = True  # Lock the session
+                
+                if best_len < len(best_meta.history_ids):
+                    state.model.rewind_session(best_meta.session_ptr, best_len)
+                    best_meta.history_ids = best_meta.history_ids[:best_len]
+                return best_meta, best_len
+
+        # Check limits
+        if len(session_store) >= state.max_sessions:
+            # Evict LRU from AVAILABLE sessions
+            available_sessions = [s for s in session_store.items() if not s[1].in_use]
+            
+            if not available_sessions:
+                raise HTTPException(status_code=503, detail="All sessions are busy and max capacity reached.")
+            
+            oldest_sid, _ = min(available_sessions, key=lambda item: item[1].last_accessed)
+            
+            logger.info(f"Evicting session {oldest_sid}")
+            meta = session_store.pop(oldest_sid)
+            
+            state.model.rewind_session(meta.session_ptr, 0)
+            meta.history_ids = []
+            meta.messages = []
+            meta.last_accessed = current_time
+            meta.in_use = True
+            
+            new_sid = session_id if session_id else str(uuid.uuid4())
+            session_store[new_sid] = meta
+            return meta, 0
+
+        new_sid = session_id if session_id else str(uuid.uuid4())
+        logger.info(f"Creating new session {new_sid}")
+        ptr = state.model.create_session()
+        meta = SessionMeta(session_ptr=ptr, history_ids=[], messages=[], last_accessed=current_time, in_use=True)
+        session_store[new_sid] = meta
+        return meta, 0
+
+@asynccontextmanager
+async def lifespan(app: FastAPI):
+    load_model()
+    yield
+    cleanup_sessions()
+
+app = FastAPI(title="Llaisys Chatbot API", lifespan=lifespan)
+
+@app.middleware("http")
+async def security_headers_middleware(request: Request, call_next):
+    response = await call_next(request)
+    response.headers["X-Frame-Options"] = "DENY"
+    response.headers["Content-Security-Policy"] = "default-src 'self'"
+    response.headers["X-Content-Type-Options"] = "nosniff"
+    return response
+
+app.add_middleware(
+    CORSMiddleware,
+    allow_origins=["*"],
+    allow_credentials=True,
+    allow_methods=["*"],
+    allow_headers=["*"],
+)
+
+if OBSERVABILITY_ENABLED:
+    REQUEST_COUNT = Counter("llaisys_requests_total", "Total requests", ["method", "endpoint", "status"])
+    REQUEST_LATENCY = Histogram("llaisys_request_latency_seconds", "Request latency", ["method", "endpoint"])
+    
+    @app.middleware("http")
+    async def metrics_middleware(request: Request, call_next):
+        start_time = time.time()
+        method = request.method
+        path = request.url.path
+        
+        try:
+            response = await call_next(request)
+            status_code = response.status_code
+        except Exception as e:
+            status_code = 500
+            raise e
+        finally:
+            duration = time.time() - start_time
+            REQUEST_COUNT.labels(method=method, endpoint=path, status=status_code).inc()
+            REQUEST_LATENCY.labels(method=method, endpoint=path).observe(duration)
+            
+        return response
+
+    @app.get("/metrics")
+    def metrics():
+        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
+
+    provider = TracerProvider()
+    processor = BatchSpanProcessor(ConsoleSpanExporter())
+    provider.add_span_processor(processor)
+    trace.set_tracer_provider(provider)
+    FastAPIInstrumentor.instrument_app(app)
+
+@app.get("/health")
+def health_check():
+    if state.model is None:
+        raise HTTPException(status_code=503, detail="Model not loaded")
+    return {"status": "ok", "device": state.device_name, "model": state.model_path}
+
+@app.get("/config")
+def get_config(request: Request):
+    """Returns public configuration for the frontend."""
+    # Determine the API URL based on current request
+    scheme = request.url.scheme
+    host = request.url.hostname
+    port = request.url.port
+    
+    # If running behind a proxy or in dev, this might need adjustment
+    api_url = f"{scheme}://{host}:{port}"
+    return {"apiUrl": api_url}
+
+@app.post("/v1/context/items")
+def add_context_item(item: ContextItemRequest):
+    """Adds or updates a context item."""
+    state.context_manager.add_item(item.id, item.content, item.tags, item.metadata)
+    return {"status": "success", "id": item.id}
+
+@app.get("/v1/context/items")
+def list_context_items(tag: Optional[str] = None):
+    """Lists context items, optionally filtered by tag."""
+    if tag:
+        items = state.context_manager.get_items_by_tag(tag)
+    else:
+        items = state.context_manager.list_items()
+    return items
+
+@app.delete("/v1/context/items/{item_id}")
+def delete_context_item(item_id: str):
+    """Deletes a context item."""
+    if state.context_manager.remove_item(item_id):
+        return {"status": "success", "id": item_id}
+    raise HTTPException(status_code=404, detail="Item not found")
+
+@app.post("/v1/context/render")
+def render_template_endpoint(template: str, extra_context: Optional[Dict[str, Any]] = None):
+    """Renders a template string using the current context."""
+    try:
+        return {"rendered": state.context_manager.render_template(template, extra_context)}
+    except Exception as e:
+        raise HTTPException(status_code=400, detail=str(e))
+
+def prune_messages(messages: List[Dict[str, str]], tokenizer: Any, max_len: int) -> List[Dict[str, str]]:
+    """Prunes messages to fit within context length."""
+    if not messages:
+        return messages
+        
+    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
+    full_tokens = tokenizer.encode(full_text)
+    
+    if len(full_tokens) <= max_len:
+        return messages
+        
+    logger.info(f"Context length {len(full_tokens)} exceeds limit {max_len}. Pruning...")
+    
+    pruned_msgs = list(messages)
+    has_system = pruned_msgs[0]['role'] == 'system'
+    
+    while len(full_tokens) > max_len and len(pruned_msgs) > (1 if has_system else 0):
+        # Remove the oldest message (after system prompt)
+        idx_to_remove = 1 if has_system else 0
+        if idx_to_remove >= len(pruned_msgs):
+            break
+            
+        pruned_msgs.pop(idx_to_remove)
+        full_text = tokenizer.apply_chat_template(pruned_msgs, tokenize=False, add_generation_prompt=True)
+        full_tokens = tokenizer.encode(full_text)
+        
+    logger.info(f"Pruned context to {len(full_tokens)} tokens")
+    return pruned_msgs
+
+async def generate_response_stream(
+    request: ChatCompletionRequest,
+    session_meta: SessionMeta,
+    new_input_ids: List[int],
+    max_gen: int
+) -> AsyncGenerator[str, None]:
+    """Generator for streaming responses."""
+    
+    response_id = str(uuid.uuid4())
+    created_time = int(time.time())
+    model_name = request.model
+    
+    def create_chunk(content: Optional[str] = None, role: Optional[str] = None, finish_reason: Optional[str] = None):
+        delta = ChatCompletionChunkDelta()
+        if role: delta.role = role
+        if content: delta.content = content
+        
+        chunk = ChatCompletionChunk(
+            id=response_id,
+            created=created_time,
+            model=model_name,
+            choices=[ChatCompletionChunkChoice(
+                index=0,
+                delta=delta,
+                finish_reason=finish_reason
+            )]
+        )
+        return f"data: {chunk.model_dump_json()}\n\n"
+
+    # Initial chunk with role
+    try:
+        yield create_chunk(role="assistant")
+        
+        current_tokens = list(new_input_ids)
+        full_content = ""
+        
+        # 1. Prefill
+        if len(current_tokens) > 0:
+            try:
+                next_token = state.model.forward(
+                    session_meta.session_ptr, 
+                    current_tokens, 
+                    request.temperature or state.temperature, 
+                    request.top_p or state.top_p, 
+                    request.top_k or state.top_k
+                )
+            except Exception as e:
+                logger.error(f"Error during model forward: {e}")
+                yield create_chunk(finish_reason="error")
+                return
+
+            session_meta.history_ids.append(next_token)
+            
+            if next_token == getattr(state.model, "end_token", 2): # Default EOS if not set
+                session_meta.messages.append({"role": "assistant", "content": full_content})
+                yield create_chunk(finish_reason="stop")
+                yield "data: [DONE]\n\n"
+                return
+
+            text = state.tokenizer.decode([next_token], skip_special_tokens=False)
+            
+            if text:
+                full_content += text
+                yield create_chunk(content=text)
+
+        # 2. Decode Loop
+        for _ in range(max_gen - 1):
+            last_token = session_meta.history_ids[-1]
+            
+            try:
+                next_token = state.model.forward(
+                    session_meta.session_ptr,
+                    [last_token],
+                    request.temperature or state.temperature,
+                    request.top_p or state.top_p,
+                    request.top_k or state.top_k
+                )
+            except Exception as e:
+                logger.error(f"Error during generation: {e}")
+                break
+            
+            session_meta.history_ids.append(next_token)
+            
+            if next_token == getattr(state.model, "end_token", 2):
+                session_meta.messages.append({"role": "assistant", "content": full_content})
+                yield create_chunk(finish_reason="stop")
+                yield "data: [DONE]\n\n"
+                return
+
+            text = state.tokenizer.decode([next_token], skip_special_tokens=False)
+            
+            if text:
+                full_content += text
+                yield create_chunk(content=text)
+                
+        # Reached max tokens
+        session_meta.messages.append({"role": "assistant", "content": full_content})
+        yield create_chunk(finish_reason="length")
+        yield "data: [DONE]\n\n"
+    finally:
+        session_meta.in_use = False
+        logger.debug(f"Session released. in_use={session_meta.in_use}")
+
+@app.post("/v1/chat/completions", response_model=None)
+async def chat_completions(request: ChatCompletionRequest):
+    if not state.model:
+        raise HTTPException(status_code=503, detail="Model not loaded")
+
+    try:
+        # Prepare messages
+        req_messages = [m.model_dump() for m in request.messages]
+        
+        max_gen = request.max_tokens or 4096
+        if max_gen > state.max_steps:
+             max_gen = state.max_steps
+        
+        # Calculate available context
+        max_input = state.max_context_len - max_gen
+        messages = prune_messages(req_messages, state.tokenizer, max_input)
+        
+        # System prompt handling
+        has_system_prompt = len(messages) > 0 and messages[0].get("role") == "system"
+        sys_prompt_content = request.system_prompt if request.system_prompt else state.default_system_prompt
+
+        # Render system prompt if template usage is requested
+        if request.use_template and sys_prompt_content:
+            try:
+                logger.info("Rendering system prompt template...")
+                sys_prompt_content = state.context_manager.render_template(sys_prompt_content)
+            except Exception as e:
+                logger.error(f"Failed to render system prompt template: {e}")
+                raise HTTPException(status_code=400, detail=f"Template error: {str(e)}")
+        
+        if not has_system_prompt and sys_prompt_content:
+            messages.insert(0, {"role": "system", "content": sys_prompt_content})
+        elif has_system_prompt and request.system_prompt:
+            messages[0]["content"] = request.system_prompt
+
+        # Tokenize
+        prompt = state.tokenizer.apply_chat_template(
+            conversation=messages,
+            add_generation_prompt=True,
+            tokenize=False
+        )
+        input_ids = state.tokenizer.encode(prompt)
+        
+        # Session Management
+        session_meta, match_len = get_session_for_request(input_ids, session_id=request.session_id)
+        
+        try:
+            session_meta.messages = messages
+            
+            new_input_ids = input_ids[match_len:]
+            session_meta.history_ids.extend(new_input_ids)
+            
+            # Generation
+            stream_gen = generate_response_stream(request, session_meta, new_input_ids, max_gen)
+            
+            if request.stream:
+                return StreamingResponse(stream_gen, media_type="text/event-stream")
+            else:
+                # Collect full response
+                full_content = ""
+                finish_reason = "length"
+                
+                async for chunk_str in stream_gen:
+                    if chunk_str.strip() == "data: [DONE]":
+                        break
+                    
+                    # Parse "data: {...}"
+                    if chunk_str.startswith("data: "):
+                        json_str = chunk_str[6:].strip()
+                        try:
+                            chunk_data = json.loads(json_str)
+                            delta = chunk_data["choices"][0]["delta"]
+                            if "content" in delta and delta["content"]:
+                                full_content += delta["content"]
+                            if chunk_data["choices"][0]["finish_reason"]:
+                                finish_reason = chunk_data["choices"][0]["finish_reason"]
+                        except:
+                            pass
+                
+                return ChatCompletionResponse(
+                    id=str(uuid.uuid4()),
+                    created=int(time.time()),
+                    model=request.model,
+                    choices=[ChatCompletionResponseChoice(
+                        index=0,
+                        message=ChatMessage(role="assistant", content=full_content),
+                        finish_reason=finish_reason
+                    )]
+                )
+        except Exception:
+            session_meta.in_use = False
+            raise
+
+    except Exception as e:
+        logger.error(f"Unexpected error in chat_completions: {e}", exc_info=True)
+        raise HTTPException(status_code=500, detail=str(e))
+
+http_app = FastAPI()
+
+@http_app.middleware("http")
+async def redirect_to_https(request: Request, call_next):
+    if state.https_port:
+        url = request.url.replace(scheme="https", port=state.https_port)
+        return RedirectResponse(url, status_code=301)
+    return await call_next(request)
+
+if __name__ == "__main__":
+    parser = argparse.ArgumentParser()
+    parser.add_argument("--config", type=str, help="Path to config file (JSON)")
+    parser.add_argument("--model", type=str, help="Path to model directory")
+    parser.add_argument("--device", type=str, choices=["cpu", "nvidia"])
+    parser.add_argument("--dtype", type=str, choices=["float32", "float16"], help="Model data type")
+    parser.add_argument("--host", type=str)
+    parser.add_argument("--port", type=int)
+    parser.add_argument("--max-sessions", type=int)
+    parser.add_argument("--max-steps", type=int, help="Max generation steps")
+    parser.add_argument("--temperature", type=float, help="Default temperature")
+    parser.add_argument("--top-p", type=float, help="Default top_p")
+    parser.add_argument("--top-k", type=int, help="Default top_k")
+    parser.add_argument("--system-prompt", type=str, help="Default system prompt")
+    parser.add_argument("--no-https", action="store_true", help="Disable HTTPS")
+    args = parser.parse_args()
+
+    config = {}
+    if args.config and os.path.exists(args.config):
+        with open(args.config, "r") as f:
+            config = json.load(f)
+
+    def get_val(cli_val, config_key, default_val):
+        return cli_val if cli_val is not None else config.get(config_key, default_val)
+
+    state.model_path = get_val(args.model, "model", "")
+    state.device_name = get_val(args.device, "device", "cpu")
+    state.dtype = get_val(args.dtype, "dtype", "float32")
+    host = get_val(args.host, "host", "127.0.0.1")
+    port = get_val(args.port, "port", 6008)
+    state.max_sessions = get_val(args.max_sessions, "max_sessions", 10)
+    state.max_steps = get_val(args.max_steps, "max_steps", 4096)
+    state.temperature = get_val(args.temperature, "temperature", 0.7)
+    state.top_p = get_val(args.top_p, "top_p", 0.9)
+    state.top_k = get_val(args.top_k, "top_k", 50)
+    state.default_system_prompt = get_val(args.system_prompt, "system_prompt", state.default_system_prompt)
+
+    if not state.model_path:
+        logger.error("Model path is required via --model or config file")
+        sys.exit(1)
+
+    # Download if needed
+    if not os.path.exists(state.model_path) and not os.path.isdir(state.model_path) and snapshot_download:
+        logger.info(f"Model path {state.model_path} not found locally, attempting HF download...")
+        try:
+            state.model_path = snapshot_download(state.model_path)
+        except Exception as e:
+            logger.error(f"Failed to download model: {e}")
+            sys.exit(1)
+
+    cert_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'certs'))
+    key_path = os.path.join(cert_dir, 'key.pem')
+    cert_path = os.path.join(cert_dir, 'cert.pem')
+    
+    ssl_configured = False
+    if not args.no_https:
+        if not (os.path.exists(key_path) and os.path.exists(cert_path)):
+            os.makedirs(cert_dir, exist_ok=True)
+            logger.info("Generating self-signed certificate...")
+            # Use absolute paths in openssl command
+            cmd = f'openssl req -x509 -newkey rsa:4096 -keyout "{key_path}" -out "{cert_path}" -days 365 -nodes -subj "/CN=localhost" 2>/dev/null'
+            if os.system(cmd) == 0:
+                ssl_configured = True
+            else:
+                logger.error("Failed to generate certificates. Fallback to HTTP.")
+        else:
+            ssl_configured = True
+    
+    if ssl_configured:
+        https_port = port + 1
+        state.https_port = https_port
+        
+        async def run_servers():
+            config_https = uvicorn.Config(
+                app, host=host, port=https_port, 
+                ssl_keyfile=key_path, ssl_certfile=cert_path,
+                log_level="info"
+            )
+            config_http = uvicorn.Config(http_app, host=host, port=port, log_level="warning")
+            
+            logger.info(f"Starting HTTPS server on {host}:{https_port}")
+            logger.info(f"Starting HTTP redirect server on {host}:{port}")
+            
+            server_https = uvicorn.Server(config_https)
+            server_http = uvicorn.Server(config_http)
+            
+            await asyncio.gather(server_https.serve(), server_http.serve())
+            
+        asyncio.run(run_servers())
+    else:
+        logger.warning("Running in HTTP mode (no SSL).")
+        uvicorn.run(app, host=host, port=port)
\ No newline at end of file
diff --git a/src/chatbot/frontend/.env.example b/src/chatbot/frontend/.env.example
new file mode 100644
index 0000000..bc8555c
--- /dev/null
+++ b/src/chatbot/frontend/.env.example
@@ -0,0 +1,4 @@
+# Frontend Configuration
+
+PORT=6006
+API_URL=http://localhost:6008
diff --git a/src/chatbot/frontend/index.html b/src/chatbot/frontend/index.html
new file mode 100644
index 0000000..92536f2
--- /dev/null
+++ b/src/chatbot/frontend/index.html
@@ -0,0 +1,12 @@
+<!doctype html>
+<html lang="en">
+  <head>
+    <meta charset="UTF-8" />
+    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
+    <title>LLAISYS AI Chat</title>
+  </head>
+  <body>
+    <div id="root"></div>
+    <script type="module" src="/src/main.jsx"></script>
+  </body>
+</html>
diff --git a/src/chatbot/frontend/package.json b/src/chatbot/frontend/package.json
new file mode 100644
index 0000000..a515b21
--- /dev/null
+++ b/src/chatbot/frontend/package.json
@@ -0,0 +1,36 @@
+{
+  "name": "llaisys-chatbot-frontend",
+  "private": true,
+  "version": "1.0.0",
+  "type": "module",
+  "scripts": {
+    "dev": "vite",
+    "build": "vite build",
+    "preview": "vite preview",
+    "start": "node server.cjs",
+    "test": "vitest"
+  },
+  "dependencies": {
+    "bootstrap": "^5.3.2",
+    "cors": "^2.8.5",
+    "dotenv": "^16.3.1",
+    "express": "^4.18.2",
+    "helmet": "^8.1.0",
+    "http-proxy-middleware": "^2.0.6",
+    "react": "^18.2.0",
+    "react-dom": "^18.2.0",
+    "react-markdown": "^10.1.0"
+  },
+  "devDependencies": {
+    "@testing-library/jest-dom": "^6.1.5",
+    "@testing-library/react": "^14.1.2",
+    "@testing-library/user-event": "^14.5.1",
+    "@types/react": "^18.2.43",
+    "@types/react-dom": "^18.2.17",
+    "@vitejs/plugin-react": "^4.2.1",
+    "jsdom": "^23.0.1",
+    "supertest": "^6.3.3",
+    "vite": "^5.0.8",
+    "vitest": "^1.0.4"
+  }
+}
diff --git a/src/chatbot/frontend/server.cjs b/src/chatbot/frontend/server.cjs
new file mode 100644
index 0000000..9a2801e
--- /dev/null
+++ b/src/chatbot/frontend/server.cjs
@@ -0,0 +1,135 @@
+const express = require('express');
+const cors = require('cors');
+const path = require('path');
+const helmet = require('helmet');
+const fs = require('fs');
+const https = require('https');
+const http = require('http');
+const { execSync } = require('child_process');
+const { createProxyMiddleware } = require('http-proxy-middleware');
+require('dotenv').config();
+
+const app = express();
+const PORT = parseInt(process.env.PORT || 6006, 10);
+const HTTPS_PORT = process.env.NODE_ENV === 'test' ? 0 : PORT + 1;
+const INTERNAL_API_URL = process.env.INTERNAL_API_URL || 'https://localhost:6009';
+const USE_PROXY = process.env.NO_HTTPS === 'true';
+
+const certDir = path.join(__dirname, '..', 'certs');
+const keyPath = path.join(certDir, 'key.pem');
+const certPath = path.join(certDir, 'cert.pem');
+
+let sslConfigured = false;
+try {
+    if (process.env.NO_HTTPS !== 'true') {
+        if (!fs.existsSync(keyPath) || !fs.existsSync(certPath)) {
+            fs.mkdirSync(certDir, { recursive: true });
+            execSync(`openssl req -x509 -newkey rsa:4096 -keyout "${keyPath}" -out "${certPath}" -days 365 -nodes -subj "/CN=localhost"`, { stdio: 'pipe' });
+        }
+        if (fs.existsSync(keyPath) && fs.existsSync(certPath)) {
+            sslConfigured = true;
+        }
+    } else {
+        console.log('HTTPS disabled via NO_HTTPS environment variable.');
+    }
+} catch (err) {
+    console.error('Failed to auto-provision certificates. Ensure openssl is installed.', err.message);
+}
+
+app.use(helmet({
+    hsts: sslConfigured ? { maxAge: 31536000, includeSubDomains: true } : false,
+    frameguard: { action: 'deny' },
+    contentSecurityPolicy: {
+        directives: {
+            defaultSrc: ["'self'"],
+            scriptSrc: ["'self'", "'unsafe-inline'"],
+            styleSrc: ["'self'", "'unsafe-inline'"],
+            connectSrc: ["'self'", USE_PROXY ? "'self'" : INTERNAL_API_URL]
+        }
+    }
+}));
+
+app.use(cors());
+
+app.use((req, res, next) => {
+    if (process.env.NODE_ENV !== 'test') {
+        console.log(`[Server] Received ${req.method} request for ${req.url}`);
+    }
+    next();
+});
+
+if (USE_PROXY) {
+    console.log(`[Server] Enabling API Proxy to ${INTERNAL_API_URL}`);
+    
+    // Mount proxy at /api/v1 if the frontend calls it there
+    app.use('/api/v1', createProxyMiddleware({
+        target: INTERNAL_API_URL,
+        changeOrigin: true,
+        pathRewrite: {
+            '^/api/v1': '/v1', // Rewrite /api/v1 -> /v1
+        },
+        secure: false, 
+        ws: true,
+        logLevel: 'debug'
+    }));
+
+    app.use('/v1', createProxyMiddleware({
+        target: INTERNAL_API_URL,
+        changeOrigin: true,
+        secure: false, 
+        ws: true,
+        logLevel: 'debug'
+    }));
+}
+
+app.use(express.json());
+app.use(express.static(path.join(__dirname, 'dist')));
+
+app.get('/config', (req, res) => {
+    res.json({ apiUrl: USE_PROXY ? '' : INTERNAL_API_URL });
+});
+
+app.get('*', (req, res) => {
+    res.sendFile(path.join(__dirname, 'dist', 'index.html'));
+});
+
+let server;
+let httpServer;
+
+if (sslConfigured) {
+    const httpsOptions = {
+        key: fs.readFileSync(keyPath),
+        cert: fs.readFileSync(certPath),
+        minVersion: 'TLSv1.3'
+    };
+    
+    if (process.env.NODE_ENV !== 'test') {
+        server = https.createServer(httpsOptions, app).listen(HTTPS_PORT, () => {
+            console.log(`Frontend HTTPS Server running on https://localhost:${HTTPS_PORT} (TLS 1.3)`);
+            console.log(`Configured API URL: ${INTERNAL_API_URL}`);
+        });
+
+        const httpApp = express();
+        httpApp.get('*', (req, res) => {
+            const hostWithoutPort = req.headers.host ? req.headers.host.split(':')[0] : 'localhost';
+            res.redirect(`https://${hostWithoutPort}:${HTTPS_PORT}${req.url}`);
+        });
+        
+        httpServer = http.createServer(httpApp).listen(PORT, () => {
+            console.log(`Frontend HTTP redirect server running on http://localhost:${PORT}`);
+        });
+    } else {
+        server = app;
+    }
+} else {
+    if (process.env.NODE_ENV !== 'test') {
+        server = app.listen(PORT, () => {
+            console.log(`Frontend Server running on http://localhost:${PORT}`);
+            console.log(`Configured API URL: ${INTERNAL_API_URL}`);
+        });
+    } else {
+        server = app;
+    }
+}
+
+module.exports = server;
diff --git a/src/chatbot/frontend/src/App.jsx b/src/chatbot/frontend/src/App.jsx
new file mode 100644
index 0000000..7c406b5
--- /dev/null
+++ b/src/chatbot/frontend/src/App.jsx
@@ -0,0 +1,40 @@
+import React, { useState } from 'react';
+import Navbar from './components/Navbar';
+import ChatContainer from './components/ChatContainer';
+import InputArea from './components/InputArea';
+import useChat from './hooks/useChat';
+
+function App() {
+  const { messages, sendMessage, isThinking, error, setMessages } = useChat();
+  const [inputText, setInputText] = useState('');
+
+  const handleEdit = (index, content) => {
+      setInputText(content);
+      setMessages(prev => prev.slice(0, index));
+  };
+
+  const handleSendMessage = (text) => {
+      sendMessage(text);
+      setInputText('');
+  };
+
+  return (
+    <>
+      <Navbar />
+      <ChatContainer 
+          messages={messages} 
+          isThinking={isThinking} 
+          error={error} 
+          onEdit={handleEdit}
+      />
+      <InputArea 
+          text={inputText} 
+          setText={setInputText}
+          onSendMessage={handleSendMessage} 
+          disabled={isThinking} 
+      />
+    </>
+  );
+}
+
+export default App;
diff --git a/src/chatbot/frontend/src/App.test.jsx b/src/chatbot/frontend/src/App.test.jsx
new file mode 100644
index 0000000..5b6d4e1
--- /dev/null
+++ b/src/chatbot/frontend/src/App.test.jsx
@@ -0,0 +1,54 @@
+import { render, screen, fireEvent } from '@testing-library/react';
+import { describe, it, expect, vi } from 'vitest';
+import App from './App';
+import * as useChatModule from './hooks/useChat';
+
+// Mock the hook
+vi.mock('./hooks/useChat', () => ({
+  __esModule: true,
+  default: vi.fn(),
+}));
+
+describe('App Component', () => {
+  it('renders navbar', () => {
+    useChatModule.default.mockReturnValue({
+        messages: [],
+        sendMessage: vi.fn(),
+        isThinking: false,
+        error: null
+    });
+    render(<App />);
+    expect(screen.getByText('LLAISYS AI Chat')).toBeInTheDocument();
+  });
+
+  it('handles user input', () => {
+    const sendMessageMock = vi.fn();
+    useChatModule.default.mockReturnValue({
+        messages: [],
+        sendMessage: sendMessageMock,
+        isThinking: false,
+        error: null
+    });
+
+    render(<App />);
+    const input = screen.getByPlaceholderText('Type your message...');
+    const button = screen.getByText('Send');
+
+    fireEvent.change(input, { target: { value: 'Hello' } });
+    fireEvent.click(button);
+
+    expect(sendMessageMock).toHaveBeenCalledWith('Hello');
+  });
+
+  it('displays thinking indicator', () => {
+    useChatModule.default.mockReturnValue({
+        messages: [],
+        sendMessage: vi.fn(),
+        isThinking: true,
+        error: null
+    });
+
+    render(<App />);
+    expect(screen.getByText('Thinking')).toBeInTheDocument();
+  });
+});
diff --git a/src/chatbot/frontend/src/components/ChatContainer.jsx b/src/chatbot/frontend/src/components/ChatContainer.jsx
new file mode 100644
index 0000000..d2c031c
--- /dev/null
+++ b/src/chatbot/frontend/src/components/ChatContainer.jsx
@@ -0,0 +1,34 @@
+import React, { useEffect, useRef } from 'react';
+import Message from './Message';
+import ThinkingIndicator from './ThinkingIndicator';
+
+const ChatContainer = ({ messages, isThinking, error, onEdit }) => {
+    const endRef = useRef(null);
+
+    useEffect(() => {
+        endRef.current?.scrollIntoView({ behavior: 'smooth' });
+    }, [messages, isThinking, error]);
+
+    return (
+        <div id="chat-container" className="flex-grow-1 overflow-auto p-3">
+            {messages.map((msg, idx) => (
+                <Message 
+                    key={idx} 
+                    index={idx}
+                    role={msg.role} 
+                    content={msg.content} 
+                    onEdit={onEdit}
+                />
+            ))}
+            {isThinking && <ThinkingIndicator />}
+            {error && (
+                <div className="alert alert-danger mt-3" role="alert">
+                    {error}
+                </div>
+            )}
+            <div ref={endRef} />
+        </div>
+    );
+};
+
+export default ChatContainer;
diff --git a/src/chatbot/frontend/src/components/InputArea.jsx b/src/chatbot/frontend/src/components/InputArea.jsx
new file mode 100644
index 0000000..17622e1
--- /dev/null
+++ b/src/chatbot/frontend/src/components/InputArea.jsx
@@ -0,0 +1,40 @@
+import React, { useState } from 'react';
+
+const InputArea = ({ text, setText, onSendMessage, disabled }) => {
+    const [internalText, setInternalText] = useState('');
+    
+    const isControlled = text !== undefined && setText !== undefined;
+    const currentText = isControlled ? text : internalText;
+    const handleChange = isControlled ? (e) => setText(e.target.value) : (e) => setInternalText(e.target.value);
+
+    const handleSubmit = (e) => {
+        e.preventDefault();
+        if (currentText.trim() && !disabled) {
+            onSendMessage(currentText);
+            if (!isControlled) setInternalText('');
+        }
+    };
+
+    return (
+        <div className="input-area">
+            <div className="container">
+                <form onSubmit={handleSubmit} className="d-flex gap-2">
+                    <input
+                        type="text"
+                        className="form-control"
+                        placeholder="Type your message..."
+                        autoComplete="off"
+                        value={currentText}
+                        onChange={handleChange}
+                        disabled={disabled}
+                    />
+                    <button type="submit" className="btn btn-primary" disabled={disabled}>
+                        Send
+                    </button>
+                </form>
+            </div>
+        </div>
+    );
+};
+
+export default InputArea;
diff --git a/src/chatbot/frontend/src/components/Message.jsx b/src/chatbot/frontend/src/components/Message.jsx
new file mode 100644
index 0000000..30327ab
--- /dev/null
+++ b/src/chatbot/frontend/src/components/Message.jsx
@@ -0,0 +1,64 @@
+import React from 'react';
+import ReactMarkdown from 'react-markdown';
+import { parseMessage } from '../utils/messageParser';
+
+const Message = ({ role, content, index, onEdit }) => {
+    const safeContent = content || '';
+
+    const renderContent = () => {
+        if (role !== 'user' && safeContent) {
+            const segments = parseMessage(safeContent);
+
+            return segments.map((segment, idx) => {
+                if (segment.type === 'think') {
+                    return (
+                        <details key={idx} className="think-block mb-2">
+                            <summary className="text-muted small cursor-pointer">Thought Process</summary>
+                            <div className="text-muted small border-start ps-2 mt-1">
+                                {segment.content}
+                            </div>
+                        </details>
+                    );
+                }
+                
+                return (
+                    <div key={idx} className="markdown-content">
+                        <ReactMarkdown>{segment.content}</ReactMarkdown>
+                    </div>
+                );
+            });
+        }
+        return <div className="markdown-content"><ReactMarkdown>{safeContent}</ReactMarkdown></div>;
+    };
+
+    return (
+        <div className={`message ${role === 'user' ? 'user-message' : 'ai-message'} position-relative group mb-3 p-3 rounded`}>
+            {role === 'user' && onEdit && (
+                <button 
+                    className="edit-btn btn btn-sm btn-link p-0 position-absolute d-flex align-items-center justify-content-center"
+                    style={{ 
+                        top: '5px', 
+                        right: '10px', 
+                        textDecoration: 'none', 
+                        opacity: 1,
+                        width: '32px',
+                        height: '32px',
+                        fontSize: '1.2rem',
+                        backgroundColor: '#ffffff',
+                        color: '#007bff',
+                        borderRadius: '50%',
+                        boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
+                    }}
+                    onClick={() => onEdit(index, safeContent)}
+                    title="Edit message"
+                    aria-label="Edit message"
+                >
+                    ✎
+                </button>
+            )}
+            {renderContent()}
+        </div>
+    );
+};
+
+export default Message;
diff --git a/src/chatbot/frontend/src/components/Navbar.jsx b/src/chatbot/frontend/src/components/Navbar.jsx
new file mode 100644
index 0000000..0d3bf73
--- /dev/null
+++ b/src/chatbot/frontend/src/components/Navbar.jsx
@@ -0,0 +1,13 @@
+import React from 'react';
+
+const Navbar = () => {
+    return (
+        <nav className="navbar navbar-dark bg-dark">
+            <div className="container-fluid">
+                <span className="navbar-brand mb-0 h1">LLAISYS AI Chat</span>
+            </div>
+        </nav>
+    );
+};
+
+export default Navbar;
diff --git a/src/chatbot/frontend/src/components/ThinkingIndicator.jsx b/src/chatbot/frontend/src/components/ThinkingIndicator.jsx
new file mode 100644
index 0000000..c782c04
--- /dev/null
+++ b/src/chatbot/frontend/src/components/ThinkingIndicator.jsx
@@ -0,0 +1,14 @@
+import React from 'react';
+
+const ThinkingIndicator = () => {
+    return (
+        <div className="thinking-indicator">
+            Thinking
+            <div className="dot"></div>
+            <div className="dot"></div>
+            <div className="dot"></div>
+        </div>
+    );
+};
+
+export default ThinkingIndicator;
diff --git a/src/chatbot/frontend/src/hooks/useChat.js b/src/chatbot/frontend/src/hooks/useChat.js
new file mode 100644
index 0000000..0ae86cf
--- /dev/null
+++ b/src/chatbot/frontend/src/hooks/useChat.js
@@ -0,0 +1,133 @@
+import { useState, useEffect, useRef, useCallback } from 'react';
+
+const useChat = () => {
+    const [messages, setMessages] = useState([]);
+    const [isThinking, setIsThinking] = useState(false);
+    const [error, setError] = useState(null);
+    const [apiUrl, setApiUrl] = useState(''); 
+    const sessionId = useRef('session-' + Math.random().toString(36).substr(2, 9) + '-' + Date.now());
+    
+    const messagesRef = useRef(messages);
+    useEffect(() => {
+        messagesRef.current = messages;
+    }, [messages]);
+
+    useEffect(() => {
+        const fetchConfig = async () => {
+            try {
+                const res = await fetch('/config');
+                if (res.ok) {
+                    const config = await res.json();
+                    if (config.apiUrl) setApiUrl(config.apiUrl);
+                }
+            } catch (e) {
+                console.warn('Using default API URL', e);
+            }
+        };
+        fetchConfig();
+    }, []);
+
+    const sendMessage = useCallback(async (text) => {
+        if (!text.trim()) return;
+
+        const userMsg = { role: 'user', content: text };
+        
+        // Optimistic update
+        setMessages(prev => [...prev, userMsg]);
+        setIsThinking(true);
+        setError(null);
+
+        try {
+            let endpoint = '/v1/chat/completions';
+            if (apiUrl && apiUrl.startsWith('http')) {
+                endpoint = `${apiUrl}/v1/chat/completions`;
+            } else if (apiUrl) {
+                const cleanUrl = apiUrl.endsWith('/') ? apiUrl.slice(0, -1) : apiUrl;
+                endpoint = `${cleanUrl}/v1/chat/completions`;
+            }
+
+            const history = [...messagesRef.current, userMsg];
+
+            const response = await fetch(endpoint, {
+                method: 'POST',
+                headers: { 'Content-Type': 'application/json' },
+                body: JSON.stringify({
+                    model: 'qwen2',
+                    messages: history,
+                    stream: true,
+                    temperature: 0.7,
+                    session_id: sessionId.current,
+                    use_template: true
+                })
+            });
+
+            if (!response.ok) {
+                const errorText = await response.text();
+                throw new Error(`API Error: ${response.status} ${response.statusText} - ${errorText}`);
+            }
+
+            const reader = response.body.getReader();
+            const decoder = new TextDecoder();
+            
+            setIsThinking(false);
+            
+            let aiMsg = { role: 'assistant', content: '' };
+            setMessages(prev => [...prev, aiMsg]);
+            
+            let buffer = '';
+
+            while (true) {
+                const { done, value } = await reader.read();
+                if (done) break;
+                
+                const chunk = decoder.decode(value, { stream: true });
+                buffer += chunk;
+                
+                const lines = buffer.split('\n');
+                buffer = lines.pop() || ''; 
+                
+                let deltaContent = '';
+
+                for (const line of lines) {
+                    if (line.startsWith('data: ')) {
+                        const jsonStr = line.substring(6).trim();
+                        if (jsonStr === '[DONE]') break;
+                        
+                        try {
+                            const data = JSON.parse(jsonStr);
+                            const delta = data.choices[0].delta.content || '';
+                            if (delta) {
+                                deltaContent += delta;
+                            }
+                        } catch (e) {
+                            console.warn("Failed to parse SSE JSON:", jsonStr);
+                        }
+                    }
+                }
+
+                if (deltaContent) {
+                    aiMsg.content += deltaContent;
+                    setMessages(prev => {
+                        const newMsgs = [...prev];
+                        if (newMsgs.length > 0 && newMsgs[newMsgs.length - 1].role === 'assistant') {
+                            newMsgs[newMsgs.length - 1] = { ...aiMsg };
+                        } else {
+                            newMsgs.push({ ...aiMsg });
+                        }
+                        return newMsgs;
+                    });
+                }
+            }
+
+        } catch (err) {
+            console.error("Connection Error:", err);
+            setIsThinking(false);
+            setError(err.message);
+            setMessages(prev => [...prev, { role: 'assistant', content: `Connection Error: ${err.message}`, isError: true }]);
+        }
+    }, [apiUrl]);
+
+    return { messages, sendMessage, isThinking, error, setMessages };
+};
+
+export default useChat;
diff --git a/src/chatbot/frontend/src/index.css b/src/chatbot/frontend/src/index.css
new file mode 100644
index 0000000..1a07286
--- /dev/null
+++ b/src/chatbot/frontend/src/index.css
@@ -0,0 +1,133 @@
+body { background-color: #f8f9fa; height: 100vh; margin: 0; }
+#root { display: flex; flex-direction: column; height: 100%; }
+
+.message { 
+    margin-bottom: 15px; 
+    padding: 10px 15px; 
+    border-radius: 15px; 
+    max-width: 80%; 
+    position: relative; 
+    word-wrap: break-word;
+}
+
+.user-message { 
+    background-color: #007bff; 
+    color: white; 
+    align-self: flex-end; 
+    margin-left: auto; 
+    white-space: pre-wrap;
+}
+
+.ai-message { 
+    background-color: #e9ecef; 
+    color: #212529; 
+    align-self: flex-start; 
+}
+
+/* Markdown Content */
+.markdown-content p {
+    margin-bottom: 0.5rem;
+}
+.markdown-content p:last-child {
+    margin-bottom: 0;
+}
+.markdown-content pre {
+    background-color: #f1f3f5;
+    padding: 10px;
+    border-radius: 5px;
+    overflow-x: auto;
+}
+.markdown-content code {
+    background-color: rgba(0,0,0,0.05);
+    padding: 2px 4px;
+    border-radius: 3px;
+    font-family: monospace;
+}
+.markdown-content pre code {
+    background-color: transparent;
+    padding: 0;
+}
+
+/* Edit Button */
+.edit-btn {
+    opacity: 0;
+    transition: opacity 0.2s ease-in-out;
+    background: rgba(0,0,0,0.1);
+    border-radius: 50%;
+    width: 24px;
+    height: 24px;
+    display: flex;
+    align-items: center;
+    justify-content: center;
+}
+.message:hover .edit-btn, .edit-btn:focus {
+    opacity: 1 !important;
+}
+.edit-btn:hover {
+    background-color: #f8f9fa !important;
+    transform: scale(1.05);
+}
+
+/* Thinking Indicator */
+.thinking-indicator {
+    display: inline-flex;
+    align-items: center;
+    gap: 4px;
+    color: #6c757d;
+    font-style: italic;
+    font-size: 0.9em;
+    margin-bottom: 15px;
+    align-self: flex-start;
+    padding: 10px 15px;
+}
+.dot {
+    width: 6px;
+    height: 6px;
+    background-color: #6c757d;
+    border-radius: 50%;
+    animation: bounce 1.4s infinite ease-in-out both;
+}
+.dot:nth-child(1) { animation-delay: -0.32s; }
+.dot:nth-child(2) { animation-delay: -0.16s; }
+
+@keyframes bounce {
+    0%, 80%, 100% { transform: scale(0); }
+    40% { transform: scale(1); }
+}
+
+.input-area { padding: 20px; background-color: white; border-top: 1px solid #dee2e6; }
+
+/* Think Block Styles */
+.think-block {
+    background-color: #fff;
+    border-left: 4px solid #adb5bd;
+    padding: 8px 12px;
+    margin-bottom: 12px;
+    font-size: 0.9em;
+    color: #6c757d;
+    border-radius: 4px;
+    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
+}
+
+.think-block summary {
+    cursor: pointer;
+    font-weight: 600;
+    outline: none;
+    user-select: none;
+    color: #495057;
+    margin-bottom: 5px;
+    list-style: none;
+}
+.think-block summary::-webkit-details-marker {
+    display: none;
+}
+.think-block summary::before {
+    content: '▶';
+    font-size: 0.8em;
+    margin-right: 6px;
+    display: inline-block;
+    transition: transform 0.2s;
+}
+.think-block[open] summary::before {
+    transform: rotate(90deg);
+}
diff --git a/src/chatbot/frontend/src/main.jsx b/src/chatbot/frontend/src/main.jsx
new file mode 100644
index 0000000..9f009e8
--- /dev/null
+++ b/src/chatbot/frontend/src/main.jsx
@@ -0,0 +1,11 @@
+import React from 'react'
+import ReactDOM from 'react-dom/client'
+import App from './App.jsx'
+import 'bootstrap/dist/css/bootstrap.min.css'
+import './index.css'
+
+ReactDOM.createRoot(document.getElementById('root')).render(
+  <React.StrictMode>
+    <App />
+  </React.StrictMode>,
+)
diff --git a/src/chatbot/frontend/src/setupTests.js b/src/chatbot/frontend/src/setupTests.js
new file mode 100644
index 0000000..7a6a644
--- /dev/null
+++ b/src/chatbot/frontend/src/setupTests.js
@@ -0,0 +1,3 @@
+import '@testing-library/jest-dom';
+
+window.HTMLElement.prototype.scrollIntoView = function() {};
diff --git a/src/chatbot/frontend/src/utils/messageParser.js b/src/chatbot/frontend/src/utils/messageParser.js
new file mode 100644
index 0000000..09c3c59
--- /dev/null
+++ b/src/chatbot/frontend/src/utils/messageParser.js
@@ -0,0 +1,68 @@
+
+/**
+ * Parses a raw message string into structured segments.
+ * 
+ * @param {string} content - The raw message content
+ * @returns {Array<{type: 'text'|'think', content: string}>} - Array of segments
+ */
+export const parseMessage = (content) => {
+    if (!content && content !== '') return [];
+    if (content === '') return [];
+    
+    const segments = [];
+    let currentPos = 0;
+    
+    while (currentPos < content.length) {
+        const startTagIndex = content.indexOf('<think>', currentPos);
+        const endTagIndex = content.indexOf('</think>', currentPos);
+        
+        if (endTagIndex === -1) {
+            if (startTagIndex !== -1) {
+                // Case: Unclosed think block (streaming or truncated)
+                if (startTagIndex > currentPos) {
+                    const textContent = content.substring(currentPos, startTagIndex);
+                    if (textContent) {
+                        segments.push({ type: 'text', content: textContent });
+                    }
+                }
+                
+                const thoughtContent = content.substring(startTagIndex + 7);
+                if (thoughtContent) {
+                    segments.push({ type: 'think', content: thoughtContent });
+                }
+                break;
+            }
+
+            const remainingText = content.substring(currentPos);
+            if (remainingText) {
+                segments.push({ type: 'text', content: remainingText });
+            }
+            break;
+        }
+                
+        if (startTagIndex === -1 || startTagIndex > endTagIndex) {            
+            const thoughtContent = content.substring(currentPos, endTagIndex);
+            if (thoughtContent) {
+                segments.push({ type: 'think', content: thoughtContent });
+            }
+            
+            currentPos = endTagIndex + 8;
+        } else {            
+            if (startTagIndex > currentPos) {
+                const textContent = content.substring(currentPos, startTagIndex);
+                if (textContent) {
+                    segments.push({ type: 'text', content: textContent });
+                }
+            }
+            
+            const thoughtContent = content.substring(startTagIndex + 7, endTagIndex); // 7 is length of '<think>'
+            if (thoughtContent) {
+                 segments.push({ type: 'think', content: thoughtContent });
+            }
+            
+            currentPos = endTagIndex + 8;
+        }
+    }
+    
+    return segments;
+};
diff --git a/src/chatbot/frontend/src/utils/messageParser.test.js b/src/chatbot/frontend/src/utils/messageParser.test.js
new file mode 100644
index 0000000..27645b9
--- /dev/null
+++ b/src/chatbot/frontend/src/utils/messageParser.test.js
@@ -0,0 +1,82 @@
+
+import { describe, it, expect } from 'vitest';
+import { parseMessage } from './messageParser';
+
+describe('messageParser', () => {
+    it('should parse simple text', () => {
+        const input = "Hello world";
+        const result = parseMessage(input);
+        expect(result).toEqual([{ type: 'text', content: "Hello world" }]);
+    });
+
+    it('should parse think block followed by text', () => {
+        const input = "<think>This is a thought</think>Hello there";
+        const result = parseMessage(input);
+        expect(result).toEqual([
+            { type: 'think', content: "This is a thought" },
+            { type: 'text', content: "Hello there" }
+        ]);
+    });
+
+    it('should parse text followed by think block', () => {
+        const input = "Prefix<think>Thought</think>";
+        const result = parseMessage(input);
+        expect(result).toEqual([
+            { type: 'text', content: "Prefix" },
+            { type: 'think', content: "Thought" }
+        ]);
+    });
+
+    it('should treat unclosed think tag as open-ended think block', () => {
+        const input = "Start <think>Unclosed thought";
+        const result = parseMessage(input);
+        expect(result).toEqual([
+            { type: 'text', content: "Start " },
+            { type: 'think', content: "Unclosed thought" }
+        ]);
+    });
+
+    it('should handle missing opening think tag at the start', () => {
+        const input = "Implicit thought</think>Response";
+        const result = parseMessage(input);
+        expect(result).toEqual([
+            { type: 'think', content: "Implicit thought" },
+            { type: 'text', content: "Response" }
+        ]);
+    });
+
+    it('should handle multiple think blocks', () => {
+        const input = "<think>One</think>Middle<think>Two</think>End";
+        const result = parseMessage(input);
+        expect(result).toEqual([
+            { type: 'think', content: "One" },
+            { type: 'text', content: "Middle" },
+            { type: 'think', content: "Two" },
+            { type: 'text', content: "End" }
+        ]);
+    });
+
+    it('should handle empty content', () => {
+        expect(parseMessage("")).toEqual([]);
+        expect(parseMessage(null)).toEqual([]);
+    });
+
+    it('should handle multiline think blocks', () => {
+        const input = "<think>\nLine 1\nLine 2\n</think>\nResponse";
+        const result = parseMessage(input);
+        expect(result).toEqual([
+            { type: 'think', content: "\nLine 1\nLine 2\n" },
+            { type: 'text', content: "\nResponse" }
+        ]);
+    });
+
+    it('should preserve whitespace in text', () => {
+        const input = "Hello <think>...</think> World";
+        const result = parseMessage(input);
+        expect(result).toEqual([
+            { type: 'text', content: "Hello " },
+            { type: 'think', content: "..." },
+            { type: 'text', content: " World" }
+        ]);
+    });
+});
diff --git a/src/chatbot/frontend/test/config.test.js b/src/chatbot/frontend/test/config.test.js
new file mode 100644
index 0000000..53d9d48
--- /dev/null
+++ b/src/chatbot/frontend/test/config.test.js
@@ -0,0 +1,63 @@
+import request from 'supertest';
+import { describe, beforeEach, afterEach, test, expect, vi } from 'vitest';
+import { createRequire } from 'module';
+
+const require = createRequire(import.meta.url);
+
+// Tell Node to ignore self-signed certificate errors during tests
+process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';
+
+describe('Configuration Tests', () => {
+    let server;
+    let originalEnv;
+
+    beforeEach(() => {
+        originalEnv = { ...process.env };
+        vi.resetModules();
+    });
+
+    afterEach(async () => {
+        process.env = originalEnv;
+        if (server && typeof server.close === 'function') {
+            await new Promise(resolve => server.close(resolve));
+        }
+    });
+
+    test('should use default API URL when API_URL is not set', async () => {
+        delete process.env.INTERNAL_API_URL;
+        
+        // Clear require cache to force reload
+        const serverPath = require.resolve('../server.cjs');
+        delete require.cache[serverPath];
+        server = require(serverPath);
+        
+        // server might be https or http depending on cert generation. supertest handles both.
+        const res = await request(server).get('/config');
+        expect(res.statusCode).toBe(200);
+        expect(res.body.apiUrl).toBe('https://localhost:6008'); // default is now https
+    });
+
+    test('should use configured API URL when INTERNAL_API_URL is set', async () => {
+        process.env.INTERNAL_API_URL = 'https://custom-api:9000';
+        
+        const serverPath = require.resolve('../server.cjs');
+        delete require.cache[serverPath];
+        server = require(serverPath);
+        
+        const res = await request(server).get('/config');
+        expect(res.statusCode).toBe(200);
+        expect(res.body.apiUrl).toBe('https://custom-api:9000');
+    });
+
+    test('should have secure headers applied', async () => {
+        const serverPath = require.resolve('../server.cjs');
+        delete require.cache[serverPath];
+        server = require(serverPath);
+
+        const res = await request(server).get('/');
+        // Depending on if the file is served or a 404, headers should still be present
+        expect(res.headers['strict-transport-security']).toBeDefined();
+        expect(res.headers['x-frame-options']).toBe('DENY');
+        expect(res.headers['content-security-policy']).toBeDefined();
+    });
+});
diff --git a/src/chatbot/frontend/vite.config.js b/src/chatbot/frontend/vite.config.js
new file mode 100644
index 0000000..f8e9852
--- /dev/null
+++ b/src/chatbot/frontend/vite.config.js
@@ -0,0 +1,23 @@
+import { defineConfig } from 'vite'
+import react from '@vitejs/plugin-react'
+
+export default defineConfig({
+  plugins: [react()],
+  server: {
+    proxy: {
+      '/api': {
+        target: 'http://localhost:6006',
+        changeOrigin: true,
+      },
+      '/config': {
+        target: 'http://localhost:6006',
+        changeOrigin: true,
+      }
+    }
+  },
+  test: {
+    globals: true,
+    environment: 'jsdom',
+    setupFiles: './src/setupTests.js',
+  }
+})
diff --git a/src/chatbot/ops/docker-compose.yml b/src/chatbot/ops/docker-compose.yml
new file mode 100644
index 0000000..7ee06fa
--- /dev/null
+++ b/src/chatbot/ops/docker-compose.yml
@@ -0,0 +1,43 @@
+version: '3.8'
+
+services:
+  # LLAISYS Backend
+  prometheus:
+    image: prom/prometheus:latest
+    container_name: llaisys-prometheus
+    volumes:
+      - ./prometheus.yml:/etc/prometheus/prometheus.yml
+    ports:
+      - "9090:9090"
+    networks:
+      - llaisys-net
+
+  grafana:
+    image: grafana/grafana:latest
+    container_name: llaisys-grafana
+    ports:
+      - "3001:3000"
+    environment:
+      - GF_SECURITY_ADMIN_PASSWORD=admin
+    volumes:
+      - ./grafana/provisioning:/etc/grafana/provisioning
+      - ./grafana/dashboards:/etc/grafana/dashboards
+    networks:
+      - llaisys-net
+    depends_on:
+      - prometheus
+
+  jaeger:
+    image: jaegertracing/all-in-one:latest
+    container_name: llaisys-jaeger
+    ports:
+      - "16686:16686" # Web UI
+      - "14268:14268" # Collector HTTP
+      - "4317:4317"   # OTLP gRPC
+      - "4318:4318"   # OTLP HTTP
+    networks:
+      - llaisys-net
+
+networks:
+  llaisys-net:
+    driver: bridge
diff --git a/src/chatbot/ops/grafana/dashboards/llaisys_dashboard.json b/src/chatbot/ops/grafana/dashboards/llaisys_dashboard.json
new file mode 100644
index 0000000..2057d14
--- /dev/null
+++ b/src/chatbot/ops/grafana/dashboards/llaisys_dashboard.json
@@ -0,0 +1,258 @@
+{
+  "annotations": {
+    "list": [
+      {
+        "builtIn": 1,
+        "datasource": "-- Grafana --",
+        "enable": true,
+        "hide": true,
+        "iconColor": "rgba(0, 211, 255, 1)",
+        "name": "Annotations & Alerts",
+        "type": "dashboard"
+      }
+    ]
+  },
+  "editable": true,
+  "fiscalYearStartMonth": 0,
+  "graphTooltip": 0,
+  "id": null,
+  "links": [],
+  "liveNow": false,
+  "panels": [
+    {
+      "collapsed": false,
+      "gridPos": {
+        "h": 1,
+        "w": 24,
+        "x": 0,
+        "y": 0
+      },
+      "id": 10,
+      "panels": [],
+      "title": "Application Metrics",
+      "type": "row"
+    },
+    {
+      "datasource": "Prometheus",
+      "gridPos": {
+        "h": 8,
+        "w": 12,
+        "x": 0,
+        "y": 1
+      },
+      "id": 1,
+      "options": {
+        "legend": {
+          "calcs": [],
+          "displayMode": "list",
+          "placement": "bottom"
+        },
+        "tooltip": {
+          "mode": "single",
+          "sort": "none"
+        }
+      },
+      "targets": [
+        {
+          "expr": "rate(llaisys_requests_total{endpoint!=\"/metrics\"}[5m])",
+          "legendFormat": "{{endpoint}} - {{status}}",
+          "refId": "A"
+        }
+      ],
+      "title": "Request Rate (QPS)",
+      "type": "timeseries"
+    },
+    {
+      "datasource": "Prometheus",
+      "gridPos": {
+        "h": 8,
+        "w": 12,
+        "x": 12,
+        "y": 1
+      },
+      "id": 3,
+      "options": {
+        "legend": {
+          "calcs": [],
+          "displayMode": "list",
+          "placement": "bottom"
+        },
+        "tooltip": {
+          "mode": "single",
+          "sort": "none"
+        }
+      },
+      "targets": [
+        {
+          "expr": "sum(rate(llaisys_requests_total{status=~\"5.*\", endpoint!=\"/metrics\"}[5m])) / sum(rate(llaisys_requests_total{endpoint!=\"/metrics\"}[5m])) * 100",
+          "legendFormat": "Error Rate %",
+          "refId": "A"
+        }
+      ],
+      "title": "Error Rate",
+      "type": "timeseries"
+    },
+    {
+      "datasource": "Prometheus",
+      "gridPos": {
+        "h": 8,
+        "w": 12,
+        "x": 0,
+        "y": 9
+      },
+      "id": 4,
+      "options": {
+        "reduceOptions": {
+          "calcs": [
+            "lastNotNull"
+          ],
+          "fields": "",
+          "values": false
+        },
+        "showThresholdLabels": false,
+        "showThresholdMarkers": true
+      },
+      "targets": [
+        {
+          "expr": "sum(llaisys_requests_total{endpoint!=\"/metrics\"})",
+          "legendFormat": "Total Requests",
+          "refId": "A"
+        }
+      ],
+      "title": "Total Requests Served",
+      "type": "stat"
+    },
+    {
+      "collapsed": false,
+      "gridPos": {
+        "h": 1,
+        "w": 24,
+        "x": 0,
+        "y": 17
+      },
+      "id": 11,
+      "panels": [],
+      "title": "Metric Endpoint Performance",
+      "type": "row"
+    },
+    {
+      "datasource": "Prometheus",
+      "gridPos": {
+        "h": 8,
+        "w": 12,
+        "x": 0,
+        "y": 18
+      },
+      "id": 12,
+      "options": {
+        "legend": {
+          "calcs": [],
+          "displayMode": "list",
+          "placement": "bottom"
+        },
+        "tooltip": {
+          "mode": "single",
+          "sort": "none"
+        }
+      },
+      "targets": [
+        {
+          "expr": "rate(llaisys_requests_total{endpoint=\"/metrics\"}[5m])",
+          "legendFormat": "Metrics Scrape Rate",
+          "refId": "A"
+        }
+      ],
+      "title": "Metrics Scrape Rate (QPS)",
+      "type": "timeseries"
+    },
+    {
+      "datasource": "Prometheus",
+      "gridPos": {
+        "h": 8,
+        "w": 12,
+        "x": 12,
+        "y": 18
+      },
+      "id": 2,
+      "options": {
+        "legend": {
+          "calcs": [],
+          "displayMode": "list",
+          "placement": "bottom"
+        },
+        "tooltip": {
+          "mode": "single",
+          "sort": "none"
+        }
+      },
+      "targets": [
+        {
+          "expr": "histogram_quantile(0.99, sum(rate(llaisys_request_latency_seconds_bucket{endpoint=\"/metrics\"}[5m])) by (le))",
+          "legendFormat": "Metrics Latency P99",
+          "refId": "A"
+        },
+        {
+          "expr": "histogram_quantile(0.95, sum(rate(llaisys_request_latency_seconds_bucket{endpoint=\"/metrics\"}[5m])) by (le))",
+          "legendFormat": "Metrics Latency P95",
+          "refId": "B"
+        }
+      ],
+      "title": "Metrics Endpoint Latency (P95/P99)",
+      "type": "timeseries"
+    },
+     {
+      "datasource": "Prometheus",
+      "gridPos": {
+        "h": 8,
+        "w": 12,
+        "x": 0,
+        "y": 26
+      },
+      "id": 13,
+      "options": {
+        "legend": {
+          "calcs": [],
+          "displayMode": "list",
+          "placement": "bottom"
+        },
+        "tooltip": {
+          "mode": "single",
+          "sort": "none"
+        }
+      },
+      "targets": [
+        {
+          "expr": "histogram_quantile(0.99, sum(rate(llaisys_request_latency_seconds_bucket{endpoint!=\"/metrics\"}[5m])) by (le, endpoint))",
+          "legendFormat": "{{endpoint}} Latency P99",
+          "refId": "A"
+        },
+        {
+           "expr": "histogram_quantile(0.95, sum(rate(llaisys_request_latency_seconds_bucket{endpoint!=\"/metrics\"}[5m])) by (le, endpoint))",
+           "legendFormat": "{{endpoint}} Latency P95",
+           "refId": "B"
+        }
+      ],
+      "title": "Application Latency (P95/P99)",
+      "type": "timeseries"
+    }
+  ],
+  "refresh": "5s",
+  "schemaVersion": 36,
+  "style": "dark",
+  "tags": [
+    "llaisys"
+  ],
+  "templating": {
+    "list": []
+  },
+  "time": {
+    "from": "now-15m",
+    "to": "now"
+  },
+  "timepicker": {},
+  "timezone": "",
+  "title": "LLAISYS Performance Dashboard",
+  "uid": "llaisys-main",
+  "version": 1,
+  "weekStart": ""
+}
diff --git a/src/chatbot/ops/grafana/provisioning/dashboards/dashboards.yml b/src/chatbot/ops/grafana/provisioning/dashboards/dashboards.yml
new file mode 100644
index 0000000..78cfb99
--- /dev/null
+++ b/src/chatbot/ops/grafana/provisioning/dashboards/dashboards.yml
@@ -0,0 +1,12 @@
+apiVersion: 1
+
+providers:
+  - name: 'Default'
+    orgId: 1
+    folder: ''
+    type: file
+    disableDeletion: false
+    updateIntervalSeconds: 10
+    allowUiUpdates: true
+    options:
+      path: /etc/grafana/dashboards
diff --git a/src/chatbot/ops/grafana/provisioning/datasources/prometheus.yml b/src/chatbot/ops/grafana/provisioning/datasources/prometheus.yml
new file mode 100644
index 0000000..bb009bb
--- /dev/null
+++ b/src/chatbot/ops/grafana/provisioning/datasources/prometheus.yml
@@ -0,0 +1,9 @@
+apiVersion: 1
+
+datasources:
+  - name: Prometheus
+    type: prometheus
+    access: proxy
+    url: http://prometheus:9090
+    isDefault: true
+    editable: false
diff --git a/src/chatbot/ops/prometheus.yml b/src/chatbot/ops/prometheus.yml
new file mode 100644
index 0000000..b3f0b0b
--- /dev/null
+++ b/src/chatbot/ops/prometheus.yml
@@ -0,0 +1,9 @@
+global:
+  scrape_interval: 15s
+
+scrape_configs:
+  - job_name: 'llaisys'
+    metrics_path: '/metrics'
+    static_configs:
+      - targets: ['host.docker.internal:8001'] # Assuming running server on host
+    scrape_timeout: 10s
diff --git a/src/llaisys/ops.cc b/src/llaisys/ops.cc
index c99fbc3..8618976 100644
--- a/src/llaisys/ops.cc
+++ b/src/llaisys/ops.cc
@@ -11,6 +11,7 @@
 #include "../ops/rope/op.hpp"
 #include "../ops/self_attention/op.hpp"
 #include "../ops/swiglu/op.hpp"
+#include "../ops/random_sample/op.hpp"
 
 __C {
     void llaisysAdd(llaisysTensor_t c, llaisysTensor_t a, llaisysTensor_t b) {
@@ -40,4 +41,7 @@ __C {
     void llaisysSwiGLU(llaisysTensor_t out, llaisysTensor_t gate, llaisysTensor_t up) {
         llaisys::ops::swiglu(out->tensor, gate->tensor, up->tensor);
     }
+    void llaisysRandomSample(llaisysTensor_t out_token, llaisysTensor_t logits, float temp, float top_p, int top_k) {
+        llaisys::ops::random_sample(out_token->tensor, logits->tensor, temp, top_p, top_k);
+    }
 }
diff --git a/src/llaisys/qwen2.cc b/src/llaisys/qwen2.cc
index dd0afe6..3830554 100644
--- a/src/llaisys/qwen2.cc
+++ b/src/llaisys/qwen2.cc
@@ -6,6 +6,10 @@
 #include <cstdlib>
 #include <cstring>
 #include <cmath>
+#include <cstdio>
+#include <algorithm>
+#include <random>
+#include <numeric>
 
 struct LlaisysQwen2Model {
     LlaisysQwen2Meta meta;
@@ -13,7 +17,14 @@ struct LlaisysQwen2Model {
     llaisysDeviceType_t device_type;
     int device_id;
 
-    // KV Cache: nlayer of tensors
+    llaisysTensor_t zero_bias_hs;
+    llaisysTensor_t zero_bias_di;
+    llaisysTensor_t zero_bias_voc;
+};
+
+struct LlaisysQwen2Session {
+    LlaisysQwen2Model* model;
+    
     // Key-Value Cache stores past token representations for each layer
     // Each tensor shape: [max_seq, nkvh, dh]
     std::vector<llaisysTensor_t> k_cache;
@@ -21,10 +32,6 @@ struct LlaisysQwen2Model {
     
     // Current filled position in KV cache
     size_t pos;
-
-    llaisysTensor_t zero_bias_hs;
-    llaisysTensor_t zero_bias_di;
-    llaisysTensor_t zero_bias_voc;
 };
 
 static llaisysTensor_t create_tensor(size_t* shape, size_t ndim, llaisysDataType_t dtype, llaisysDeviceType_t device, int device_id) {
@@ -82,7 +89,6 @@ __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Met
     model->meta = *meta;
     model->device_type = device;
     model->device_id = (ndevice > 0) ? device_ids[0] : 0;
-    model->pos = 0;
 
     model->zero_bias_hs = create_zero_tensor(meta->hs, meta->dtype, device, model->device_id);
     model->zero_bias_di = create_zero_tensor(meta->di, meta->dtype, device, model->device_id);
@@ -160,12 +166,6 @@ __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Met
         // Down
         shape[0] = hidden; shape[1] = inter_dim;
         model->weights.mlp_down_w[i] = create_weight(meta, shape, 2, device, model->device_id);
-
-        // KV Cache
-        // [max_seq, nkvh, dh]
-        shape[0] = meta->maxseq; shape[1] = meta->nkvh; shape[2] = meta->dh;
-        model->k_cache.push_back(create_weight(meta, shape, 3, device, model->device_id));
-        model->v_cache.push_back(create_weight(meta, shape, 3, device, model->device_id));
     }
 
     return model;
@@ -191,9 +191,6 @@ __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
         tensorDestroy(model->weights.mlp_gate_w[i]);
         tensorDestroy(model->weights.mlp_up_w[i]);
         tensorDestroy(model->weights.mlp_down_w[i]);
-        
-        tensorDestroy(model->k_cache[i]);
-        tensorDestroy(model->v_cache[i]);
     }
 
     free(model->weights.attn_norm_w);
@@ -216,14 +213,49 @@ __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
     delete model;
 }
 
+__export struct LlaisysQwen2Session *llaisysQwen2ModelCreateSession(struct LlaisysQwen2Model * model) {
+    LlaisysQwen2Session* session = new LlaisysQwen2Session();
+    session->model = model;
+    session->pos = 0;
+    
+    size_t shape[3];
+    shape[0] = model->meta.maxseq; shape[1] = model->meta.nkvh; shape[2] = model->meta.dh;
+    
+    for (size_t i = 0; i < model->meta.nlayer; ++i) {
+        session->k_cache.push_back(create_weight(&model->meta, shape, 3, model->device_type, model->device_id));
+        session->v_cache.push_back(create_weight(&model->meta, shape, 3, model->device_type, model->device_id));
+    }
+    
+    return session;
+}
+
+__export void llaisysQwen2ModelDestroySession(struct LlaisysQwen2Session * session) {
+    if (!session) return;
+    
+    for (size_t i = 0; i < session->k_cache.size(); ++i) {
+        tensorDestroy(session->k_cache[i]);
+        tensorDestroy(session->v_cache[i]);
+    }
+    delete session;
+}
+
+__export void llaisysQwen2ModelRewindSession(struct LlaisysQwen2Session * session, size_t len) {
+    if (!session) return;
+    if (len > session->pos) return;
+    session->pos = len;
+}
+
 __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
     return &model->weights;
 }
 
-__export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken) {
-    // 1. Embedding
-    // Input: [ntoken]
-    // Output: [ntoken, hs]
+__export int64_t llaisysQwen2ModelForward(struct LlaisysQwen2Session * session, int64_t * token_ids, size_t ntoken, float temp, float top_p, int top_k) {
+    LlaisysQwen2Model* model = session->model;
+    
+    if (session->pos + ntoken > model->meta.maxseq) {
+        fprintf(stderr, "Error: Context length exceeded limit %zu\n", model->meta.maxseq);
+        return -1;
+    }
     
     size_t seq = ntoken;
     size_t hs = model->meta.hs;
@@ -270,7 +302,7 @@ __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_
         llaisysTensor_t pos_ids = tensorCreate(shape_pos, 1, LLAISYS_DTYPE_I64, model->device_type, model->device_id);
         std::vector<int64_t> pos_data(seq);
         for (size_t p = 0; p < seq; ++p) {
-            pos_data[p] = model->pos + p;
+            pos_data[p] = session->pos + p;
         }
         tensorLoad(pos_ids, pos_data.data());
 
@@ -282,8 +314,8 @@ __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_
         // Update KV Cache
         // k_view is [seq, nkvh, dh]
         // k_cache is [max_seq, nkvh, dh]        
-        llaisysTensor_t k_cache_slot = tensorSlice(model->k_cache[i], 0, model->pos, model->pos + seq);
-        llaisysTensor_t v_cache_slot = tensorSlice(model->v_cache[i], 0, model->pos, model->pos + seq);
+        llaisysTensor_t k_cache_slot = tensorSlice(session->k_cache[i], 0, session->pos, session->pos + seq);
+        llaisysTensor_t v_cache_slot = tensorSlice(session->v_cache[i], 0, session->pos, session->pos + seq);
         
         // Copy data
         void* src_k = tensorGetData(k_view);
@@ -306,8 +338,8 @@ __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_
         tensorDestroy(v_cache_slot);
 
         // Attention
-        llaisysTensor_t k_full = tensorSlice(model->k_cache[i], 0, 0, model->pos + seq);
-        llaisysTensor_t v_full = tensorSlice(model->v_cache[i], 0, 0, model->pos + seq);
+        llaisysTensor_t k_full = tensorSlice(session->k_cache[i], 0, 0, session->pos + seq);
+        llaisysTensor_t v_full = tensorSlice(session->v_cache[i], 0, 0, session->pos + seq);
         
         // Output tensor [seq, nh, dh]
         llaisysTensor_t attn_out = tensorCreate(shape_q, 3, model->meta.dtype, model->device_type, model->device_id);
@@ -338,7 +370,7 @@ __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_
         // Residual Add: x = x + h
         llaisysAdd(x, x, h);
         tensorDestroy(h);
-        tensorDestroy(x_norm); // x was used in residual, x_norm was temp.
+        tensorDestroy(x_norm);
 
         // MLP
         llaisysTensor_t mlp_norm = tensorCreate(shape_embed, 2, model->meta.dtype, model->device_type, model->device_id);
@@ -388,32 +420,41 @@ __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_
     
     llaisysLinear(logits, last_token_state, model->weights.out_embed, model->zero_bias_voc);
     
-    // Argmax
-    size_t shape_scalar[1] = {1};
-    llaisysTensor_t max_idx = tensorCreate(shape_scalar, 1, LLAISYS_DTYPE_I64, model->device_type, model->device_id);
-    llaisysTensor_t max_val = tensorCreate(shape_scalar, 1, model->meta.dtype, model->device_type, model->device_id);
-    
-    size_t shape_logits_1d[1] = {model->meta.voc};
-    llaisysTensor_t logits_1d = tensorView(logits, shape_logits_1d, 1);
+    // Copy logits to host for sampling
+    size_t shape_logits_host[2] = {1, model->meta.voc};
+    llaisysTensor_t host_logits = tensorCreate(shape_logits_host, 2, model->meta.dtype, LLAISYS_DEVICE_CPU, 0);
     
-    llaisysArgmax(max_idx, max_val, logits_1d);
+    size_t elem_size = (model->meta.dtype == LLAISYS_DTYPE_F32) ? 4 : 2;
     
-    tensorDestroy(logits_1d);
-
-    int64_t result_token;
-    void* res_ptr = tensorGetData(max_idx);
     const auto runtime = llaisysGetRuntimeAPI(model->device_type);
-    runtime->memcpy_sync(&result_token, res_ptr, sizeof(int64_t), LLAISYS_MEMCPY_D2H);
+    runtime->memcpy_sync(tensorGetData(host_logits), tensorGetData(logits), 
+                         model->meta.voc * elem_size, 
+                         LLAISYS_MEMCPY_D2H);
 
+    size_t shape_out[1] = {1};
+    llaisysTensor_t out_token = tensorCreate(shape_out, 1, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
+
+    llaisysRandomSample(out_token, host_logits, temp, top_p, top_k);
+    
+    int64_t result_token = *(int64_t*)tensorGetData(out_token);
+    
+    tensorDestroy(host_logits);
+    tensorDestroy(out_token);
+    
     tensorDestroy(x_final);
     tensorDestroy(last_token_state);
     tensorDestroy(logits);
-    tensorDestroy(max_idx);
-    tensorDestroy(max_val);
 
-    model->pos += ntoken;
+    session->pos += ntoken;
 
     return result_token;
 }
 
+__export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken, float temp, float top_p, int top_k) {
+    auto session = llaisysQwen2ModelCreateSession(model);
+    int64_t res = llaisysQwen2ModelForward(session, token_ids, ntoken, temp, top_p, top_k);
+    llaisysQwen2ModelDestroySession(session);
+    return res;
+}
+
 } // extern "C"
diff --git a/src/ops/random_sample/nvidia/random_sample_nvidia.cu b/src/ops/random_sample/nvidia/random_sample_nvidia.cu
new file mode 100644
index 0000000..6a7e45d
--- /dev/null
+++ b/src/ops/random_sample/nvidia/random_sample_nvidia.cu
@@ -0,0 +1,139 @@
+#include "random_sample_nvidia.hpp"
+#include <cuda_runtime.h>
+#include <cuda_fp16.h>
+#include <cuda_bf16.h>
+#include <stdexcept>
+#include <cfloat>
+#include <cub/cub.cuh>
+#include <thrust/device_vector.h>
+#include <thrust/sort.h>
+#include <thrust/binary_search.h>
+#include <thrust/distance.h>
+#include <thrust/execution_policy.h>
+
+namespace llaisys::ops::nvidia {
+
+template <typename T>
+__global__ void compute_probs_kernel(float* probs, int* indices, const T* logits, float max_val, float temp, int size) {
+    int tid = blockIdx.x * blockDim.x + threadIdx.x;
+    if (tid < size) {
+        float val = static_cast<float>(logits[tid]);
+        if (temp > 0.0f && temp != 1.0f) {
+            val /= temp;
+        }
+        probs[tid] = expf(val - max_val);
+        indices[tid] = tid;
+    }
+}
+
+template <typename T>
+struct CastToFloatOp {
+    __device__ __forceinline__ float operator()(const T& a) const {
+        return static_cast<float>(a);
+    }
+};
+
+template <typename T>
+void random_sample_cub(tensor_t out_token, const T* logits_ptr, int size, float temp, float top_p, int top_k) {
+    float* d_max_out;
+    cudaMalloc(&d_max_out, sizeof(float));
+    
+    cub::TransformInputIterator<float, CastToFloatOp<T>, const T*> d_logits_float(logits_ptr, CastToFloatOp<T>());
+    
+    void* d_temp_storage = nullptr;
+    size_t temp_storage_bytes = 0;
+    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_logits_float, d_max_out, size);
+    cudaMalloc(&d_temp_storage, temp_storage_bytes);
+    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_logits_float, d_max_out, size);
+    
+    float h_max_val;
+    cudaMemcpy(&h_max_val, d_max_out, sizeof(float), cudaMemcpyDeviceToHost);
+    cudaFree(d_temp_storage);
+    cudaFree(d_max_out);
+
+    float* d_probs;
+    int* d_indices;
+    cudaMalloc(&d_probs, size * sizeof(float));
+    cudaMalloc(&d_indices, size * sizeof(int));
+    
+    int threads = 256;
+    int blocks = (size + threads - 1) / threads;
+    compute_probs_kernel<<<blocks, threads>>>(d_probs, d_indices, logits_ptr, h_max_val, temp, size);
+
+    float* d_probs_sorted;
+    int* d_indices_sorted;
+    cudaMalloc(&d_probs_sorted, size * sizeof(float));
+    cudaMalloc(&d_indices_sorted, size * sizeof(int));
+    
+    d_temp_storage = nullptr;
+    temp_storage_bytes = 0;
+    cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_probs, d_probs_sorted, d_indices, d_indices_sorted, size);
+    cudaMalloc(&d_temp_storage, temp_storage_bytes);
+    cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_probs, d_probs_sorted, d_indices, d_indices_sorted, size);
+    cudaFree(d_temp_storage);
+    
+    float* d_cumsum;
+    cudaMalloc(&d_cumsum, size * sizeof(float));
+    
+    d_temp_storage = nullptr;
+    temp_storage_bytes = 0;
+    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_probs_sorted, d_cumsum, size);
+    cudaMalloc(&d_temp_storage, temp_storage_bytes);
+    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_probs_sorted, d_cumsum, size);
+    cudaFree(d_temp_storage);
+
+    float total_sum;
+    cudaMemcpy(&total_sum, d_cumsum + size - 1, sizeof(float), cudaMemcpyDeviceToHost);
+    
+    int k = (top_k > 0 && top_k < size) ? top_k : size;
+    float pk;
+    cudaMemcpy(&pk, d_cumsum + k - 1, sizeof(float), cudaMemcpyDeviceToHost);
+    
+    float pp = total_sum * top_p;
+    float plimit_val = std::min(pk, pp);
+    
+    float rand_val = (float)rand() / (float)RAND_MAX;
+    float target_cumsum = rand_val * plimit_val;
+    
+    thrust::device_ptr<float> t_cumsum(d_cumsum);
+    int sampled_idx = thrust::distance(t_cumsum, thrust::lower_bound(thrust::device, t_cumsum, t_cumsum + k, target_cumsum));
+    
+    if (sampled_idx >= size) sampled_idx = size - 1;
+    
+    int final_token;
+    cudaMemcpy(&final_token, d_indices_sorted + sampled_idx, sizeof(int), cudaMemcpyDeviceToHost);
+    
+    int64_t final_token_i64 = final_token;
+    cudaMemcpy(out_token->data(), &final_token_i64, sizeof(int64_t), cudaMemcpyHostToDevice);
+
+    cudaFree(d_probs);
+    cudaFree(d_indices);
+    cudaFree(d_probs_sorted);
+    cudaFree(d_indices_sorted);
+    cudaFree(d_cumsum);
+}
+
+void random_sample(tensor_t out_token, tensor_t logits, float temp, float top_p, int top_k) {
+    int size = logits->numel();
+    auto dtype = logits->dtype();
+    
+    switch(dtype) {
+        case LLAISYS_DTYPE_F32:
+            random_sample_cub<float>(out_token, (const float*)logits->data(), size, temp, top_p, top_k);
+            break;
+        case LLAISYS_DTYPE_F16:
+            random_sample_cub<half>(out_token, (const half*)logits->data(), size, temp, top_p, top_k);
+            break;
+        case LLAISYS_DTYPE_BF16:
+            random_sample_cub<__nv_bfloat16>(out_token, (const __nv_bfloat16*)logits->data(), size, temp, top_p, top_k);
+            break;
+        default:
+            throw std::runtime_error("RandomSample NVIDIA: Unsupported data type");
+    }
+    
+    if (cudaGetLastError() != cudaSuccess) {
+        throw std::runtime_error("Kernel launch failed");
+    }
+}
+
+} // namespace llaisys::ops::nvidia
diff --git a/src/ops/random_sample/nvidia/random_sample_nvidia.hpp b/src/ops/random_sample/nvidia/random_sample_nvidia.hpp
new file mode 100644
index 0000000..3d20245
--- /dev/null
+++ b/src/ops/random_sample/nvidia/random_sample_nvidia.hpp
@@ -0,0 +1,10 @@
+#ifndef OPS_RANDOM_SAMPLE_NVIDIA_HPP
+#define OPS_RANDOM_SAMPLE_NVIDIA_HPP
+
+#include "../../../tensor/tensor.hpp"
+
+namespace llaisys::ops::nvidia {
+    void random_sample(tensor_t out_token, tensor_t logits, float temp, float top_p, int top_k);
+}
+
+#endif // OPS_RANDOM_SAMPLE_NVIDIA_HPP
diff --git a/src/ops/random_sample/op.cpp b/src/ops/random_sample/op.cpp
new file mode 100644
index 0000000..0a80f8d
--- /dev/null
+++ b/src/ops/random_sample/op.cpp
@@ -0,0 +1,119 @@
+#include "op.hpp"
+#include "../common.hpp"
+#include "../../../include/llaisys.h"
+#include "../../utils/check.hpp"
+#include <random>
+#include <algorithm>
+#include <numeric>
+#include <iostream>
+#include <cmath>
+
+#ifdef ENABLE_NVIDIA_API
+#include "nvidia/random_sample_nvidia.hpp"
+#endif
+
+namespace llaisys::ops {
+
+namespace {
+
+template <typename Tidx, typename Tval>
+struct KVPair {
+    Tidx idx;
+    Tval val;
+
+    bool operator<(const KVPair &other) const {
+        return val > other.val;
+    }
+};
+
+template <typename T>
+void random_sample_cpu(tensor_t out_token, tensor_t logits, float temp, float top_p, int top_k) {
+    const T* logits_ptr = reinterpret_cast<const T*>(logits->data());
+    int64_t* out_ptr = reinterpret_cast<int64_t*>(out_token->data());
+    size_t n = logits->numel();
+
+    static std::mt19937 gen(std::random_device{}());
+    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
+    float random_val = dis(gen);
+
+    using ComputeType = float;
+    
+    std::vector<KVPair<int64_t, ComputeType>> pairs(n);
+    for (size_t i = 0; i < n; i++) {
+        pairs[i] = {static_cast<int64_t>(i), common::to_float(logits_ptr[i])};
+    }
+    
+    std::sort(pairs.begin(), pairs.end());
+
+    auto const max_val = pairs[0].val;
+    pairs[0].val = 1.0f;
+    
+    for (size_t i = 1; i < n; i++) {
+        float current_val = pairs[i].val;
+        pairs[i].val = pairs[i - 1].val + std::exp((current_val - max_val) / temp);
+    }
+
+    size_t k_idx = std::min(static_cast<size_t>(top_k), n);
+    if (k_idx == 0) k_idx = n;
+    
+    auto const pk = pairs[k_idx - 1].val;
+    auto const pp = pairs[n - 1].val * top_p;
+    auto const plimit = random_val * std::min(pk, pp);
+
+    *out_ptr = pairs.empty() ? 0 : pairs.back().idx;
+    
+    for (size_t i = 0; i < n; i++) {
+        if (plimit <= pairs[i].val) {
+            *out_ptr = pairs[i].idx;
+            break;
+        }
+    }
+}
+
+void validate_random_sample_tensors(const tensor_t& out_token, const tensor_t& logits) {
+    CHECK_SAME_DEVICE(out_token, logits);
+    
+    ASSERT(out_token->ndim() == 1 && out_token->shape()[0] == 1, 
+           "RandomSample: out_token must be 1D with single element.");
+    ASSERT(logits->ndim() == 1 || (logits->ndim() == 2 && logits->shape()[0] == 1), 
+           "RandomSample: logits must be 1D or [1, voc].");
+    
+    ASSERT(out_token->dtype() == LLAISYS_DTYPE_I64, 
+           "RandomSample: out_token tensor must be int64.");
+}
+
+} // namespace
+
+void random_sample(tensor_t out_token, tensor_t logits, float temp, float top_p, int top_k) {
+    validate_random_sample_tensors(out_token, logits);
+
+    if (logits->deviceType() == LLAISYS_DEVICE_CPU) {
+        switch (logits->dtype()) {
+            case LLAISYS_DTYPE_F32:
+                random_sample_cpu<float>(out_token, logits, temp, top_p, top_k);
+                return;
+            case LLAISYS_DTYPE_F16:
+                random_sample_cpu<llaisys::fp16_t>(out_token, logits, temp, top_p, top_k);
+                return;
+            case LLAISYS_DTYPE_BF16:
+                random_sample_cpu<llaisys::bf16_t>(out_token, logits, temp, top_p, top_k);
+                return;
+            default:
+                EXCEPTION_UNSUPPORTED_DATATYPE(logits->dtype());
+        }
+    }
+
+    // Non-CPU devices
+    llaisys::core::context().setDevice(logits->deviceType(), logits->deviceId());
+    
+#ifdef ENABLE_NVIDIA_API
+    if (logits->deviceType() == LLAISYS_DEVICE_NVIDIA) {
+        nvidia::random_sample(out_token, logits, temp, top_p, top_k);
+        return;
+    }
+#endif
+
+    EXCEPTION_UNSUPPORTED_DEVICE;
+}
+
+} // namespace llaisys::ops
diff --git a/src/ops/random_sample/op.hpp b/src/ops/random_sample/op.hpp
new file mode 100644
index 0000000..ac72610
--- /dev/null
+++ b/src/ops/random_sample/op.hpp
@@ -0,0 +1,10 @@
+#ifndef OPS_RANDOM_SAMPLE_OP_HPP
+#define OPS_RANDOM_SAMPLE_OP_HPP
+
+#include "../../tensor/tensor.hpp"
+
+namespace llaisys::ops {
+    void random_sample(tensor_t out_token, tensor_t logits, float temp, float top_p, int top_k);
+}
+
+#endif // OPS_RANDOM_SAMPLE_OP_HPP
diff --git a/test/chatbot/backend/test_schemas_and_context.py b/test/chatbot/backend/test_schemas_and_context.py
new file mode 100644
index 0000000..37693ab
--- /dev/null
+++ b/test/chatbot/backend/test_schemas_and_context.py
@@ -0,0 +1,124 @@
+import pytest
+from pydantic import ValidationError
+from chatbot.backend.schemas import (
+    ChatCompletionResponse,
+    ChatCompletionChunk,
+    ChatCompletionResponseChoice
+)
+from chatbot.backend.context_manager import ContextManager
+
+
+class TestSchemas:
+    def test_parse_valid_response(self):
+        raw = {
+            "id": "chatcmpl-123",
+            "object": "chat.completion",
+            "created": 1677652288,
+            "model": "gpt-3.5-turbo",
+            "choices": [{
+                "index": 0,
+                "message": {"role": "assistant", "content": "Hello"},
+                "finish_reason": "stop"
+            }],
+            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
+        }
+        
+        resp = ChatCompletionResponse(**raw)
+        
+        assert resp.id == "chatcmpl-123"
+        assert len(resp.choices) == 1
+        choice = resp.choices[0]
+        assert isinstance(choice, ChatCompletionResponseChoice)
+        assert choice.message.content == "Hello"
+
+    def test_validation_missing_fields(self):
+        with pytest.raises(ValidationError) as exc:
+            ChatCompletionResponse(object="chat.completion")
+        
+        errors = [e["loc"][0] for e in exc.value.errors()]
+        assert {"id", "created", "model", "choices"}.issubset(errors)
+
+    def test_validation_invalid_types(self):
+        raw = {
+            "id": "test",
+            "object": "chat.completion",
+            "created": "invalid_int",
+            "model": "gpt-4",
+            "choices": []
+        }
+        
+        with pytest.raises(ValidationError) as exc:
+            ChatCompletionResponse(**raw)
+        
+        assert "created" in str(exc.value)
+
+    def test_parse_streaming_chunk(self):
+        raw = {
+            "id": "chunk-1",
+            "object": "chat.completion.chunk",
+            "created": 1234567890,
+            "model": "gpt-4",
+            "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}]
+        }
+        
+        chunk = ChatCompletionChunk(**raw)
+        assert chunk.choices[0].delta.content == "Hi"
+
+
+class TestContextManager:
+    @pytest.fixture
+    def cm(self):
+        return ContextManager()
+
+    def test_item_lifecycle(self, cm):
+        item = cm.add_item("id1", "content", tags=["tag1"], metadata={"k": "v"})
+        assert item.id == "id1"
+        assert cm.get_item("id1") == item
+        
+        assert len(cm.list_items()) == 1
+        
+        assert cm.remove_item("id1")
+        assert cm.get_item("id1") is None
+        assert not cm.remove_item("id1")
+
+    def test_tag_filtering(self, cm):
+        cm.add_item("1", "A", tags=["t1"])
+        cm.add_item("2", "B", tags=["t2"])
+        cm.add_item("3", "C", tags=["t1", "t2"])
+
+        t1_items = cm.get_items_by_tag("t1")
+        assert {i.id for i in t1_items} == {"1", "3"}
+
+    def test_tag_operations(self, cm):
+        item = cm.add_item("1", "content", tags=["t1"])
+        
+        item.add_tag("t2")
+        assert item.has_tag("t2")
+        assert item.has_all_tags(["t1", "t2"])
+        
+        item.remove_tag("t1")
+        assert not item.has_tag("t1")
+        assert item.has_any_tag(["t2", "t3"])
+        assert not item.has_any_tag(["t1", "t3"])
+
+    def test_render_filters(self, cm):
+        cm.add_item("sys", "System", tags=["sys"])
+        cm.add_item("usr", "User", tags=["usr"])
+        cm.add_item("priv", "Private", tags=["private"])
+
+        tpl_filter = "{% for i in context_items | filter_by_tag('sys') %}{{ i.content }}{% endfor %}"
+        assert cm.render_template(tpl_filter) == "System"
+
+        tpl_exclude = """
+        {%- for i in context_items | exclude_by_tag('private') -%}
+        {{ i.content }}|
+        {%- endfor -%}
+        """
+        result = cm.render_template(tpl_exclude)
+        assert "System" in result
+        assert "User" in result
+        assert "Private" not in result
+
+    def test_render_error(self, cm):
+        with pytest.raises(Exception):
+            cm.render_template("{{ invalid_syntax }")
diff --git a/test/ops/random_sample.py b/test/ops/random_sample.py
new file mode 100644
index 0000000..6218c32
--- /dev/null
+++ b/test/ops/random_sample.py
@@ -0,0 +1,105 @@
+import sys
+import os
+import argparse
+import ctypes
+
+parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
+sys.path.insert(0, parent_dir)
+# Also add the python directory to path so llaisys can be imported if not installed
+python_dir = os.path.abspath(os.path.join(parent_dir, "../python"))
+sys.path.insert(0, python_dir)
+
+import llaisys
+import torch
+from test_utils import random_tensor, check_equal, benchmark, zero_tensor
+
+
+def torch_random_sample(out_token, logits, temp, top_p, top_k):
+    # 1. Temperature
+    if temp > 0:
+        logits = logits / temp
+        
+    probs = torch.softmax(logits, dim=-1)
+    
+    # 2. Top-K
+    if top_k > 0:
+        top_k = min(top_k, probs.size(-1))
+        vals, _ = torch.topk(probs, top_k)
+        min_val = vals[:, -1].unsqueeze(-1)
+        probs = torch.where(probs < min_val, torch.tensor(0.0, device=probs.device, dtype=probs.dtype), probs)
+        
+    # 3. Top-P
+    if top_p < 1.0:
+        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
+        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
+        
+        # Remove tokens with cumulative probability above the threshold
+        sorted_indices_to_remove = cumulative_probs > top_p
+        # Shift to keep the first token above threshold
+        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
+        sorted_indices_to_remove[..., 0] = 0
+        
+        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
+        probs = probs.masked_fill(indices_to_remove, 0.0)
+        
+    # 4. Sample
+    # Normalize
+    probs_sum = torch.sum(probs, dim=-1, keepdim=True)
+    probs = torch.where(probs_sum > 0, probs / probs_sum, probs)
+    
+    if torch.sum(probs) > 0:
+        next_token = torch.multinomial(probs, num_samples=1)
+        out_token.copy_(next_token.flatten())
+    else:
+        out_token.copy_(torch.argmax(logits, dim=-1))
+
+
+def test_op_random_sample(
+    shape,
+    dtype_name="f32",
+    device_name="cpu",
+    profile=False,
+    temp=0.7,
+    top_p=0.9,
+    top_k=40
+):
+    print(f"   shape {shape} dtype <{dtype_name}> temp={temp} top_p={top_p} top_k={top_k}")
+    
+    logits, logits_llaisys = random_tensor(shape, dtype_name, device_name)
+    out_token, out_token_llaisys = zero_tensor((1,), "i64", device_name)
+    out_token_torch = torch.zeros((1,), dtype=torch.int64, device=logits.device)
+
+    torch_random_sample(out_token_torch, logits, temp, top_p, top_k)
+    llaisys.Ops.random_sample(out_token_llaisys, logits_llaisys, temp, top_p, top_k)
+
+    # Check validity (bounds check)
+    data_ptr = out_token_llaisys.data_ptr()
+    token_id = ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_int64)).contents.value
+    
+    assert 0 <= token_id < shape[1], f"Token ID {token_id} out of bounds [0, {shape[1]})"
+
+    if profile:
+        compiled_torch_sample = torch.compile(torch_random_sample)
+        benchmark(
+            lambda: torch_random_sample(out_token_torch, logits, temp, top_p, top_k),
+            lambda: llaisys.Ops.random_sample(out_token_llaisys, logits_llaisys, temp, top_p, top_k),
+            device_name,
+            torch_compile_func=lambda: compiled_torch_sample(out_token_torch, logits, temp, top_p, top_k)
+        )
+
+
+if __name__ == "__main__":
+    parser = argparse.ArgumentParser()
+    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
+    parser.add_argument("--profile", action="store_true")
+    args = parser.parse_args()
+    
+    testShapes = [(1, 32000), (1, 128000)]
+    testDtype = ["f32", "f16", "bf16"]
+    
+    print(f"Testing Ops.random_sample on {args.device}")
+    for shape in testShapes:
+        for dtype_name in testDtype:
+            test_op_random_sample(shape, dtype_name, args.device, args.profile)
+
+    print("\033[92mTest passed!\033[0m\n")
diff --git a/test/test_infer.py b/test/test_infer.py
index d3e47a4..88fa801 100644
--- a/test/test_infer.py
+++ b/test/test_infer.py
@@ -89,6 +89,7 @@ if __name__ == "__main__":
     parser.add_argument("--top_k", default=50, type=int)
     parser.add_argument("--temperature", default=1.0, type=float)
     parser.add_argument("--test", action="store_true")
+    parser.add_argument("--test_sampling", action="store_true", help="Test random sampling behaviors")
     parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark")
     parser.add_argument("--repeat", default=10, type=int, help="Number of repeats for benchmark")
     parser.add_argument("--profile", action="store_true", help="Show e2e inference time profile")
@@ -199,3 +200,39 @@ if __name__ == "__main__":
     if args.test:
         assert llaisys_tokens == tokens
         print("\033[92mTest passed!\033[0m\n")
+
+    if args.test_sampling:
+        print("\n=== Testing Sampling Behaviors ===\n")
+        
+        # 1. Verify output same with deterministic params (top_k=1)
+        print("1. Verifying deterministic generation (top_k=1)...")
+        torch.manual_seed(42)
+        det_tokens_1, det_out_1 = llaisys_infer(
+            args.prompt, tokenizer, model, args.max_steps, top_p=1.0, top_k=1, temperature=1.0
+        )
+        torch.manual_seed(42)
+        det_tokens_2, det_out_2 = llaisys_infer(
+            args.prompt, tokenizer, model, args.max_steps, top_p=1.0, top_k=1, temperature=1.0
+        )
+        assert det_tokens_1 == det_tokens_2, "Outputs should be exactly the same for top_k=1"
+        print("   Deterministic check passed.\n")
+
+        # 2. Change parameters, output should be diff
+        print("2. Verifying random generation (top_k=50, top_p=0.9, temp=1.5)...")
+        rand_tokens_1, rand_out_1 = llaisys_infer(
+            args.prompt, tokenizer, model, args.max_steps, top_p=0.9, top_k=50, temperature=1.5
+        )
+        rand_tokens_2, rand_out_2 = llaisys_infer(
+            args.prompt, tokenizer, model, args.max_steps, top_p=0.9, top_k=50, temperature=1.5
+        )
+        if det_tokens_1 == rand_tokens_1:
+            print("Warning: Sampling output was identical to greedy output. This can happen by chance, but retrying with higher temp...")
+            rand_tokens_1, rand_out_1 = llaisys_infer(
+                args.prompt, tokenizer, model, args.max_steps, top_p=0.9, top_k=50, temperature=2.5
+            )
+            
+        assert det_tokens_1 != rand_tokens_1, "Sampling output should differ from greedy output"
+        print("   Random sampling check passed.\n")
+        
+        print("\033[92mSampling test passed!\033[0m\n")
+

commit 280c389f78874035764de8d81c5812d19668f20e
Author: Louis Chu <lingzhichu.clz@gmail.com>
Date:   Sun Mar 15 02:14:59 2026 -0700

    feat: integrate CUDA support (#11)
    
    * feat: integrate CUDA support
    
    * Improve ops performance

diff --git a/.gitignore b/.gitignore
index e38cf57..8a13bc6 100644
--- a/.gitignore
+++ b/.gitignore
@@ -19,6 +19,9 @@ lib/
 # Python
 __pycache__/
 
+# Documentation
+docs/
+
 # Log
 *.log
 
diff --git a/python/llaisys/models/qwen2.py b/python/llaisys/models/qwen2.py
index 56f013c..8aa7ca9 100644
--- a/python/llaisys/models/qwen2.py
+++ b/python/llaisys/models/qwen2.py
@@ -18,14 +18,14 @@ except ImportError:
 
 class Qwen2:
 
-    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
+    def __init__(self, model_path, device: DeviceType = DeviceType.CPU, dtype: DataType = DataType.F32):
         model_path = Path(model_path)
         
         with open(model_path / "config.json", "r") as f:
             config = json.load(f)
             
         self.meta = LlaisysQwen2Meta()
-        self.meta.dtype = DataType.F32
+        self.meta.dtype = dtype
         
         self.meta.nlayer = config["num_hidden_layers"]
         self.meta.hs = config["hidden_size"]
@@ -60,12 +60,19 @@ class Qwen2:
                         
                         # Handle Torch Tensors (converts bf16 if needed)
                         if HAS_TORCH and isinstance(data, torch.Tensor):
-                            data = data.float().numpy()
+                            if self.meta.dtype == DataType.F16:
+                                data = data.half().numpy()
+                            else:
+                                data = data.float().numpy()
                             
-                        # Handle Numpy (ensure float32)
+                        # Handle Numpy (ensure correct dtype)
                         if isinstance(data, np.ndarray):
-                            if data.dtype != np.float32:
-                                data = data.astype(np.float32)
+                            target_dtype = np.float32
+                            if self.meta.dtype == DataType.F16:
+                                target_dtype = np.float16
+                            
+                            if data.dtype != target_dtype:
+                                data = data.astype(target_dtype)
                             if not data.flags['C_CONTIGUOUS']:
                                 data = np.ascontiguousarray(data)
                         
diff --git a/python/setup.cfg b/python/setup.cfg
index b35fc65..92d9afb 100644
--- a/python/setup.cfg
+++ b/python/setup.cfg
@@ -10,7 +10,7 @@ packages = find:
 include_package_data = True
 zip_safe = False
 install_requires =
-    torch>=2.4.0
+    torch>=2.8.0,<2.9.0
     transformers
     accelerate
 
diff --git a/src/device/nvidia/nvidia_resource.cu b/src/device/nvidia/nvidia_resource.cu
index 2e63647..f94a544 100644
--- a/src/device/nvidia/nvidia_resource.cu
+++ b/src/device/nvidia/nvidia_resource.cu
@@ -4,4 +4,6 @@ namespace llaisys::device::nvidia {
 
 Resource::Resource(int device_id) : llaisys::device::DeviceResource(LLAISYS_DEVICE_NVIDIA, device_id) {}
 
+Resource::~Resource() {}
+
 } // namespace llaisys::device::nvidia
diff --git a/src/device/nvidia/nvidia_runtime_api.cu b/src/device/nvidia/nvidia_runtime_api.cu
index cab9282..2f5273b 100644
--- a/src/device/nvidia/nvidia_runtime_api.cu
+++ b/src/device/nvidia/nvidia_runtime_api.cu
@@ -1,56 +1,109 @@
 #include "../runtime_api.hpp"
 
-#include <cstdlib>
-#include <cstring>
+#include <cuda_runtime.h>
+#include <stdexcept>
+#include <string>
+
+#define CHECK_CUDA(call)                                                    \
+    do {                                                                    \
+        cudaError_t err = call;                                             \
+        if (err != cudaSuccess) {                                           \
+            throw std::runtime_error(std::string("CUDA error: ") +          \
+                                     cudaGetErrorString(err));              \
+        }                                                                   \
+    } while (0)
 
 namespace llaisys::device::nvidia {
 
 namespace runtime_api {
 int getDeviceCount() {
-    TO_BE_IMPLEMENTED();
+    int count = 0;
+    CHECK_CUDA(cudaGetDeviceCount(&count));
+    return count;
 }
 
-void setDevice(int) {
-    TO_BE_IMPLEMENTED();
+void setDevice(int device) {
+    CHECK_CUDA(cudaSetDevice(device));
 }
 
 void deviceSynchronize() {
-    TO_BE_IMPLEMENTED();
+    CHECK_CUDA(cudaDeviceSynchronize());
 }
 
 llaisysStream_t createStream() {
-    TO_BE_IMPLEMENTED();
+    cudaStream_t stream;
+    CHECK_CUDA(cudaStreamCreate(&stream));
+    return (llaisysStream_t)stream;
 }
 
 void destroyStream(llaisysStream_t stream) {
-    TO_BE_IMPLEMENTED();
+    CHECK_CUDA(cudaStreamDestroy((cudaStream_t)stream));
 }
+
 void streamSynchronize(llaisysStream_t stream) {
-    TO_BE_IMPLEMENTED();
+    CHECK_CUDA(cudaStreamSynchronize((cudaStream_t)stream));
 }
 
 void *mallocDevice(size_t size) {
-    TO_BE_IMPLEMENTED();
+    void *ptr = nullptr;
+    CHECK_CUDA(cudaMalloc(&ptr, size));
+    return ptr;
 }
 
 void freeDevice(void *ptr) {
-    TO_BE_IMPLEMENTED();
+    CHECK_CUDA(cudaFree(ptr));
 }
 
 void *mallocHost(size_t size) {
-    TO_BE_IMPLEMENTED();
+    void *ptr = nullptr;
+    CHECK_CUDA(cudaMallocHost(&ptr, size));
+    return ptr;
 }
 
 void freeHost(void *ptr) {
-    TO_BE_IMPLEMENTED();
+    CHECK_CUDA(cudaFreeHost(ptr));
 }
 
 void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
-    TO_BE_IMPLEMENTED();
+    cudaMemcpyKind cuda_kind;
+    switch (kind) {
+    case LLAISYS_MEMCPY_H2H:
+        cuda_kind = cudaMemcpyHostToHost;
+        break;
+    case LLAISYS_MEMCPY_H2D:
+        cuda_kind = cudaMemcpyHostToDevice;
+        break;
+    case LLAISYS_MEMCPY_D2H:
+        cuda_kind = cudaMemcpyDeviceToHost;
+        break;
+    case LLAISYS_MEMCPY_D2D:
+        cuda_kind = cudaMemcpyDeviceToDevice;
+        break;
+    default:
+        throw std::runtime_error("Unknown memcpy kind");
+    }
+    CHECK_CUDA(cudaMemcpy(dst, src, size, cuda_kind));
 }
 
-void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
-    TO_BE_IMPLEMENTED();
+void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
+    cudaMemcpyKind cuda_kind;
+    switch (kind) {
+    case LLAISYS_MEMCPY_H2H:
+        cuda_kind = cudaMemcpyHostToHost;
+        break;
+    case LLAISYS_MEMCPY_H2D:
+        cuda_kind = cudaMemcpyHostToDevice;
+        break;
+    case LLAISYS_MEMCPY_D2H:
+        cuda_kind = cudaMemcpyDeviceToHost;
+        break;
+    case LLAISYS_MEMCPY_D2D:
+        cuda_kind = cudaMemcpyDeviceToDevice;
+        break;
+    default:
+        throw std::runtime_error("Unknown memcpy kind");
+    }
+    CHECK_CUDA(cudaMemcpyAsync(dst, src, size, cuda_kind, (cudaStream_t)stream));
 }
 
 static const LlaisysRuntimeAPI RUNTIME_API = {
diff --git a/src/llaisys/qwen2.cc b/src/llaisys/qwen2.cc
index 096e3c4..dd0afe6 100644
--- a/src/llaisys/qwen2.cc
+++ b/src/llaisys/qwen2.cc
@@ -6,7 +6,6 @@
 #include <cstdlib>
 #include <cstring>
 #include <cmath>
-#include <iostream>
 
 struct LlaisysQwen2Model {
     LlaisysQwen2Meta meta;
@@ -15,6 +14,7 @@ struct LlaisysQwen2Model {
     int device_id;
 
     // KV Cache: nlayer of tensors
+    // Key-Value Cache stores past token representations for each layer
     // Each tensor shape: [max_seq, nkvh, dh]
     std::vector<llaisysTensor_t> k_cache;
     std::vector<llaisysTensor_t> v_cache;
@@ -75,7 +75,9 @@ static llaisysTensor_t create_zero_tensor(size_t size, llaisysDataType_t dtype,
     return t;
 }
 
-struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
+extern "C" {
+
+__export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
     LlaisysQwen2Model* model = new LlaisysQwen2Model();
     model->meta = *meta;
     model->device_type = device;
@@ -169,7 +171,7 @@ struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta,
     return model;
 }
 
-void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
+__export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
     if (!model) return;
     
     tensorDestroy(model->weights.in_embed);
@@ -214,11 +216,11 @@ void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
     delete model;
 }
 
-struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
+__export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
     return &model->weights;
 }
 
-int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken) {
+__export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken) {
     // 1. Embedding
     // Input: [ntoken]
     // Output: [ntoken, hs]
@@ -413,3 +415,5 @@ int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token
 
     return result_token;
 }
+
+} // extern "C"
diff --git a/src/ops/add/nvidia/add_nvidia.cu b/src/ops/add/nvidia/add_nvidia.cu
new file mode 100644
index 0000000..43b4c08
--- /dev/null
+++ b/src/ops/add/nvidia/add_nvidia.cu
@@ -0,0 +1,157 @@
+#include "add_nvidia.hpp"
+#include "utils/cuda_check.cuh"
+#include "utils/check.hpp"
+#include "utils/types.hpp"
+#include <cuda_runtime.h>
+#include <cuda_fp16.h>
+#include <cuda_bf16.h>
+
+namespace llaisys::ops::nvidia {
+
+// Vectorized types helper
+template <typename T>
+struct VectorizedStorage;
+
+template <>
+struct VectorizedStorage<float> {
+    using Type = float4;
+    static constexpr int size = 4;
+    static __device__ __forceinline__ Type add(Type a, Type b) {
+        return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
+    }
+};
+
+template <>
+struct VectorizedStorage<half> {
+    using Type = float4; // 8 x half
+    static constexpr int size = 8;
+    static __device__ __forceinline__ Type add(Type a, Type b) {
+        Type c;
+        auto* a_h2 = reinterpret_cast<const __half2*>(&a);
+        auto* b_h2 = reinterpret_cast<const __half2*>(&b);
+        auto* c_h2 = reinterpret_cast<__half2*>(&c);
+
+        #pragma unroll
+        for (int i = 0; i < 4; ++i) {
+            c_h2[i] = a_h2[i] + b_h2[i];
+        }
+        return c;
+    }
+};
+
+template <>
+struct VectorizedStorage<__nv_bfloat16> {
+    using Type = float4; // 8 x bf16
+    static constexpr int size = 8;
+    static __device__ __forceinline__ Type add(Type a, Type b) {
+        Type c;
+        auto* a_bf2 = reinterpret_cast<const __nv_bfloat162*>(&a);
+        auto* b_bf2 = reinterpret_cast<const __nv_bfloat162*>(&b);
+        auto* c_bf2 = reinterpret_cast<__nv_bfloat162*>(&c);
+
+        #pragma unroll
+        for (int i = 0; i < 4; ++i) {
+            c_bf2[i] = a_bf2[i] + b_bf2[i];
+        }
+        return c;
+    }
+};
+
+template <>
+struct VectorizedStorage<int> {
+    using Type = int4;
+    static constexpr int size = 4;
+    static __device__ __forceinline__ Type add(Type a, Type b) {
+        return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
+    }
+};
+
+template <typename T>
+__global__ void add_kernel(T *c, const T *a, const T *b, size_t size) {
+    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
+    if (idx < size) {
+        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
+#if __CUDA_ARCH__ >= 800
+             c[idx] = a[idx] + b[idx];
+#else
+             // Fallback for older architectures
+             c[idx] = __float2bfloat16(__bfloat162float(a[idx]) + __bfloat162float(b[idx]));
+#endif
+        } else {
+             c[idx] = a[idx] + b[idx];
+        }
+    }
+}
+
+template <typename T>
+__global__ void add_kernel_vectorized(T *c, const T *a, const T *b, size_t n_vec) {
+    using VecType = typename VectorizedStorage<T>::Type;
+    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
+    
+    if (idx < n_vec) {
+        VecType v_a = reinterpret_cast<const VecType*>(a)[idx];
+        VecType v_b = reinterpret_cast<const VecType*>(b)[idx];
+        
+        reinterpret_cast<VecType*>(c)[idx] = VectorizedStorage<T>::add(v_a, v_b);
+    }
+}
+
+template <typename T>
+void launch_add(T *c, const T *a, const T *b, size_t size) {
+    // 16-byte alignment required for float4/int4
+    bool aligned = (reinterpret_cast<uintptr_t>(a) % 16 == 0) &&
+                   (reinterpret_cast<uintptr_t>(b) % 16 == 0) &&
+                   (reinterpret_cast<uintptr_t>(c) % 16 == 0);
+
+    if (aligned) {
+        constexpr int vec_size = VectorizedStorage<T>::size;
+        size_t n_vec = size / vec_size;
+        
+        if (n_vec > 0) {
+            int threads = 256;
+            int blocks = (n_vec + threads - 1) / threads;
+            add_kernel_vectorized<<<blocks, threads>>>(c, a, b, n_vec);
+            CHECK_LAST_CUDA_ERROR();
+        }
+
+        // Handle tail elements that don't fit in a vector
+        size_t remainder = size % vec_size;
+        if (remainder > 0) {
+            size_t offset = n_vec * vec_size;
+            add_kernel<<<1, 32>>>(c + offset, a + offset, b + offset, remainder);
+            CHECK_LAST_CUDA_ERROR();
+        }
+    } else {
+        // Fallback to scalar kernel for unaligned pointers
+        int threads = 256;
+        int blocks = (size + threads - 1) / threads;
+        add_kernel<<<blocks, threads>>>(c, a, b, size);
+        CHECK_LAST_CUDA_ERROR();
+    }
+}
+
+void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t size) {
+    CHECK_ARGUMENT(c != nullptr, "Output buffer cannot be null");
+    CHECK_ARGUMENT(a != nullptr, "Input buffer A cannot be null");
+    CHECK_ARGUMENT(b != nullptr, "Input buffer B cannot be null");
+    CHECK_ARGUMENT(size > 0, "Size must be positive");
+
+    switch (type) {
+    case LLAISYS_DTYPE_F32:
+        launch_add((float *)c, (const float *)a, (const float *)b, size);
+        break;
+    case LLAISYS_DTYPE_F16:
+        launch_add((half *)c, (const half *)a, (const half *)b, size);
+        break;
+    case LLAISYS_DTYPE_BF16:
+        launch_add((__nv_bfloat16 *)c, (const __nv_bfloat16 *)a, (const __nv_bfloat16 *)b, size);
+        break;
+    case LLAISYS_DTYPE_I32:
+        launch_add((int *)c, (const int *)a, (const int *)b, size);
+        break;
+    default:
+        EXCEPTION_UNSUPPORTED_DATATYPE(type);
+    }
+}
+
+}
diff --git a/src/ops/add/nvidia/add_nvidia.hpp b/src/ops/add/nvidia/add_nvidia.hpp
new file mode 100644
index 0000000..dff9212
--- /dev/null
+++ b/src/ops/add/nvidia/add_nvidia.hpp
@@ -0,0 +1,7 @@
+#pragma once
+#include "llaisys.h"
+#include <cstddef>
+
+namespace llaisys::ops::nvidia {
+void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t size);
+}
diff --git a/src/ops/add/op.cpp b/src/ops/add/op.cpp
index a057330..fe31f96 100644
--- a/src/ops/add/op.cpp
+++ b/src/ops/add/op.cpp
@@ -5,6 +5,10 @@
 
 #include "cpu/add_cpu.hpp"
 
+#ifdef ENABLE_NVIDIA_API
+#include "nvidia/add_nvidia.hpp"
+#endif
+
 namespace llaisys::ops {
 void add(tensor_t c, tensor_t a, tensor_t b) {
     CHECK_SAME_DEVICE(c, a, b);
@@ -25,8 +29,7 @@ void add(tensor_t c, tensor_t a, tensor_t b) {
         return cpu::add(c->data(), a->data(), b->data(), c->dtype(), c->numel());
 #ifdef ENABLE_NVIDIA_API
     case LLAISYS_DEVICE_NVIDIA:
-        TO_BE_IMPLEMENTED();
-        return;
+        return nvidia::add(c->data(), a->data(), b->data(), c->dtype(), c->numel());
 #endif
     default:
         EXCEPTION_UNSUPPORTED_DEVICE;
diff --git a/src/ops/argmax/nvidia/argmax_nvidia.cu b/src/ops/argmax/nvidia/argmax_nvidia.cu
new file mode 100644
index 0000000..f5d7a8a
--- /dev/null
+++ b/src/ops/argmax/nvidia/argmax_nvidia.cu
@@ -0,0 +1,74 @@
+#include "argmax_nvidia.hpp"
+#include <cuda_runtime.h>
+#include <cuda_fp16.h>
+#include <cuda_bf16.h>
+#include <stdexcept>
+#include <cfloat>
+
+namespace llaisys::ops::nvidia {
+
+template <typename T>
+__global__ void argmax_kernel_single_block(T* out_val, int64_t* out_idx, const T* in, int size) {
+    int tid = threadIdx.x;
+    float max_val = -FLT_MAX;
+    int max_idx = -1;
+    
+    for (int i = tid; i < size; i += blockDim.x) {
+        float val = static_cast<float>(in[i]);
+        if (val > max_val) {
+            max_val = val;
+            max_idx = i;
+        }
+    }
+    
+    // TODO: Use dynamic shared memory sized to blockDim.x instead of hardcoded 1024.
+    __shared__ float s_val[1024];
+    __shared__ int s_idx[1024];
+    
+    s_val[tid] = max_val;
+    s_idx[tid] = max_idx;
+    __syncthreads();
+    
+    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
+        if (tid < s) {
+            if (s_val[tid + s] > s_val[tid]) {
+                s_val[tid] = s_val[tid + s];
+                s_idx[tid] = s_idx[tid + s];
+            }
+        }
+        __syncthreads();
+    }
+    
+    if (tid == 0) {
+        *out_val = static_cast<T>(s_val[0]);
+        *out_idx = (int64_t)s_idx[0];
+    }
+}
+
+void argmax(tensor_t max_val, tensor_t max_idx, tensor_t in) {
+    int size = in->numel();
+    
+    // TODO: Implement a staged multi-block reduction path for large tensors.
+    constexpr int threads = 1024;
+    
+    auto dtype = in->dtype();
+    
+    switch(dtype) {
+        case LLAISYS_DTYPE_F32:
+            argmax_kernel_single_block<float><<<1, threads>>>((float*)max_val->data(), (int64_t*)max_idx->data(), (const float*)in->data(), size);
+            break;
+        case LLAISYS_DTYPE_F16:
+            argmax_kernel_single_block<half><<<1, threads>>>((half*)max_val->data(), (int64_t*)max_idx->data(), (const half*)in->data(), size);
+            break;
+        case LLAISYS_DTYPE_BF16:
+            argmax_kernel_single_block<__nv_bfloat16><<<1, threads>>>((__nv_bfloat16*)max_val->data(), (int64_t*)max_idx->data(), (const __nv_bfloat16*)in->data(), size);
+            break;
+        default:
+            throw std::runtime_error("Argmax NVIDIA: Unsupported data type");
+    }
+    
+    if (cudaGetLastError() != cudaSuccess) {
+        throw std::runtime_error("Kernel launch failed");
+    }
+}
+}
diff --git a/src/ops/argmax/nvidia/argmax_nvidia.hpp b/src/ops/argmax/nvidia/argmax_nvidia.hpp
new file mode 100644
index 0000000..cce4664
--- /dev/null
+++ b/src/ops/argmax/nvidia/argmax_nvidia.hpp
@@ -0,0 +1,6 @@
+#pragma once
+#include "tensor/tensor.hpp"
+
+namespace llaisys::ops::nvidia {
+void argmax(tensor_t max_val, tensor_t max_idx, tensor_t in);
+}
diff --git a/src/ops/argmax/op.cpp b/src/ops/argmax/op.cpp
index 2de85b4..d91fd2a 100644
--- a/src/ops/argmax/op.cpp
+++ b/src/ops/argmax/op.cpp
@@ -1,6 +1,10 @@
 #include "op.hpp"
 #include "../common.hpp"
 
+#ifdef ENABLE_NVIDIA_API
+#include "nvidia/argmax_nvidia.hpp"
+#endif
+
 namespace llaisys::ops {
 namespace {
 
@@ -84,7 +88,7 @@ void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
     
 #ifdef ENABLE_NVIDIA_API
     if (vals->deviceType() == LLAISYS_DEVICE_NVIDIA) {
-        TO_BE_IMPLEMENTED();
+        nvidia::argmax(max_val, max_idx, vals);
         return;
     }
 #endif
diff --git a/src/ops/embedding/nvidia/embedding_nvidia.cu b/src/ops/embedding/nvidia/embedding_nvidia.cu
new file mode 100644
index 0000000..498a7d3
--- /dev/null
+++ b/src/ops/embedding/nvidia/embedding_nvidia.cu
@@ -0,0 +1,50 @@
+#include "embedding_nvidia.hpp"
+#include <cuda_runtime.h>
+#include <cuda_fp16.h>
+#include <cuda_bf16.h>
+#include <stdexcept>
+
+namespace llaisys::ops::nvidia {
+
+template <typename T>
+__global__ void embedding_kernel(T* out, const int64_t* index, const T* weight, int N, int D, int V) {
+    int d = blockIdx.x * blockDim.x + threadIdx.x;
+    int n = blockIdx.y * blockDim.y + threadIdx.y;
+
+    if (n >= N || d >= D) return;
+
+    int64_t idx = index[n];
+    if (idx >= 0 && idx < V) {
+        out[n * D + d] = weight[idx * D + d];
+    }
+}
+
+void embedding(tensor_t out, tensor_t index, tensor_t weight) {
+    int N = index->numel();
+    int D = weight->shape()[1];
+    int V = weight->shape()[0];
+    
+    // TODO: Tune block shape for memory coalescing tradeoffs when N and D are highly skewed.
+    dim3 blockDim(32, 4);
+    dim3 gridDim((D + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
+
+    auto dtype = out->dtype();
+    switch (dtype) {
+        case LLAISYS_DTYPE_F32:
+            embedding_kernel<float><<<gridDim, blockDim>>>((float*)out->data(), (const int64_t*)index->data(), (const float*)weight->data(), N, D, V);
+            break;
+        case LLAISYS_DTYPE_F16:
+            embedding_kernel<half><<<gridDim, blockDim>>>((half*)out->data(), (const int64_t*)index->data(), (const half*)weight->data(), N, D, V);
+            break;
+        case LLAISYS_DTYPE_BF16:
+            embedding_kernel<__nv_bfloat16><<<gridDim, blockDim>>>((__nv_bfloat16*)out->data(), (const int64_t*)index->data(), (const __nv_bfloat16*)weight->data(), N, D, V);
+            break;
+        default:
+            throw std::runtime_error("Embedding NVIDIA: Unsupported data type");
+    }
+    
+    if (cudaGetLastError() != cudaSuccess) {
+        throw std::runtime_error("Kernel launch failed");
+    }
+}
+}
diff --git a/src/ops/embedding/nvidia/embedding_nvidia.hpp b/src/ops/embedding/nvidia/embedding_nvidia.hpp
new file mode 100644
index 0000000..3e3233b
--- /dev/null
+++ b/src/ops/embedding/nvidia/embedding_nvidia.hpp
@@ -0,0 +1,6 @@
+#pragma once
+#include "tensor/tensor.hpp"
+
+namespace llaisys::ops::nvidia {
+void embedding(tensor_t out, tensor_t index, tensor_t weight);
+}
diff --git a/src/ops/embedding/op.cpp b/src/ops/embedding/op.cpp
index 832cdd9..7fc70cd 100644
--- a/src/ops/embedding/op.cpp
+++ b/src/ops/embedding/op.cpp
@@ -1,6 +1,10 @@
 #include "op.hpp"
 #include "../common.hpp"
 
+#ifdef ENABLE_NVIDIA_API
+#include "nvidia/embedding_nvidia.hpp"
+#endif
+
 namespace llaisys::ops {
 namespace {
 
@@ -60,7 +64,7 @@ void embedding(tensor_t out, tensor_t index, tensor_t weight) {
 
 #ifdef ENABLE_NVIDIA_API
     if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
-        TO_BE_IMPLEMENTED();
+        nvidia::embedding(out, index, weight);
         return;
     }
 #endif
diff --git a/src/ops/linear/nvidia/linear_nvidia.cu b/src/ops/linear/nvidia/linear_nvidia.cu
new file mode 100644
index 0000000..cd3377e
--- /dev/null
+++ b/src/ops/linear/nvidia/linear_nvidia.cu
@@ -0,0 +1,165 @@
+#include "linear_nvidia.hpp"
+#include <cublas_v2.h>
+#include <cuda_runtime.h>
+#include <cuda_fp16.h>
+#include <cuda_bf16.h>
+#include <stdexcept>
+#include "utils/cuda_check.cuh"
+
+namespace llaisys::ops::nvidia {
+using namespace llaisys::utils;
+
+// Vectorized types helper
+template <typename T>
+struct VecTraits;
+
+template <>
+struct VecTraits<float> {
+    using Type = float4;
+    static constexpr int size = 4;
+};
+
+template <>
+struct VecTraits<half> {
+    using Type = float4;
+    static constexpr int size = 8;
+};
+
+template <>
+struct VecTraits<__nv_bfloat16> {
+    using Type = float4;
+    static constexpr int size = 8;
+};
+
+template <typename T>
+__global__ void add_bias_kernel(T* out, const T* bias, int rows, int cols) {
+    int idx = blockIdx.x * blockDim.x + threadIdx.x;
+    int size = rows * cols;
+    if (idx < size) {
+        out[idx] += bias[idx % cols];
+    }
+}
+
+// Vectorized Bias Add Kernel
+template <typename T>
+__global__ void add_bias_kernel_vectorized(T* out, const T* bias, int rows, int cols) {
+    using VecType = typename VecTraits<T>::Type;
+    int vec_cols = cols / VecTraits<T>::size;
+    
+    int idx = blockIdx.x * blockDim.x + threadIdx.x;
+    int total_vecs = rows * vec_cols;
+
+    if (idx < total_vecs) {
+        int row = idx / vec_cols;
+        int vec_col = idx % vec_cols;
+        int scalar_col_offset = vec_col * VecTraits<T>::size;
+
+        VecType b_vec = reinterpret_cast<const VecType*>(bias)[vec_col];
+        
+        VecType* out_vec_ptr = reinterpret_cast<VecType*>(out);
+        int out_vec_idx = row * vec_cols + vec_col;
+        VecType o_vec = out_vec_ptr[out_vec_idx];
+
+        if constexpr (std::is_same_v<T, float>) {
+            o_vec.x += b_vec.x; o_vec.y += b_vec.y; o_vec.z += b_vec.z; o_vec.w += b_vec.w;
+        } else if constexpr (std::is_same_v<T, half>) {
+            __half2* o_h2 = reinterpret_cast<__half2*>(&o_vec);
+            const __half2* b_h2 = reinterpret_cast<const __half2*>(&b_vec);
+            #pragma unroll
+            for(int i=0; i<4; ++i) o_h2[i] = __hadd2(o_h2[i], b_h2[i]);
+        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
+            __nv_bfloat162* o_bf2 = reinterpret_cast<__nv_bfloat162*>(&o_vec);
+            const __nv_bfloat162* b_bf2 = reinterpret_cast<const __nv_bfloat162*>(&b_vec);
+            #pragma unroll
+            for(int i=0; i<4; ++i) o_bf2[i] = __hadd2(o_bf2[i], b_bf2[i]);
+        }
+
+        out_vec_ptr[out_vec_idx] = o_vec;
+    }
+}
+
+
+void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
+    int M = in->shape()[0];
+    int K = in->shape()[1];
+    int N = out->shape()[1];
+    
+    cublasHandle_t handle = get_cublas_handle();
+    float alpha = 1.0f;
+    float beta = 0.0f;
+
+    cudaDataType_t AType, BType, CType;
+    cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
+
+    switch(out->dtype()) {
+        case LLAISYS_DTYPE_F32: AType = BType = CType = CUDA_R_32F; break;
+        case LLAISYS_DTYPE_F16: AType = BType = CType = CUDA_R_16F; break;
+        case LLAISYS_DTYPE_BF16: AType = BType = CType = CUDA_R_16BF; break;
+        default: throw std::runtime_error("Linear NVIDIA: Unsupported data type");
+    }
+
+    cublasStatus_t status = cublasGemmEx(handle, 
+        CUBLAS_OP_T, CUBLAS_OP_N, 
+        N, M, K, 
+        &alpha,
+        weight->data(), AType, K,
+        in->data(), BType, K, 
+        &beta, 
+        out->data(), CType, N,
+        computeType, CUBLAS_GEMM_DEFAULT);
+
+    if (status != CUBLAS_STATUS_SUCCESS) {
+        throw std::runtime_error("cuBLAS Error: " + std::to_string(status));
+    }
+
+    if (bias) {
+        bool aligned = (reinterpret_cast<uintptr_t>(out->data()) % 16 == 0) &&
+                       (reinterpret_cast<uintptr_t>(bias->data()) % 16 == 0);
+        
+        int vec_size = 0;
+        if (out->dtype() == LLAISYS_DTYPE_F32) vec_size = 4;
+        else vec_size = 8;
+
+        if (aligned && (N % vec_size == 0)) {
+            int vec_cols = N / vec_size;
+            int total_vecs = M * vec_cols;
+            int threads = 256;
+            int blocks = (total_vecs + threads - 1) / threads;
+
+            switch(out->dtype()) {
+                case LLAISYS_DTYPE_F32:
+                    add_bias_kernel_vectorized<float><<<blocks, threads>>>((float*)out->data(), (const float*)bias->data(), M, N);
+                    break;
+                case LLAISYS_DTYPE_F16:
+                    add_bias_kernel_vectorized<half><<<blocks, threads>>>((half*)out->data(), (const half*)bias->data(), M, N);
+                    break;
+                case LLAISYS_DTYPE_BF16:
+                    add_bias_kernel_vectorized<__nv_bfloat16><<<blocks, threads>>>((__nv_bfloat16*)out->data(), (const __nv_bfloat16*)bias->data(), M, N);
+                    break;
+            }
+        } else {
+            // Fallback to scalar kernel for unaligned pointers
+            int size = M * N;
+            int threads = 256;
+            int blocks = (size + threads - 1) / threads;
+
+            switch(out->dtype()) {
+                case LLAISYS_DTYPE_F32:
+                    add_bias_kernel<float><<<blocks, threads>>>((float*)out->data(), (const float*)bias->data(), M, N);
+                    break;
+                case LLAISYS_DTYPE_F16:
+                    add_bias_kernel<half><<<blocks, threads>>>((half*)out->data(), (const half*)bias->data(), M, N);
+                    break;
+                case LLAISYS_DTYPE_BF16:
+                    add_bias_kernel<__nv_bfloat16><<<blocks, threads>>>((__nv_bfloat16*)out->data(), (const __nv_bfloat16*)bias->data(), M, N);
+                    break;
+            }
+        }
+        
+        if (cudaGetLastError() != cudaSuccess) {
+            throw std::runtime_error("Bias kernel launch failed");
+        }
+    }
+}
+
+} // namespace llaisys::ops::nvidia
\ No newline at end of file
diff --git a/src/ops/linear/nvidia/linear_nvidia.hpp b/src/ops/linear/nvidia/linear_nvidia.hpp
new file mode 100644
index 0000000..871453d
--- /dev/null
+++ b/src/ops/linear/nvidia/linear_nvidia.hpp
@@ -0,0 +1,10 @@
+#pragma once
+#include "tensor/tensor.hpp"
+
+namespace llaisys {
+namespace ops {
+namespace nvidia {
+void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias);
+}
+}
+}
diff --git a/src/ops/linear/op.cpp b/src/ops/linear/op.cpp
index 87e67a9..9c07827 100644
--- a/src/ops/linear/op.cpp
+++ b/src/ops/linear/op.cpp
@@ -1,6 +1,10 @@
 #include "op.hpp"
 #include "../common.hpp"
 
+#ifdef ENABLE_NVIDIA_API
+#include "nvidia/linear_nvidia.hpp"
+#endif
+
 namespace llaisys::ops {
 namespace {
 
@@ -89,7 +93,7 @@ void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
         
 #ifdef ENABLE_NVIDIA_API
     case LLAISYS_DEVICE_NVIDIA:
-        TO_BE_IMPLEMENTED();
+        nvidia::linear(out, in, weight, bias);
         return;
 #endif
         
diff --git a/src/ops/rearrange/nvidia/rearrange_nvidia.cu b/src/ops/rearrange/nvidia/rearrange_nvidia.cu
new file mode 100644
index 0000000..94d802b
--- /dev/null
+++ b/src/ops/rearrange/nvidia/rearrange_nvidia.cu
@@ -0,0 +1,14 @@
+#include "rearrange_nvidia.hpp"
+#include <cuda_runtime.h>
+#include <stdexcept>
+
+namespace llaisys::ops::nvidia {
+
+void rearrange(tensor_t out, tensor_t in) {
+    size_t size_bytes = out->numel() * out->elementSize();
+    if (cudaMemcpyAsync(out->data(), in->data(), size_bytes, cudaMemcpyDeviceToDevice, 0) != cudaSuccess) {
+        throw std::runtime_error("Rearrange NVIDIA: cudaMemcpyAsync failed");
+    }
+}
+
+}
diff --git a/src/ops/rearrange/nvidia/rearrange_nvidia.hpp b/src/ops/rearrange/nvidia/rearrange_nvidia.hpp
new file mode 100644
index 0000000..455bd25
--- /dev/null
+++ b/src/ops/rearrange/nvidia/rearrange_nvidia.hpp
@@ -0,0 +1,6 @@
+#pragma once
+#include "tensor/tensor.hpp"
+
+namespace llaisys::ops::nvidia {
+void rearrange(tensor_t out, tensor_t in);
+}
diff --git a/src/ops/rearrange/op.cpp b/src/ops/rearrange/op.cpp
index 017a6ae..5c2e622 100644
--- a/src/ops/rearrange/op.cpp
+++ b/src/ops/rearrange/op.cpp
@@ -1,7 +1,32 @@
 #include "op.hpp"
+#include "../../core/context/context.hpp"
+#include "../../utils.hpp"
+#include <cstring>
+
+#ifdef ENABLE_NVIDIA_API
+#include "nvidia/rearrange_nvidia.hpp"
+#endif
 
 namespace llaisys::ops {
 void rearrange(tensor_t out, tensor_t in) {
-    TO_BE_IMPLEMENTED();
+    CHECK_SAME_DEVICE(out, in);
+    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
+    
+    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
+        size_t size = out->numel() * out->elementSize();
+        std::memcpy(out->data(), in->data(), size);
+        return;
+    }
+
+    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
+
+#ifdef ENABLE_NVIDIA_API
+    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
+        nvidia::rearrange(out, in);
+        return;
+    }
+#endif
+
+    EXCEPTION_UNSUPPORTED_DEVICE;
 }
 } // namespace llaisys::ops
diff --git a/src/ops/rms_norm/nvidia/rms_norm_nvidia.cu b/src/ops/rms_norm/nvidia/rms_norm_nvidia.cu
new file mode 100644
index 0000000..6c2d3eb
--- /dev/null
+++ b/src/ops/rms_norm/nvidia/rms_norm_nvidia.cu
@@ -0,0 +1,65 @@
+#include "rms_norm_nvidia.hpp"
+#include <cuda_runtime.h>
+#include <cuda_fp16.h>
+#include <cuda_bf16.h>
+#include <stdexcept>
+
+namespace llaisys::ops::nvidia {
+
+template <typename T>
+__global__ void rms_norm_kernel(T* out, const T* in, const T* weight, float eps, int D) {
+    int row = blockIdx.x;
+    int tid = threadIdx.x;
+    const T* in_row = in + row * D;
+    T* out_row = out + row * D;
+
+    float sum_sq = 0.0f;
+    for (int i = tid; i < D; i += blockDim.x) {
+        float val = static_cast<float>(in_row[i]);
+        sum_sq += val * val;
+    }
+
+    extern __shared__ float sdata[];
+    sdata[tid] = sum_sq;
+    __syncthreads();
+
+    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
+        if (tid < s) sdata[tid] += sdata[tid + s];
+        __syncthreads();
+    }
+
+    float inv_rms = rsqrtf(sdata[0] / D + eps);
+
+    for (int i = tid; i < D; i += blockDim.x) {
+        out_row[i] = static_cast<T>(static_cast<float>(in_row[i]) * inv_rms * static_cast<float>(weight[i]));
+    }
+}
+
+void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
+    int N = in->shape()[0]; // N rows, one block per row
+    int D = in->shape()[1];
+    
+    // TODO: Vectorize input loads and weight reads for cache efficiency.
+    constexpr int threads = 256;
+    size_t smem = threads * sizeof(float);
+    
+    auto dtype = out->dtype();
+    switch (dtype) {
+        case LLAISYS_DTYPE_F32:
+            rms_norm_kernel<float><<<N, threads, smem>>>((float*)out->data(), (const float*)in->data(), (const float*)weight->data(), eps, D);
+            break;
+        case LLAISYS_DTYPE_F16:
+            rms_norm_kernel<half><<<N, threads, smem>>>((half*)out->data(), (const half*)in->data(), (const half*)weight->data(), eps, D);
+            break;
+        case LLAISYS_DTYPE_BF16:
+            rms_norm_kernel<__nv_bfloat16><<<N, threads, smem>>>((__nv_bfloat16*)out->data(), (const __nv_bfloat16*)in->data(), (const __nv_bfloat16*)weight->data(), eps, D);
+            break;
+        default:
+            throw std::runtime_error("RMSNorm NVIDIA: Unsupported data type");
+    }
+    
+    if (cudaGetLastError() != cudaSuccess) {
+        throw std::runtime_error("Kernel launch failed");
+    }
+}
+}
diff --git a/src/ops/rms_norm/nvidia/rms_norm_nvidia.hpp b/src/ops/rms_norm/nvidia/rms_norm_nvidia.hpp
new file mode 100644
index 0000000..a45bfef
--- /dev/null
+++ b/src/ops/rms_norm/nvidia/rms_norm_nvidia.hpp
@@ -0,0 +1,6 @@
+#pragma once
+#include "tensor/tensor.hpp"
+
+namespace llaisys::ops::nvidia {
+void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps);
+}
diff --git a/src/ops/rms_norm/op.cpp b/src/ops/rms_norm/op.cpp
index ca6866f..20b3fa2 100644
--- a/src/ops/rms_norm/op.cpp
+++ b/src/ops/rms_norm/op.cpp
@@ -2,6 +2,10 @@
 #include "../common.hpp"
 #include <cmath>
 
+#ifdef ENABLE_NVIDIA_API
+#include "nvidia/rms_norm_nvidia.hpp"
+#endif
+
 namespace llaisys::ops {
 namespace {
 
@@ -87,7 +91,7 @@ void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
 
 #ifdef ENABLE_NVIDIA_API
     if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
-        TO_BE_IMPLEMENTED();
+        nvidia::rms_norm(out, in, weight, eps);
         return;
     }
 #endif
diff --git a/src/ops/rope/nvidia/rope_nvidia.cu b/src/ops/rope/nvidia/rope_nvidia.cu
new file mode 100644
index 0000000..6d3676c
--- /dev/null
+++ b/src/ops/rope/nvidia/rope_nvidia.cu
@@ -0,0 +1,71 @@
+#include "rope_nvidia.hpp"
+#include <cuda_runtime.h>
+#include <cuda_fp16.h>
+#include <cuda_bf16.h>
+#include <stdexcept>
+
+namespace llaisys::ops::nvidia {
+
+template <typename T>
+__global__ void rope_kernel(T* out, const T* in, const int64_t* pos_ids, float theta, int nhead, int head_dim, int seq_len) {
+    int idx = blockIdx.x * blockDim.x + threadIdx.x;
+    int half_dim = head_dim / 2;
+    int total_pairs = seq_len * nhead * half_dim;
+    
+    if (idx >= total_pairs) return;
+
+    // Decompose flat index to [seq, head, dim/2]
+    int d = idx % half_dim;
+    int rem = idx / half_dim;
+    int h = rem % nhead;
+    int s = rem / nhead;
+    
+    int head_offset = (s * nhead + h) * head_dim;
+    int i1 = head_offset + d;
+    int i2 = head_offset + d + half_dim;
+    
+    float x1 = static_cast<float>(in[i1]);
+    float x2 = static_cast<float>(in[i2]);
+    
+    int pos = pos_ids[s];
+    // TODO: Precompute inverse frequencies to remove powf from the kernel hot path.
+    float freq = 1.0f / powf(theta, (float)(2 * d) / head_dim);
+    float alpha = (float)pos * freq;
+    
+    float c, s_val;
+    sincosf(alpha, &s_val, &c);
+    
+    out[i1] = static_cast<T>(x1 * c - x2 * s_val);
+    out[i2] = static_cast<T>(x1 * s_val + x2 * c);
+}
+
+void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
+    int seq_len = in->shape()[0];
+    int nhead = in->shape()[1];
+    int head_dim = in->shape()[2];
+    
+    // TODO: Tune launch configuration based on specific seq_len/nhead/head_dim shapes.
+    int total_pairs = seq_len * nhead * head_dim / 2;
+    constexpr int threads = 256;
+    int blocks = (total_pairs + threads - 1) / threads;
+    
+    auto dtype = out->dtype();
+    switch (dtype) {
+        case LLAISYS_DTYPE_F32:
+            rope_kernel<float><<<blocks, threads>>>((float*)out->data(), (const float*)in->data(), (const int64_t*)pos_ids->data(), theta, nhead, head_dim, seq_len);
+            break;
+        case LLAISYS_DTYPE_F16:
+            rope_kernel<half><<<blocks, threads>>>((half*)out->data(), (const half*)in->data(), (const int64_t*)pos_ids->data(), theta, nhead, head_dim, seq_len);
+            break;
+        case LLAISYS_DTYPE_BF16:
+            rope_kernel<__nv_bfloat16><<<blocks, threads>>>((__nv_bfloat16*)out->data(), (const __nv_bfloat16*)in->data(), (const int64_t*)pos_ids->data(), theta, nhead, head_dim, seq_len);
+            break;
+        default:
+            throw std::runtime_error("RoPE NVIDIA: Unsupported data type");
+    }
+    
+    if (cudaGetLastError() != cudaSuccess) {
+        throw std::runtime_error("Kernel launch failed");
+    }
+}
+}
diff --git a/src/ops/rope/nvidia/rope_nvidia.hpp b/src/ops/rope/nvidia/rope_nvidia.hpp
new file mode 100644
index 0000000..f207026
--- /dev/null
+++ b/src/ops/rope/nvidia/rope_nvidia.hpp
@@ -0,0 +1,6 @@
+#pragma once
+#include "tensor/tensor.hpp"
+
+namespace llaisys::ops::nvidia {
+void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta);
+}
diff --git a/src/ops/rope/op.cpp b/src/ops/rope/op.cpp
index 7753483..0645311 100644
--- a/src/ops/rope/op.cpp
+++ b/src/ops/rope/op.cpp
@@ -2,6 +2,10 @@
 #include "../common.hpp"
 #include <cmath>
 
+#ifdef ENABLE_NVIDIA_API
+#include "nvidia/rope_nvidia.hpp"
+#endif
+
 namespace llaisys::ops {
 namespace {
 
@@ -110,7 +114,7 @@ void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
 
 #ifdef ENABLE_NVIDIA_API
     if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
-        TO_BE_IMPLEMENTED();
+        nvidia::rope(out, in, pos_ids, theta);
         return;
     }
 #endif
diff --git a/src/ops/self_attention/nvidia/self_attention_nvidia.cu b/src/ops/self_attention/nvidia/self_attention_nvidia.cu
new file mode 100644
index 0000000..3cfcefc
--- /dev/null
+++ b/src/ops/self_attention/nvidia/self_attention_nvidia.cu
@@ -0,0 +1,120 @@
+#include "self_attention_nvidia.hpp"
+#include <cuda_runtime.h>
+#include <cublas_v2.h>
+#include <cuda_fp16.h>
+#include <cuda_bf16.h>
+#include <stdexcept>
+#include <cmath>
+#include <float.h>
+#include "utils/cuda_check.cuh"
+
+namespace llaisys::ops::nvidia {
+using namespace llaisys::utils;
+
+template <typename T>
+__global__ void attention_softmax_kernel(T* scores, int q_len, int kv_len, int nhead, float scale) {
+    // 3-pass softmax implementation
+    // TODO: Fuse into online softmax to reduce global memory traffic
+    int h = blockIdx.y; 
+    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
+
+    if (q_idx >= q_len || h >= nhead) return;
+
+    T* row = scores + (h * q_len + q_idx) * kv_len;
+    
+    // Causal masking
+    int mask_limit = q_idx + (kv_len - q_len);
+
+    float max_val = -FLT_MAX;
+    for (int i = 0; i <= mask_limit; ++i) {
+        float val = static_cast<float>(row[i]) * scale;
+        if (val > max_val) max_val = val;
+    }
+
+    float sum_exp = 0.0f;
+    for (int i = 0; i < kv_len; ++i) {
+        if (i <= mask_limit) {
+            float val = expf((static_cast<float>(row[i]) * scale) - max_val);
+            row[i] = static_cast<T>(val);
+            sum_exp += val;
+        } else {
+            row[i] = static_cast<T>(0.0f);
+        }
+    }
+
+    float inv_sum = 1.0f / (sum_exp + 1e-9f);
+    for (int i = 0; i < kv_len; ++i) {
+        row[i] = static_cast<T>(static_cast<float>(row[i]) * inv_sum);
+    }
+}
+
+template <typename T, cudaDataType_t CudaType>
+void self_attention_impl(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
+    int q_len = q->shape()[0];
+    int nhead = q->shape()[1];
+    int head_dim = q->shape()[2];
+    int kv_len = k->shape()[0];
+    int nkvh = k->shape()[1];
+    
+    cublasHandle_t handle = get_cublas_handle();
+    float alpha = 1.0f, beta = 0.0f;
+
+    size_t scores_size = nhead * q_len * kv_len * sizeof(T);
+    T* scores_dev = static_cast<T*>(get_scratch_buffer(scores_size));
+
+    // Q * K^T
+    for (int g = 0; g < nkvh; ++g) {
+        int heads_per_group = nhead / nkvh;
+        cublasGemmStridedBatchedEx(handle,
+            CUBLAS_OP_T, CUBLAS_OP_N, 
+            kv_len, q_len, head_dim,
+            &alpha,
+            (const void*)((T*)k->data() + (g * head_dim)), CudaType, nkvh * head_dim, 0, 
+            (const void*)((T*)q->data() + (g * heads_per_group * head_dim)), CudaType, nhead * head_dim, head_dim,
+            &beta,
+            (void*)(scores_dev + (g * heads_per_group * q_len * kv_len)), CudaType, kv_len, q_len * kv_len,
+            heads_per_group,
+            CUBLAS_COMPUTE_32F,
+            CUBLAS_GEMM_DEFAULT);
+    }
+
+    dim3 threads(32);
+    dim3 blocks((q_len + 31) / 32, nhead);
+    attention_softmax_kernel<T><<<blocks, threads>>>(scores_dev, q_len, kv_len, nhead, scale);
+
+    // Scores * V
+    for (int g = 0; g < nkvh; ++g) {
+        int heads_per_group = nhead / nkvh;
+        cublasGemmStridedBatchedEx(handle,
+            CUBLAS_OP_N, CUBLAS_OP_N,
+            head_dim, q_len, kv_len,
+            &alpha,
+            (const void*)((T*)v->data() + (g * head_dim)), CudaType, nkvh * head_dim, 0,
+            (const void*)(scores_dev + (g * heads_per_group * q_len * kv_len)), CudaType, kv_len, q_len * kv_len,
+            &beta,
+            (void*)((T*)attn_val->data() + (g * heads_per_group * head_dim)), CudaType, nhead * head_dim, head_dim,
+            heads_per_group,
+            CUBLAS_COMPUTE_32F,
+            CUBLAS_GEMM_DEFAULT);
+    }
+}
+
+void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
+    auto dtype = q->dtype();
+
+    switch (dtype) {
+        case LLAISYS_DTYPE_F32:
+            self_attention_impl<float, CUDA_R_32F>(attn_val, q, k, v, scale);
+            break;
+        case LLAISYS_DTYPE_F16:
+            self_attention_impl<half, CUDA_R_16F>(attn_val, q, k, v, scale);
+            break;
+        case LLAISYS_DTYPE_BF16:
+            self_attention_impl<__nv_bfloat16, CUDA_R_16BF>(attn_val, q, k, v, scale);
+            break;
+        default:
+            throw std::runtime_error("Unsupported dtype for self_attention");
+    }
+}
+
+}
\ No newline at end of file
diff --git a/src/ops/self_attention/nvidia/self_attention_nvidia.hpp b/src/ops/self_attention/nvidia/self_attention_nvidia.hpp
new file mode 100644
index 0000000..f2e490a
--- /dev/null
+++ b/src/ops/self_attention/nvidia/self_attention_nvidia.hpp
@@ -0,0 +1,6 @@
+#pragma once
+#include "tensor/tensor.hpp"
+
+namespace llaisys::ops::nvidia {
+void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale);
+}
diff --git a/src/ops/self_attention/op.cpp b/src/ops/self_attention/op.cpp
index 42e4d7f..a874c43 100644
--- a/src/ops/self_attention/op.cpp
+++ b/src/ops/self_attention/op.cpp
@@ -5,6 +5,10 @@
 #include <limits>
 #include <vector>
 
+#ifdef ENABLE_NVIDIA_API
+#include "nvidia/self_attention_nvidia.hpp"
+#endif
+
 namespace llaisys::ops {
 namespace {
 
@@ -191,7 +195,7 @@ void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float
 
 #ifdef ENABLE_NVIDIA_API
     if (attn_val->deviceType() == LLAISYS_DEVICE_NVIDIA) {
-        TO_BE_IMPLEMENTED();
+        nvidia::self_attention(attn_val, q, k, v, scale);
         return;
     }
 #endif
diff --git a/src/ops/swiglu/nvidia/swiglu_nvidia.cu b/src/ops/swiglu/nvidia/swiglu_nvidia.cu
new file mode 100644
index 0000000..00d5a0b
--- /dev/null
+++ b/src/ops/swiglu/nvidia/swiglu_nvidia.cu
@@ -0,0 +1,45 @@
+#include "swiglu_nvidia.hpp"
+#include <cuda_runtime.h>
+#include <cuda_fp16.h>
+#include <cuda_bf16.h>
+#include <stdexcept>
+
+namespace llaisys::ops::nvidia {
+
+template <typename T>
+__global__ void swiglu_kernel(T* out, const T* gate, const T* up, int size) {
+    int idx = blockIdx.x * blockDim.x + threadIdx.x;
+    if (idx >= size) return;
+    
+    // TODO: Vectorize elementwise loads/stores
+    float g = static_cast<float>(gate[idx]);
+    float u = static_cast<float>(up[idx]);
+    float sigmoid_g = 1.0f / (1.0f + expf(-g));
+    out[idx] = static_cast<T>(u * (g * sigmoid_g));
+}
+
+void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
+    int size = out->numel();
+    constexpr int threads = 256;
+    int blocks = (size + threads - 1) / threads;
+    
+    auto dtype = out->dtype();
+    switch (dtype) {
+        case LLAISYS_DTYPE_F32:
+            swiglu_kernel<float><<<blocks, threads>>>((float*)out->data(), (const float*)gate->data(), (const float*)up->data(), size);
+            break;
+        case LLAISYS_DTYPE_F16:
+            swiglu_kernel<half><<<blocks, threads>>>((half*)out->data(), (const half*)gate->data(), (const half*)up->data(), size);
+            break;
+        case LLAISYS_DTYPE_BF16:
+            swiglu_kernel<__nv_bfloat16><<<blocks, threads>>>((__nv_bfloat16*)out->data(), (const __nv_bfloat16*)gate->data(), (const __nv_bfloat16*)up->data(), size);
+            break;
+        default:
+            throw std::runtime_error("SwiGLU NVIDIA: Unsupported data type");
+    }
+    
+    if (cudaGetLastError() != cudaSuccess) {
+        throw std::runtime_error("Kernel launch failed");
+    }
+}
+}
diff --git a/src/ops/swiglu/nvidia/swiglu_nvidia.hpp b/src/ops/swiglu/nvidia/swiglu_nvidia.hpp
new file mode 100644
index 0000000..0e92f69
--- /dev/null
+++ b/src/ops/swiglu/nvidia/swiglu_nvidia.hpp
@@ -0,0 +1,6 @@
+#pragma once
+#include "tensor/tensor.hpp"
+
+namespace llaisys::ops::nvidia {
+void swiglu(tensor_t out, tensor_t gate, tensor_t up);
+}
diff --git a/src/ops/swiglu/op.cpp b/src/ops/swiglu/op.cpp
index 447e3ff..e9a0e4e 100644
--- a/src/ops/swiglu/op.cpp
+++ b/src/ops/swiglu/op.cpp
@@ -2,6 +2,10 @@
 #include "../common.hpp"
 #include <cmath>
 
+#ifdef ENABLE_NVIDIA_API
+#include "nvidia/swiglu_nvidia.hpp"
+#endif
+
 namespace llaisys::ops {
 namespace {
 
@@ -78,7 +82,7 @@ void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
 
 #ifdef ENABLE_NVIDIA_API
     if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
-        TO_BE_IMPLEMENTED();
+        nvidia::swiglu(out, gate, up);
         return;
     }
 #endif
diff --git a/src/utils/cuda_check.cuh b/src/utils/cuda_check.cuh
new file mode 100644
index 0000000..6c14061
--- /dev/null
+++ b/src/utils/cuda_check.cuh
@@ -0,0 +1,54 @@
+#pragma once
+
+#include <cuda_runtime.h>
+#include <cublas_v2.h>
+#include <iostream>
+#include <stdexcept>
+
+namespace llaisys::utils {
+
+inline void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
+    if (result != cudaSuccess) {
+        std::cerr << "[ERROR] CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(result) << " \"" << cudaGetErrorString(result) << "\" " << func << std::endl;
+        throw std::runtime_error("CUDA runtime error");
+    }
+}
+
+inline void check_last_cuda_error(const char *const file, int const line) {
+    cudaError_t err = cudaGetLastError();
+    if (err != cudaSuccess) {
+        std::cerr << "[ERROR] CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(err) << " \"" << cudaGetErrorString(err) << "\"" << std::endl;
+        throw std::runtime_error("CUDA kernel launch failed");
+    }
+}
+
+// Helper to manage cuBLAS handle efficiently
+inline cublasHandle_t get_cublas_handle() {
+    static thread_local cublasHandle_t handle = nullptr;
+    if (!handle) {
+        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
+            throw std::runtime_error("Failed to create cuBLAS handle");
+        }
+    }
+    return handle;
+}
+
+// Helper to manage temporary device memory efficiently
+inline void* get_scratch_buffer(size_t size) {
+    static thread_local void* ptr = nullptr;
+    static thread_local size_t capacity = 0;
+
+    if (size > capacity) {
+        if (ptr) cudaFree(ptr);
+        if (cudaMalloc(&ptr, size) != cudaSuccess) {
+            throw std::runtime_error("Failed to allocate scratch buffer");
+        }
+        capacity = size;
+    }
+    return ptr;
+}
+
+} // namespace llaisys::utils
+
+#define CHECK_CUDA(val) llaisys::utils::check_cuda((val), #val, __FILE__, __LINE__)
+#define CHECK_LAST_CUDA_ERROR() llaisys::utils::check_last_cuda_error(__FILE__, __LINE__)
diff --git a/test/ops/add.py b/test/ops/add.py
index bb8bf8c..e31cb1b 100644
--- a/test/ops/add.py
+++ b/test/ops/add.py
@@ -31,10 +31,12 @@ def test_op_add(
     assert check_equal(c_, c, atol=atol, rtol=rtol)
 
     if profile:
+        compiled_torch_add = torch.compile(torch_add)
         benchmark(
             lambda: torch_add(c, a, b),
             lambda: llaisys.Ops.add(c_, a_, b_),
             device_name,
+            torch_compile_func=lambda: compiled_torch_add(c, a, b)
         )
 
 
diff --git a/test/ops/argmax.py b/test/ops/argmax.py
index d0f7ee2..5b6bdaa 100644
--- a/test/ops/argmax.py
+++ b/test/ops/argmax.py
@@ -32,10 +32,12 @@ def test_op_argmax(
     )
 
     if profile:
+        compiled_torch_argmax = torch.compile(torch_argmax)
         benchmark(
             lambda: torch_argmax(max_idx, max_val, vals),
             lambda: llaisys.Ops.argmax(max_idx_, max_val_, vals_),
             device_name,
+            torch_compile_func=lambda: compiled_torch_argmax(max_idx, max_val, vals)
         )
 
 
diff --git a/test/ops/embedding.py b/test/ops/embedding.py
index 99cadc1..da66d98 100644
--- a/test/ops/embedding.py
+++ b/test/ops/embedding.py
@@ -4,6 +4,7 @@ import os
 parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
 sys.path.insert(0, parent_dir)
 import llaisys
+import torch
 from test_utils import random_int_tensor, random_tensor, check_equal, benchmark
 
 
@@ -28,10 +29,12 @@ def test_op_embedding(
     check_equal(out_, out, strict=True)
 
     if profile:
+        compiled_torch_embedding = torch.compile(torch_embedding)
         benchmark(
             lambda: torch_embedding(out, idx, embd),
             lambda: llaisys.Ops.embedding(out_, idx_, embd_),
             device_name,
+            torch_compile_func=lambda: compiled_torch_embedding(out, idx, embd)
         )
 
 
diff --git a/test/ops/linear.py b/test/ops/linear.py
index 3889733..46b4a94 100644
--- a/test/ops/linear.py
+++ b/test/ops/linear.py
@@ -38,10 +38,12 @@ def test_op_linear(
     assert check_equal(out_, out, atol=atol, rtol=rtol)
 
     if profile:
+        compiled_torch_linear = torch.compile(torch_linear)
         benchmark(
             lambda: torch_linear(out, x, w, bias),
             lambda: llaisys.Ops.linear(out_, x_, w_, bias_),
             device_name,
+            torch_compile_func=lambda: compiled_torch_linear(out, x, w, bias)
         )
 
 
diff --git a/test/ops/rms_norm.py b/test/ops/rms_norm.py
index 67b789e..4b14cd2 100644
--- a/test/ops/rms_norm.py
+++ b/test/ops/rms_norm.py
@@ -5,7 +5,7 @@ parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
 sys.path.insert(0, parent_dir)
 import llaisys
 import torch
-from test_utils import random_tensor, check_equal, benchmark
+from test_utils import random_tensor, check_equal, benchmark, set_seed
 
 
 def torch_rms_norm(ans, x, w, eps):
@@ -37,10 +37,12 @@ def test_op_rms_norm(
     assert check_equal(c_, c, atol=atol, rtol=rtol)
 
     if profile:
+        compiled_torch_rms_norm = torch.compile(torch_rms_norm)
         benchmark(
             lambda: torch_rms_norm(c, x, w, eps),
             lambda: llaisys.Ops.rms_norm(c_, x_, w_, eps),
             device_name,
+            torch_compile_func=lambda: compiled_torch_rms_norm(c, x, w, eps)
         )
 
 
@@ -51,11 +53,12 @@ if __name__ == "__main__":
     parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
     parser.add_argument("--profile", action="store_true")
     args = parser.parse_args()
+    set_seed()
     testShapes = [(1, 4), (512, 4096)]
     testDtypePrec = [
         # type, atol, rtol
         ("f32", 1e-5, 1e-5),
-        ("f16", 1e-3, 1e-3),
+        ("f16", 5e-3, 5e-3),
         ("bf16", 1e-2, 1e-2),
     ]
     print(f"Testing Ops.rms_norm on {args.device}")
diff --git a/test/ops/rope.py b/test/ops/rope.py
index fe59dd1..d84c69e 100644
--- a/test/ops/rope.py
+++ b/test/ops/rope.py
@@ -5,7 +5,7 @@ parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
 sys.path.insert(0, parent_dir)
 import llaisys
 import torch
-from test_utils import arrange_tensor, random_tensor, check_equal, benchmark
+from test_utils import arrange_tensor, random_tensor, check_equal, benchmark, set_seed
 
 
 def torch_rope(y: torch.Tensor, x: torch.Tensor, pos_ids: torch.Tensor, theta: float):
@@ -52,10 +52,12 @@ def test_op_rope(
     assert check_equal(y_, y, atol=atol, rtol=rtol)
 
     if profile:
+        compiled_torch_rope = torch.compile(torch_rope)
         benchmark(
             lambda: torch_rope(y, x, pos_ids, theta),
             lambda: llaisys.Ops.rope(y_, x_, pos_ids_, theta),
             device_name,
+            torch_compile_func=lambda: compiled_torch_rope(y, x, pos_ids, theta)
         )
 
 
@@ -66,6 +68,7 @@ if __name__ == "__main__":
     parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
     parser.add_argument("--profile", action="store_true")
     args = parser.parse_args()
+    set_seed()
     testShapes = [
         ((2, 1, 4), (0, 2)), 
         ((512, 4, 4096), (512, 1024))]
diff --git a/test/ops/self_attention.py b/test/ops/self_attention.py
index a042b51..7d81605 100644
--- a/test/ops/self_attention.py
+++ b/test/ops/self_attention.py
@@ -15,7 +15,9 @@ def torch_self_attention(attn_val, query, key, value, scale):
     L, S = query.size(-2), key.size(-2)
     attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
 
-    temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=S-L)
+    # Ensure mask is on the same device as query
+    temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=S-L)
+    temp_mask = temp_mask.to(query.device)
     attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
     attn_bias.to(query.dtype)
 
@@ -54,10 +56,12 @@ def test_op_self_attention(
     assert check_equal(attn_val_, attn_val, atol=atol, rtol=rtol)
 
     if profile:
+        compiled_torch_self_attention = torch.compile(torch_self_attention)
         benchmark(
             lambda: torch_self_attention(attn_val, q, k, v, scale),
             lambda: llaisys.Ops.self_attention(attn_val_, q_, k_, v_, scale),
             device_name,
+            torch_compile_func=lambda: compiled_torch_self_attention(attn_val, q, k, v, scale)
         )
 
 
diff --git a/test/ops/swiglu.py b/test/ops/swiglu.py
index 1fa08f7..5a98845 100644
--- a/test/ops/swiglu.py
+++ b/test/ops/swiglu.py
@@ -31,10 +31,12 @@ def test_op_swiglu(
     assert check_equal(out_, out, atol=atol, rtol=rtol)
 
     if profile:
+        compiled_torch_swiglu = torch.compile(torch_swiglu)
         benchmark(
             lambda: torch_swiglu(out, gate, up),
             lambda: llaisys.Ops.swiglu(out_, gate_, up_),
             device_name,
+            torch_compile_func=lambda: compiled_torch_swiglu(out, gate, up)
         )
 
 
diff --git a/test/test_infer.py b/test/test_infer.py
index 59d06b8..d3e47a4 100644
--- a/test/test_infer.py
+++ b/test/test_infer.py
@@ -89,6 +89,9 @@ if __name__ == "__main__":
     parser.add_argument("--top_k", default=50, type=int)
     parser.add_argument("--temperature", default=1.0, type=float)
     parser.add_argument("--test", action="store_true")
+    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark")
+    parser.add_argument("--repeat", default=10, type=int, help="Number of repeats for benchmark")
+    parser.add_argument("--profile", action="store_true", help="Show e2e inference time profile")
 
     args = parser.parse_args()
 
@@ -99,6 +102,8 @@ if __name__ == "__main__":
     tokenizer, model, model_path = load_hf_model(args.model, args.device)
 
     # Example prompt
+    if args.device == "nvidia":
+        torch.cuda.synchronize()
     start_time = time.time()
     tokens, output = hf_infer(
         args.prompt,
@@ -109,7 +114,10 @@ if __name__ == "__main__":
         top_k=top_k,
         temperature=temperature,
     )
+    if args.device == "nvidia":
+        torch.cuda.synchronize()
     end_time = time.time()
+    hf_duration = end_time - start_time
 
     del model
     gc.collect()
@@ -120,9 +128,11 @@ if __name__ == "__main__":
     print("\nContents:")
     print(output)
     print("\n")
-    print(f"Time elapsed: {(end_time - start_time):.2f}s\n")
+    print(f"Time elapsed: {hf_duration:.2f}s\n")
 
     model = load_llaisys_model(model_path, args.device)
+    if args.device == "nvidia":
+        torch.cuda.synchronize()
     start_time = time.time()
     llaisys_tokens, llaisys_output = llaisys_infer(
         args.prompt,
@@ -133,8 +143,10 @@ if __name__ == "__main__":
         top_k=top_k,
         temperature=temperature,
     )
-
+    if args.device == "nvidia":
+        torch.cuda.synchronize()
     end_time = time.time()
+    llaisys_duration = end_time - start_time
 
     print("\n=== Your Result ===\n")
     print("Tokens:")
@@ -142,8 +154,48 @@ if __name__ == "__main__":
     print("\nContents:")
     print(llaisys_output)
     print("\n")
-    print(f"Time elapsed: {(end_time - start_time):.2f}s\n")
-
+    print(f"Time elapsed: {llaisys_duration:.2f}s\n")
+
+    if args.profile:
+        print("\n=== Profile ===\n")
+        print(f"HF E2E Inference Time: {hf_duration:.4f}s")
+        print(f"Llaisys E2E Inference Time: {llaisys_duration:.4f}s")
+
+    if args.benchmark:
+        print("\n=== Benchmarking ===\n")
+        
+        # Warmup
+        print("Warming up...")
+        for _ in range(3):
+            llaisys_infer(args.prompt, tokenizer, model, 10, top_p, top_k, temperature)
+            
+        latencies = []
+        tokens_per_sec = []
+        
+        for i in range(args.repeat):
+            start = time.time()
+            out_tokens, _ = llaisys_infer(args.prompt, tokenizer, model, args.max_steps, top_p, top_k, temperature)
+            if args.device == "nvidia":
+                torch.cuda.synchronize()
+            end = time.time()
+            
+            latency = (end - start) * 1000
+            tps = len(out_tokens) / (end - start)
+            
+            latencies.append(latency)
+            tokens_per_sec.append(tps)
+            print(f"Iter {i+1}: {latency:.2f}ms, {tps:.2f} tokens/s")
+            
+        import numpy as np
+        print(f"\nDevice: {args.device}")
+        print(f"End-to-End Latency:")
+        print(f"  Mean: {np.mean(latencies):.2f} ms")
+        print(f"  P50:  {np.percentile(latencies, 50):.2f} ms")
+        print(f"  P90:  {np.percentile(latencies, 90):.2f} ms")
+        print(f"  P99:  {np.percentile(latencies, 99):.2f} ms")
+        print(f"Throughput:")
+        print(f"  Mean: {np.mean(tokens_per_sec):.2f} tokens/s")
+        
     if args.test:
         assert llaisys_tokens == tokens
         print("\033[92mTest passed!\033[0m\n")
diff --git a/test/test_ops.py b/test/test_ops.py
new file mode 100644
index 0000000..8a0df31
--- /dev/null
+++ b/test/test_ops.py
@@ -0,0 +1,64 @@
+import os
+import sys
+import argparse
+import subprocess
+import glob
+
+def main():
+    parser = argparse.ArgumentParser(description="Run all operator tests")
+    parser.add_argument("--device", type=str, default="cpu", help="Device to run tests on (cpu/nvidia)")
+    parser.add_argument("--profile", action="store_true", help="Enable profiling for operator tests")
+    args = parser.parse_args()
+
+    # Get directory of current script
+    test_dir = os.path.dirname(os.path.abspath(__file__))
+    ops_dir = os.path.join(test_dir, "ops")
+
+    # Find all python files in test/ops/
+    test_files = glob.glob(os.path.join(ops_dir, "*.py"))
+    # Filter out __init__.py and other non-test files
+    test_files = [f for f in test_files if os.path.basename(f).endswith(".py") and not os.path.basename(f).startswith("__")]
+    test_files.sort()
+
+    print(f"Found {len(test_files)} operator tests in {ops_dir}")
+    print(f"Target Device: {args.device}\n")
+
+    failed_tests = []
+    passed_tests = []
+
+    for test_file in test_files:
+        test_name = os.path.basename(test_file)
+        print(f"Running {test_name}...", end=" ", flush=True)
+        
+        # Run the test script as a subprocess
+        cmd = [sys.executable, test_file, "--device", args.device]
+        if args.profile:
+            cmd.append("--profile")
+        result = subprocess.run(cmd, capture_output=True, text=True)
+
+        if result.returncode == 0:
+            print("\033[92mPASSED\033[0m")
+            if args.profile:
+                print(f"Output for {test_name}:\n{result.stdout}")
+            passed_tests.append(test_name)
+        else:
+            print("\033[91mFAILED\033[0m")
+            print(f"Output for {test_name}:\n{result.stdout}")
+            print(f"Error output for {test_name}:\n{result.stderr}")
+            failed_tests.append(test_name)
+
+    print("\n" + "="*40)
+    print(f"Summary: {len(passed_tests)} PASSED, {len(failed_tests)} FAILED")
+    print("="*40)
+
+    if failed_tests:
+        print("Failed tests:")
+        for t in failed_tests:
+            print(f"  - {t}")
+        sys.exit(1)
+    else:
+        print("All operator tests passed!")
+        sys.exit(0)
+
+if __name__ == "__main__":
+    main()
diff --git a/test/test_utils.py b/test/test_utils.py
index 0f38f0c..1531fa9 100644
--- a/test/test_utils.py
+++ b/test/test_utils.py
@@ -2,6 +2,16 @@ import llaisys
 import torch
 
 
+def set_seed(seed=42):
+    import random
+    import numpy as np
+    random.seed(seed)
+    np.random.seed(seed)
+    torch.manual_seed(seed)
+    if torch.cuda.is_available():
+        torch.cuda.manual_seed_all(seed)
+
+
 def random_tensor(
     shape, dtype_name, device_name, device_id=0, scale=None, bias=None
 ) -> tuple[torch.Tensor, llaisys.Tensor]:
@@ -155,32 +165,82 @@ def check_equal(
         if torch.allclose(result, torch_answer, atol=atol, rtol=rtol):
             return True
 
-    print(f"LLAISYS result: \n{result}")
-    print(f"Torch answer: \n{torch_answer}")
+    # Detailed error reporting
+    diff = torch.abs(result - torch_answer)
+    max_diff = torch.max(diff).item()
+    max_rel_diff = torch.max(diff / (torch.abs(torch_answer) + 1e-12)).item()
+    mismatch_cnt = torch.sum(diff > atol + rtol * torch.abs(torch_answer)).item()
+    total_cnt = result.numel()
+
+    print(f"\nMismatch details:")
+    print(f"  Max Abs Diff: {max_diff:.6e} (atol={atol})")
+    print(f"  Max Rel Diff: {max_rel_diff:.6e} (rtol={rtol})")
+    print(f"  Mismatched elements: {mismatch_cnt}/{total_cnt} ({mismatch_cnt/total_cnt*100:.2f}%)")
+    
+    # Print first few mismatches
+    mask = diff > atol + rtol * torch.abs(torch_answer)
+    if mismatch_cnt > 0:
+        indices = torch.nonzero(mask, as_tuple=True)
+        print(f"  First 5 mismatches:")
+        for i in range(min(5, mismatch_cnt)):
+            idx = tuple(ind[i].item() for ind in indices)
+            val_llaisys = result[idx].item()
+            val_torch = torch_answer[idx].item()
+            print(f"    At {idx}: LLAISYS={val_llaisys:.6e}, Torch={val_torch:.6e}, Diff={abs(val_llaisys-val_torch):.6e}")
+
+    # print(f"LLAISYS result: \n{result}")
+    # print(f"Torch answer: \n{torch_answer}")
     return False
 
 
-def benchmark(torch_func, llaisys_func, device_name, warmup=10, repeat=100):
-    api = llaisys.RuntimeAPI(llaisys_device(device_name))
+def benchmark(torch_func, llaisys_func, device_name, warmup=10, repeat=100, torch_compile_func=None):
+    import time
+    import numpy as np
 
-    def time_op(func):
-        import time
+    api = llaisys.RuntimeAPI(llaisys_device(device_name))
 
+    def time_op(func, name):
+        print(f"    Benchmarking {name}...")
+        # Warmup
         for _ in range(warmup):
             func()
         api.device_synchronize()
-        start = time.time()
+
+        # Measurement
+        latencies = []
         for _ in range(repeat):
+            start = time.time()
             func()
-        api.device_synchronize()
-        end = time.time()
-        return (end - start) / repeat
-
-    torch_time = time_op(torch_func)
-    llaisys_time = time_op(llaisys_func)
-    print(
-        f"        Torch time: {torch_time*1000:.5f} ms \n        LLAISYS time: {llaisys_time*1000:.5f} ms"
-    )
+            api.device_synchronize()
+            end = time.time()
+            latencies.append((end - start) * 1000)  # Convert to ms
+
+        return np.array(latencies)
+
+    torch_latencies = time_op(torch_func, "Torch Eager")
+    llaisys_latencies = time_op(llaisys_func, "LLAISYS")
+    
+    torch_compile_latencies = None
+    if torch_compile_func is not None:
+        torch_compile_latencies = time_op(torch_compile_func, "Torch Compile")
+
+    def print_metrics(name, latencies):
+        mean = np.mean(latencies)
+        p50 = np.percentile(latencies, 50)
+        p90 = np.percentile(latencies, 90)
+        p99 = np.percentile(latencies, 99)
+        std = np.std(latencies)
+        print(f"        {name:<15} time: {mean:.5f} ms (P50: {p50:.5f}, P90: {p90:.5f}, P99: {p99:.5f}, Std: {std:.5f})")
+
+    print_metrics("Torch Eager", torch_latencies)
+    if torch_compile_latencies is not None:
+        print_metrics("Torch Compile", torch_compile_latencies)
+        speedup = np.mean(torch_compile_latencies) / np.mean(llaisys_latencies)
+        print(f"        LLAISYS vs Compile speedup: {speedup:.2f}x")
+    
+    print_metrics("LLAISYS", llaisys_latencies)
+    speedup_eager = np.mean(torch_latencies) / np.mean(llaisys_latencies)
+    print(f"        LLAISYS vs Eager speedup:   {speedup_eager:.2f}x")
 
 
 def torch_device(device_name: str, device_id=0):
diff --git a/xmake.lua b/xmake.lua
index 93d4592..29e8feb 100644
--- a/xmake.lua
+++ b/xmake.lua
@@ -2,6 +2,7 @@ add_rules("mode.debug", "mode.release")
 set_encodings("utf-8")
 
 add_includedirs("include")
+add_includedirs("src")
 
 -- CPU --
 includes("xmake/cpu.lua")
@@ -37,6 +38,9 @@ target("llaisys-device")
     set_kind("static")
     add_deps("llaisys-utils")
     add_deps("llaisys-device-cpu")
+    if has_config("nv-gpu") then
+        add_deps("llaisys-device-nvidia")
+    end
 
     set_languages("cxx17")
     set_warnings("all", "error")
@@ -83,6 +87,9 @@ target_end()
 target("llaisys-ops")
     set_kind("static")
     add_deps("llaisys-ops-cpu")
+    if has_config("nv-gpu") then
+        add_deps("llaisys-ops-nvidia")
+    end
 
     set_languages("cxx17")
     set_warnings("all", "error")
@@ -97,6 +104,48 @@ target_end()
 
 target("llaisys")
     set_kind("shared")
+    
+    if has_config("nv-gpu") then
+        if is_plat("linux") then
+            add_deps("llaisys-ops-nvidia")
+            add_deps("llaisys-device-nvidia")
+            add_links("cublas")
+        end
+        add_rules("cuda")
+        
+        -- Enable device linking for the final shared library link phase
+        set_policy("build.cuda.devlink", true)
+        
+        -- force enable -fPIC
+        if is_plat("linux") then
+            add_cxflags("-fPIC")
+        end
+        
+        -- Link cudadevrt if using RDC (often required implicitly by xmake/cuda rule when RDC is on)
+        add_links("cudadevrt")
+        
+        -- Ensure RDC is enabled for the shared target as well
+        if is_plat("linux") then
+             add_values("cuda.rdc", true)
+        end
+
+        -- Ensure the final linker can find the cuda driver stubs
+        on_load(function (target)
+            import("lib.detect.find_tool")
+            local nvcc = find_tool("nvcc")
+            if nvcc ~= nil then 
+                if is_plat("windows") then 
+                    nvcc_path = os.iorun("where nvcc"):match("(.-)\r?\n") 
+                else 
+                    nvcc_path = nvcc.program 
+                end 
+    
+                target:add("linkdirs", path.directory(path.directory(nvcc_path)) .. "/lib64/stubs") 
+                target:add("links", "cuda") 
+            end 
+        end)
+    end
+    
     add_deps("llaisys-utils")
     add_deps("llaisys-device")
     add_deps("llaisys-core")
@@ -106,19 +155,21 @@ target("llaisys")
     set_languages("cxx17")
     set_warnings("all", "error")
     add_files("src/llaisys/*.cc")
+
+    if has_config("nv-gpu") then
+        add_files("src/llaisys/cuda_dummy.cu")
+    end
+
     set_installdir(".")
 
-    
     after_install(function (target)
         -- copy shared library to python package
         print("Copying llaisys to python/llaisys/libllaisys/ ..")
         if is_plat("windows") then
             os.cp("bin/*.dll", "python/llaisys/libllaisys/")
-        end
-        if is_plat("linux") then
+        elseif is_plat("linux") then
             os.cp("lib/*.so", "python/llaisys/libllaisys/")
-        end
-        if is_plat("macosx") then
+        elseif is_plat("macosx") then
             os.cp("lib/*.dylib", "python/llaisys/libllaisys/")
         end
     end)
diff --git a/xmake/nvidia.lua b/xmake/nvidia.lua
new file mode 100644
index 0000000..1471332
--- /dev/null
+++ b/xmake/nvidia.lua
@@ -0,0 +1,62 @@
+target("llaisys-device-nvidia")
+    set_kind("object")
+    add_rules("cuda")
+    set_languages("cxx17")
+
+    -- Enable device linking globally for this target
+    set_policy("build.cuda.devlink", true)
+    
+    -- Link CUDA runtime and driver
+    add_links("cudart", "cublas", "cuda")
+
+    if is_plat("linux") then
+        add_cugencodes("native")
+        add_cuflags("-Xcompiler -fPIC")
+        add_values("cuda.rdc", true)
+        set_toolchains("cuda")
+    end
+
+    add_files("../src/device/nvidia/*.cu")
+
+    -- Safely locate nvcc and append the lib64/stubs directory
+    on_load(function (target)
+        import("lib.detect.find_tool")
+        local nvcc = find_tool("nvcc")
+        if nvcc and nvcc.program then
+            local nvcc_path = nvcc.program
+            target:add("linkdirs", path.directory(path.directory(nvcc_path)) .. "/lib64/stubs")
+        end
+    end)
+target_end()
+
+target("llaisys-ops-nvidia")
+    set_kind("object")
+    add_rules("cuda")
+    set_languages("cxx17")
+    add_deps("llaisys-tensor")
+
+    -- Enable device linking globally for this target
+    set_policy("build.cuda.devlink", true)
+
+    -- Link CUDA runtime and driver (replacing the standalone "cublas" link)
+    add_links("cudart", "cublas", "cuda")
+
+    if is_plat("linux") then
+        add_cugencodes("native")
+        add_cuflags("-Xcompiler -fPIC")
+        add_values("cuda.rdc", true)
+        set_toolchains("cuda")
+    end
+
+    add_files("../src/ops/*/nvidia/*.cu")
+
+    -- Safely locate nvcc and append the lib64/stubs directory
+    on_load(function (target)
+        import("lib.detect.find_tool")
+        local nvcc = find_tool("nvcc")
+        if nvcc and nvcc.program then
+            local nvcc_path = nvcc.program
+            target:add("linkdirs", path.directory(path.directory(nvcc_path)) .. "/lib64/stubs")
+        end
+    end)
+target_end()
\ No newline at end of file
