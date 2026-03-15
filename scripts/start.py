#!/usr/bin/env python3
import subprocess
import sys
import os
import time
import urllib.request
import ssl
import signal
import atexit
import argparse
from pathlib import Path
import logging

ROOT_DIR = Path(__file__).parent.parent.absolute()
BACKEND_DIR = ROOT_DIR / "src" / "chatbot" / "backend"
FRONTEND_DIR = ROOT_DIR / "src" / "chatbot" / "frontend"
LOGS_DIR = ROOT_DIR / "logs"

LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "launcher.log")
    ]
)
logger = logging.getLogger("launcher")

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

processes = []

def cleanup():
    logger.info("Initiating graceful shutdown...")
    for p, name in processes:
        if p.poll() is None:
            logger.info(f"Stopping {name} (PID {p.pid})...")
            if sys.platform == "win32":
                p.terminate()
            else:
                p.send_signal(signal.SIGTERM)
            
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(f"{name} did not terminate, forcing kill...")
                p.kill()
    logger.info("Shutdown complete.")

atexit.register(cleanup)

def handle_sigint(sig, frame):
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)

def check_health(url: str, timeout: int = 120, interval: int = 2) -> bool:
    start_time = time.time()
    logger.info(f"Waiting for {url} (timeout: {timeout}s)...")
    
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, context=ssl_context, timeout=5) as response:
                if response.status == 200:
                    logger.info(f"Successfully connected to {url}")
                    return True
        except (urllib.error.URLError, ConnectionResetError, ConnectionRefusedError):
            pass
        except Exception as e:
            logger.debug(f"Health check exception: {e}")
            
        time.sleep(interval)
        
    logger.error(f"Timeout waiting for {url}")
    return False

def start_process(name: str, cmd: list, cwd: Path, log_file: str, env: dict = None) -> subprocess.Popen:
    log_path = LOGS_DIR / log_file
    logger.info(f"Starting {name}... (Logs: {log_path})")
    
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
        
    with open(log_path, 'w') as f:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=process_env,
            text=True
        )
        
    processes.append((process, name))
    return process

def main():
    parser = argparse.ArgumentParser(description="LLAISYS Chatbot Launcher")
    parser.add_argument("--model_path", type=str, help="Path to the LLM model or HuggingFace ID", default=None)
    parser.add_argument("--max-sessions", type=int, help="Maximum concurrent sessions", default=100)
    parser.add_argument("--max-steps", type=int, help="Maximum generation steps per request", default=1024)
    parser.add_argument("--device", type=str, choices=["cpu", "nvidia"], help="Device to run the model on (cpu or nvidia)", default="cpu")
    parser.add_argument("--dtype", type=str, choices=["float32", "float16"], help="Model data type", default="float32")
    parser.add_argument("--no-https", action="store_true", help="Disable HTTPS and run in HTTP mode only")
    args = parser.parse_args()

    logger.info("=== LLAISYS Chatbot Launcher ===")
    
    model_path = args.model_path or os.environ.get("MODEL_PATH", "Qwen/Qwen2-0.5B-Instruct")
    device = args.device or os.environ.get("DEVICE", "cpu")
    dtype = args.dtype or os.environ.get("DTYPE", "float32")
    backend_port = os.environ.get("BACKEND_PORT", "6008")
    frontend_port = os.environ.get("FRONTEND_PORT", "6006")
    
    backend_https_port = int(backend_port) + 1
    frontend_https_port = int(frontend_port) + 1
    
    backend_cmd = [
        sys.executable, "server.py", 
        "--model", model_path, 
        "--device", device, 
        "--dtype", dtype,
        "--port", backend_port,
        "--max-sessions", str(args.max_sessions),
        "--max-steps", str(args.max_steps)
    ]
    
    if args.no_https:
        backend_cmd.append("--no-https")
    
    backend_proc = start_process(
        "Backend Server", 
        backend_cmd, 
        BACKEND_DIR, 
        "backend.log",
        env={"PYTHONUNBUFFERED": "1"}
    )
    
    time.sleep(2)
    if backend_proc.poll() is not None:
        logger.error("Backend server crashed immediately. Check logs/backend.log")
        sys.exit(1)
        
    logger.info("Waiting for model to load...")
    
    if args.no_https:
        backend_url = f"http://localhost:{backend_port}/health"
        backend_http_url = f"http://localhost:{backend_port}/health"
    else:
        backend_url = f"https://localhost:{backend_https_port}/health"
        backend_http_url = f"http://localhost:{backend_port}/health"
    
    if not (check_health(backend_url, timeout=300) or check_health(backend_http_url, timeout=5)):
        logger.error("Backend failed to start properly. Check logs/backend.log")
        sys.exit(1)
        
    if not (FRONTEND_DIR / "node_modules").exists():
        logger.info("Installing frontend dependencies...")
        subprocess.run(["npm", "install"], cwd=FRONTEND_DIR, check=True)
    
    frontend_env = {"PORT": frontend_port}
    
    if args.no_https:
        frontend_env["NO_HTTPS"] = "true"
        frontend_env["INTERNAL_API_URL"] = f"http://localhost:{backend_port}"
    else:
        frontend_env["INTERNAL_API_URL"] = f"https://localhost:{backend_https_port}"
        
    frontend_proc = start_process(
        "Frontend Server",
        ["npm", "start"],
        FRONTEND_DIR,
        "frontend.log",
        env=frontend_env
    )
    
    if args.no_https:
        frontend_url = f"http://localhost:{frontend_port}/config"
        frontend_http_url = f"http://localhost:{frontend_port}/config"
    else:
        frontend_url = f"https://localhost:{frontend_https_port}/config"
        frontend_http_url = f"http://localhost:{frontend_port}/config"
    
    if not (check_health(frontend_url, timeout=60) or check_health(frontend_http_url, timeout=5)):
        logger.error("Frontend failed to start properly. Check logs/frontend.log")
        sys.exit(1)
        
    logger.info("="*50)
    logger.info("ALL SERVICES STARTED SUCCESSFULLY")
    if args.no_https:
        logger.info(f"Frontend URL: http://localhost:{frontend_port}")
        logger.info(f"Backend URL:  http://localhost:{backend_port}")
    else:
        logger.info(f"Frontend URL: http://localhost:{frontend_port} (HTTPS: https://localhost:{frontend_https_port})")
        logger.info(f"Backend URL:  http://localhost:{backend_port} (HTTPS: https://localhost:{backend_https_port})")
    logger.info("Press Ctrl+C to stop all services.")
    logger.info("="*50)
    
    try:
        while True:
            for p, name in processes:
                if p.poll() is not None:
                    logger.error(f"CRITICAL: {name} crashed with exit code {p.returncode}!")
                    sys.exit(1)
            time.sleep(2)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
