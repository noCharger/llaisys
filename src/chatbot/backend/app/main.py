import logging
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import sys

# Get log level from environment variable, default to INFO
log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)

# Configure root logger to handle Uvicorn and application logs uniformly
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Explicitly override uvicorn access and error loggers to use our format
for logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"):
    uvicorn_logger = logging.getLogger(logger_name)
    uvicorn_logger.handlers = []
    uvicorn_logger.propagate = True

logger = logging.getLogger(__name__)

from .routers import chat, admin
from .dependencies import scheduler, tenant_manager, rate_limiter
from .middleware.auth import AuthMiddleware
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    from .dependencies import tenant_manager
    app.state.tenant_manager = tenant_manager
    
    await scheduler.start()
    yield
    await scheduler.stop()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(AuthMiddleware, tenant_manager=tenant_manager, rate_limiter=rate_limiter)

app.include_router(chat.router, prefix="/v1")
app.include_router(admin.router, prefix="/v1")

@app.get("/config")
def get_config(request: Request):
    """Returns public configuration for the frontend."""
    scheme = request.url.scheme
    host = request.url.hostname
    port = request.url.port

    api_url = f"{scheme}://{host}:{port}" if port else f"{scheme}://{host}"
    return {"apiUrl": api_url}

@app.get("/health")
def health_check():
    return {"status": "ok"}
