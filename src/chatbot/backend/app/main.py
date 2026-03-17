from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from .routers import chat
from .dependencies import scheduler, tenant_manager, rate_limiter
from .middleware.auth import AuthMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):

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
