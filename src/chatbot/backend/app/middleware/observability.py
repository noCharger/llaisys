import time
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
import logging

logger = logging.getLogger("llaisys.observability")

class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        logger.info(f"Method={request.method} Path={request.url.path} Status={response.status_code} Latency={process_time:.4f}s")
        response.headers["X-Process-Time"] = str(process_time)
        return response
