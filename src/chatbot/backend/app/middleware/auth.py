from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import logging
from ..services.tenant_manager import TenantManager
from ..services.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, tenant_manager: TenantManager, rate_limiter: RateLimiter = None):
        super().__init__(app)
        self.tenant_manager = tenant_manager
        self.rate_limiter = rate_limiter

    async def dispatch(self, request: Request, call_next):
        if request.url.path in ["/", "/docs", "/openapi.json", "/health", "/favicon.ico", "/config"]:
            return await call_next(request)

        if request.url.path.startswith("/v1/admin"):
            return await call_next(request)

        tenant_id = None

        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            api_key = auth_header.split(" ")[1]
            logger.debug(f"AuthMiddleware: Received API Key starting with: {api_key[:6] if len(api_key)>6 else '***'}")
            
            if api_key == "super-secret-admin-token" and request.url.path.startswith("/v1/admin"):
                return await call_next(request)
                
            tenant = await self.tenant_manager.validate_api_key(api_key)
            if not tenant:
                logger.warning(f"AuthMiddleware: Validation failed for key starting with: {api_key[:6] if len(api_key)>6 else '***'}")
                return JSONResponse(status_code=403, content={"detail": "Invalid API Key"})
            tenant_id = tenant.id
            logger.debug(f"AuthMiddleware: Validation successful. Tenant ID: {tenant_id}")
        elif request.headers.get("x-tenant-id"):

            tenant_id = request.headers.get("x-tenant-id")
        else:
            return JSONResponse(status_code=401, content={"detail": "Missing or invalid Authorization header"})

        if not tenant_id:
             return JSONResponse(status_code=403, content={"detail": "Unable to resolve Tenant ID"})

        if self.rate_limiter:
            allowed = await self.rate_limiter.check_limit(tenant_id)
            if not allowed:
                return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

        request.state.tenant_id = tenant_id

        response = await call_next(request)
        return response
