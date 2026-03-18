from fastapi import APIRouter, Depends, HTTPException, Header
from typing import List
from ..models.tenant import Tenant, TenantCreate, Quota, APIKey
from ..services.tenant_manager import TenantManager
from ..dependencies import get_tenant_manager

router = APIRouter(prefix="/admin", tags=["admin"])

ADMIN_TOKEN = "super-secret-admin-token"

def verify_admin(authorization: str = Header(None)):
    if authorization:
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid admin token format")
        token = authorization.split(" ")[1]
        if token != ADMIN_TOKEN:
            raise HTTPException(status_code=403, detail="Forbidden: Invalid admin token")
    else:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    return True

@router.post("/tenants", response_model=Tenant, status_code=201)
async def create_tenant(
    tenant_in: TenantCreate,
    tenant_manager: TenantManager = Depends(get_tenant_manager),
    _=Depends(verify_admin)
):
    try:
        tenant = await tenant_manager.create_tenant(tenant_in)
        return tenant
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/tenants", response_model=List[Tenant])
async def list_tenants(
    tenant_manager: TenantManager = Depends(get_tenant_manager),
    _=Depends(verify_admin)
):
    tenants = await tenant_manager.storage.list_tenants()
    return tenants

@router.post("/tenants/{tenant_id}/keys", status_code=201)
async def generate_api_key(
    tenant_id: str,
    tenant_manager: TenantManager = Depends(get_tenant_manager),
    _=Depends(verify_admin)
):
    tenant = await tenant_manager.get_tenant(tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")
    
    api_key = await tenant_manager.create_api_key(tenant_id)
    return {"key": api_key, "tenant_id": tenant_id}
