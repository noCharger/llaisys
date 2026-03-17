from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

class Quota(BaseModel):
    requests_per_minute: int = 60
    max_tokens_per_day: int = 100000
    kv_pool_slice: int = 1024

class APIKey(BaseModel):
    key: str
    created_at: datetime = Field(default_factory=datetime.now)
    is_active: bool = True

class TenantCreate(BaseModel):
    name: str
    quotas: Optional[Quota] = None
    features: Dict[str, bool] = {}

class TenantUpdate(BaseModel):
    name: Optional[str] = None
    quotas: Optional[Quota] = None
    features: Optional[Dict[str, bool]] = None

class Tenant(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    api_keys: List[APIKey] = []
    quotas: Quota
    features: Dict[str, bool] = {}
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
