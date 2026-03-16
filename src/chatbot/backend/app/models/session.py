from pydantic import BaseModel, Field
from typing import List, Dict, Any
from datetime import datetime
import uuid
from .chat import Message

class Session(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    messages: List[Message] = []
    metadata: Dict[str, Any] = {}
