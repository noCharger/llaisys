from typing import Dict, Optional, List
from ..models.session import Session
from ..models.chat import Message
import asyncio

class SessionDAO:
    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._lock = None

    @property
    def lock(self):
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def create_session(self, tenant_id: str) -> Session:
        session = Session(tenant_id=tenant_id)
        async with self.lock:
            self._sessions[session.id] = session
        return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    async def add_message(self, session_id: str, message: Message) -> Optional[Session]:
        async with self.lock:
            session = self._sessions.get(session_id)
            if session:
                session.messages.append(message)
                return session
            return None

    async def list_sessions(self, tenant_id: str) -> List[Session]:
        return [s for s in self._sessions.values() if s.tenant_id == tenant_id]
