from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import time
import asyncio
import logging

logger = logging.getLogger("llaisys.queue")

@dataclass(order=True)
class QueueMessage:
    id: str = field(compare=False)
    payload: Dict[str, Any] = field(compare=False)
    metadata: Dict[str, Any] = field(compare=False, default_factory=dict)
    attempt: int = field(compare=False, default=0)
    created_at: float = field(compare=True, default_factory=time.time)

class IQueueService(ABC):
    @abstractmethod
    async def enqueue(self, topic: str, message: QueueMessage) -> str:
        pass

    @abstractmethod
    async def lease(self, topic: str, count: int, timeout_ms: int = 0) -> List[QueueMessage]:
        pass

    @abstractmethod
    async def ack(self, topic: str, message_ids: List[str]) -> None:
        pass

    @abstractmethod
    async def nack(self, topic: str, message_ids: List[str]) -> None:
        pass

class MemoryQueueService(IQueueService):
    def __init__(self, visibility_timeout: float = 30.0):
        self.visibility_timeout = visibility_timeout
        self.queues: Dict[str, asyncio.PriorityQueue] = {}
        self.leased: Dict[str, Tuple[float, QueueMessage, str]] = {}
        self._lock = None

    @property
    def lock(self):
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _get_queue(self, topic: str) -> asyncio.PriorityQueue:
        if topic not in self.queues:
            self.queues[topic] = asyncio.PriorityQueue()
        return self.queues[topic]

    async def enqueue(self, topic: str, message: QueueMessage) -> str:
        queue = self._get_queue(topic)
        priority = message.metadata.get("priority", 0)
        await queue.put((-priority, message.created_at, message.id, message))
        return message.id

    async def lease(self, topic: str, count: int, timeout_ms: int = 0) -> List[QueueMessage]:
        now = time.time()
        expired_ids = []

        async with self.lock:
            for msg_id, (expiry, msg, msg_topic) in self.leased.items():
                if msg_topic == topic and now > expiry:
                    expired_ids.append(msg_id)

            for msg_id in expired_ids:
                expiry, msg, msg_topic = self.leased.pop(msg_id)
                queue = self._get_queue(msg_topic)
                priority = msg.metadata.get("priority", 0)
                msg.attempt += 1
                await queue.put((-priority, msg.created_at, msg.id, msg))

            queue = self._get_queue(topic)
            messages = []
            deadline = now + (timeout_ms / 1000.0)

            for _ in range(count):
                try:
                    if timeout_ms > 0:
                        remaining = deadline - time.time()
                        if remaining <= 0:
                            if not queue.empty():
                                item = queue.get_nowait()
                            else:
                                break
                        else:
                            try:
                                item = await asyncio.wait_for(queue.get(), timeout=remaining)
                            except asyncio.TimeoutError:
                                break
                    else:
                        item = queue.get_nowait()

                    _, _, _, msg = item
                    self.leased[msg.id] = (time.time() + self.visibility_timeout, msg, topic)
                    messages.append(msg)
                except asyncio.QueueEmpty:
                    break
            return messages

    async def ack(self, topic: str, message_ids: List[str]) -> None:
        async with self.lock:
            for msg_id in message_ids:
                if msg_id in self.leased:
                    _, _, msg_topic = self.leased[msg_id]
                    if msg_topic == topic:
                        del self.leased[msg_id]

    async def nack(self, topic: str, message_ids: List[str]) -> None:
        async with self.lock:
            for msg_id in message_ids:
                if msg_id in self.leased:
                    expiry, msg, msg_topic = self.leased.pop(msg_id)
                    if msg_topic == topic:
                        msg.attempt += 1
                        queue = self._get_queue(topic)
                        priority = msg.metadata.get("priority", 0)
                        await queue.put((-priority, msg.created_at, msg.id, msg))
