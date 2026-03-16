import pytest
import asyncio
from chatbot.backend.app.services.queue_service import MemoryQueueService, QueueMessage

@pytest.mark.anyio
class TestMemoryQueueService:
    async def test_enqueue_dequeue(self):
        queue = MemoryQueueService()
        msg = QueueMessage(id="msg-1", payload={"data": 123})

        await queue.enqueue("topic-1", msg)

        leased = await queue.lease("topic-1", count=1)
        assert len(leased) == 1
        assert leased[0].id == "msg-1"

        leased_again = await queue.lease("topic-1", count=1, timeout_ms=10)
        assert len(leased_again) == 0

    async def test_ack(self):
        queue = MemoryQueueService()
        msg = QueueMessage(id="msg-1", payload={})
        await queue.enqueue("topic-1", msg)

        leased = await queue.lease("topic-1", count=1)
        assert len(leased) == 1

        await queue.ack("topic-1", [msg.id])

        assert len(queue.leased) == 0

    async def test_nack(self):
        queue = MemoryQueueService()
        msg = QueueMessage(id="msg-1", payload={})
        await queue.enqueue("topic-1", msg)

        leased = await queue.lease("topic-1", count=1)
        assert len(leased) == 1

        await queue.nack("topic-1", [msg.id])

        leased_again = await queue.lease("topic-1", count=1)
        assert len(leased_again) == 1
        assert leased_again[0].attempt == 1

    async def test_visibility_timeout(self):

        queue = MemoryQueueService(visibility_timeout=0.1)
        msg = QueueMessage(id="msg-1", payload={})
        await queue.enqueue("topic-1", msg)

        await queue.lease("topic-1", count=1)

        await asyncio.sleep(0.2)

        leased = await queue.lease("topic-1", count=1)
        assert len(leased) == 1
        assert leased[0].attempt == 1
