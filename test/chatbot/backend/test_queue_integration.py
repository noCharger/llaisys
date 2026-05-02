import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from chatbot.backend.app.services.queue_service import MemoryQueueService, QueueMessage
from chatbot.backend.app.services.scheduler import ClipperScheduler, RequestJob

@pytest.mark.anyio
class TestQueueSchedulerIntegration:
    async def test_scheduler_consumes_from_queue(self):

        queue_service = MemoryQueueService(visibility_timeout=5.0)

        mock_model_service = MagicMock()
        # Strings -> legacy stub mode -> jobs finish in one execute step.
        mock_model_service.forward_batch = AsyncMock(return_value=["A"])
        mock_model_service.forward = MagicMock(return_value=1)

        scheduler = ClipperScheduler(mock_model_service, queue_service=queue_service)

        topic = "model-inference"
        msg = QueueMessage(id="req-1", payload={
            "input_ids": [1, 2, 3],
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "request_id": "req-1",
            "tenant_id": "tenant-test"
        })
        await queue_service.enqueue(topic, msg)

        await scheduler.start()

        await asyncio.sleep(0.1)

        leased = await queue_service.lease(topic, 1)
        assert len(leased) == 0

        await scheduler.stop()

    async def test_scheduler_nack_on_failure(self):
        queue_service = MemoryQueueService(visibility_timeout=5.0)
        mock_model_service = MagicMock()

        mock_model_service.forward_batch = AsyncMock(side_effect=Exception("Model Error"))

        mock_model_service.forward = MagicMock(side_effect=Exception("Model Error"))

        scheduler = ClipperScheduler(mock_model_service, queue_service=queue_service)

        topic = "model-inference"
        msg = QueueMessage(id="req-fail", payload={
            "input_ids": [1], "request_id": "req-fail", "tenant_id": "tenant-test"
        })
        await queue_service.enqueue(topic, msg)

        await scheduler.start()

        await asyncio.sleep(0.5)
        await scheduler.stop()

        leased_msgs = await queue_service.lease(topic, 1)

        if len(leased_msgs) == 0:
            assert len(queue_service.leased) == 1
            msg_id = list(queue_service.leased.keys())[0]
            _, msg, _ = queue_service.leased[msg_id]
            assert msg.id == "req-fail"
            assert msg.attempt >= 1
        else:
            assert leased_msgs[0].id == "req-fail"
            assert leased_msgs[0].attempt >= 1
