import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from chatbot.backend.app.services.scheduler import ClipperScheduler, RequestJob

@pytest.mark.anyio
class TestClipperScheduler:
    async def test_aimd_logic(self):
        mock_model = MagicMock()
        mock_model.forward_batch = AsyncMock(return_value=[1])

        scheduler = ClipperScheduler(mock_model, slo_ms=10.0)
        assert scheduler.current_batch_size == 1.0

        scheduler._update_batch_size(5.0)
        scheduler._update_batch_size(5.0)
        scheduler._update_batch_size(5.0)

        assert scheduler.current_batch_size == 2.0

        scheduler._update_batch_size(15.0)

        assert scheduler.current_batch_size == 2.0 * 0.9

    async def test_delayed_batching(self):
        mock_model = MagicMock()
        # Strings trigger legacy stub-mode: scheduler marks jobs finished
        # after one Execute, suitable for testing batching latency.
        mock_model.forward_batch = AsyncMock(return_value=["A", "B"])

        scheduler = ClipperScheduler(mock_model, batch_wait_ms=50.0)
        scheduler.current_batch_size = 2.0

        await scheduler.start()

        job1 = RequestJob(request_id="1", tenant_id="s1", input_ids=[1], temperature=0.7, top_p=0.9, top_k=50)
        await scheduler.submit(job1)

        assert not job1.future.done()

        job2 = RequestJob(request_id="2", tenant_id="s2", input_ids=[2], temperature=0.7, top_p=0.9, top_k=50)
        await scheduler.submit(job2)

        await asyncio.sleep(0.1)

        assert job1.future.done()
        assert job2.future.done()

        await scheduler.stop()
