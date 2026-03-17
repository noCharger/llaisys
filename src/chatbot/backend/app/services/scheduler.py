import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any, Dict
from .queue_service import IQueueService, QueueMessage

logger = logging.getLogger("llaisys.scheduler")

@dataclass
class RequestJob:
    request_id: str
    session_ptr: Any
    input_ids: List[int]
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int = 4096
    future: asyncio.Future = field(default_factory=asyncio.Future)
    created_at: float = field(default_factory=time.time)
    queue_msg: Optional[QueueMessage] = None

class ClipperScheduler:
    def __init__(self, model_service: Any, queue_service: Optional[IQueueService] = None, slo_ms: float = 200.0, batch_wait_ms: float = 5.0):
        self.model_service = model_service
        self.queue_service = queue_service
        self.internal_queue: asyncio.Queue[RequestJob] = asyncio.Queue()
        self.running = False
        self.task: Optional[asyncio.Task] = None

        self.slo_ms = slo_ms
        self.batch_wait_ms = batch_wait_ms / 1000.0
        self.current_batch_size = 1.0
        self.max_batch_size = 32.0
        self.min_batch_size = 1.0
        self.success_count = 0
        self.aimd_additive_step = 1.0
        self.aimd_multiplicative_factor = 0.9

        self.total_inferences = 0
        self.last_update_time = time.time()
        self.latency_ema = 0.0
        self.alpha = 0.1

        self.pending_jobs: Dict[str, RequestJob] = {}

    async def start(self):
        self.running = True
        self.task = asyncio.create_task(self._run_loop())

    async def stop(self):
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

    async def submit(self, job: RequestJob):
        if self.queue_service:
            self.pending_jobs[job.request_id] = job
            msg = QueueMessage(
                id=job.request_id,
                payload={
                    "session_ptr": job.session_ptr,
                    "input_ids": job.input_ids,
                    "temperature": job.temperature,
                    "top_p": job.top_p,
                    "top_k": job.top_k,
                    "max_tokens": job.max_tokens
                },
                metadata={"priority": 1}
            )
            await self.queue_service.enqueue("model-inference", msg)
        else:
            await self.internal_queue.put(job)

    async def _run_loop(self):
        while self.running:
            try:
                batch: List[RequestJob] = []
                try:
                    if self.queue_service:
                        topic = "model-inference"
                        msgs = await self.queue_service.lease(topic, count=1, timeout_ms=1000)
                        if msgs:
                            for msg in msgs:
                                job = self.pending_jobs.pop(msg.id, None)
                                if not job:
                                    job = RequestJob(
                                        request_id=msg.id,
                                        session_ptr=msg.payload.get("session_ptr"),
                                        input_ids=msg.payload.get("input_ids"),
                                        temperature=msg.payload.get("temperature", 0.7),
                                        top_p=msg.payload.get("top_p", 0.9),
                                        top_k=msg.payload.get("top_k", 50),
                                        max_tokens=msg.payload.get("max_tokens", 4096)
                                    )
                                job.queue_msg = msg
                                batch.append(job)
                    else:
                        job = await asyncio.wait_for(self.internal_queue.get(), timeout=1.0)
                        batch.append(job)
                except asyncio.TimeoutError:
                    continue

                target_bs = int(self.current_batch_size)
                if len(batch) < target_bs and batch:
                    start_wait = time.time()
                    while len(batch) < target_bs:
                        remaining_time = self.batch_wait_ms - (time.time() - start_wait)
                        if remaining_time <= 0:
                            break
                        try:
                            if self.queue_service:
                                topic = "model-inference"
                                timeout_ms = int(remaining_time * 1000)
                                if timeout_ms <= 0: timeout_ms = 1
                                needed = target_bs - len(batch)
                                msgs = await self.queue_service.lease(topic, count=needed, timeout_ms=timeout_ms)
                                for msg in msgs:
                                    job = self.pending_jobs.pop(msg.id, None)
                                    if not job:
                                        job = RequestJob(
                                            request_id=msg.id,
                                            session_ptr=msg.payload.get("session_ptr"),
                                            input_ids=msg.payload.get("input_ids"),
                                            temperature=msg.payload.get("temperature", 0.7),
                                            top_p=msg.payload.get("top_p", 0.9),
                                            top_k=msg.payload.get("top_k", 50),
                                            max_tokens=msg.payload.get("max_tokens", 4096)
                                        )
                                    job.queue_msg = msg
                                    batch.append(job)
                            else:
                                job = await asyncio.wait_for(self.internal_queue.get(), timeout=remaining_time)
                                batch.append(job)
                        except asyncio.TimeoutError:
                            break

                start_time = time.time()
                if batch:
                    await self._process_batch(batch)
                    if self.queue_service:
                        ack_ids = []
                        fail_ids = []
                        for job in batch:
                            if job.queue_msg:
                                if job.future.done() and (job.future.cancelled() or job.future.exception()):
                                    fail_ids.append(job.queue_msg.id)
                                else:
                                    ack_ids.append(job.queue_msg.id)
                        if ack_ids:
                            await self.queue_service.ack(topic, ack_ids)
                        if fail_ids:
                            await self.queue_service.nack(topic, fail_ids)

                    latency = (time.time() - start_time) * 1000.0
                    self._update_batch_size(latency)
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}", exc_info=True)
                await asyncio.sleep(0.1)

    async def _process_batch(self, batch: List[RequestJob]):
        if not batch: return
        if hasattr(self.model_service, "forward_batch"):
            try:
                results = await self.model_service.forward_batch(batch)
                for job, result in zip(batch, results):
                    if not job.future.done():
                        job.future.set_result(result)
            except Exception as e:
                for job in batch:
                    if not job.future.done():
                        job.future.set_exception(e)
        else:
            for job in batch:
                try:
                    next_token = self.model_service.forward(job.session_ptr, job.input_ids, job.temperature, job.top_p, job.top_k)
                    if not job.future.done():
                        job.future.set_result(next_token)
                except Exception as e:
                    if not job.future.done():
                        job.future.set_exception(e)

    def _update_batch_size(self, latency_ms: float):
        self.total_inferences += 1
        if self.total_inferences == 1:
            self.latency_ema = latency_ms
        else:
            self.latency_ema = (self.alpha * latency_ms) + ((1 - self.alpha) * self.latency_ema)

        if latency_ms <= self.slo_ms:
            self.success_count += 1
            if self.success_count >= 3:
                self.current_batch_size += self.aimd_additive_step
                self.success_count = 0
                if self.current_batch_size > self.max_batch_size:
                    self.current_batch_size = self.max_batch_size
        else:
            self.current_batch_size *= self.aimd_multiplicative_factor
            self.success_count = 0
            if self.current_batch_size < self.min_batch_size:
                self.current_batch_size = self.min_batch_size
