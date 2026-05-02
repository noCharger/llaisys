"""Three-phase continuous-batching scheduler.

Each tick: Cleanup (release finished jobs, commit prefix history) → Inject
(acquire pool blocks, preempt if exhausted) → Execute (pack under token_budget,
mix prefill chunks with decode). All pool access goes through KVPoolService
with the request's tenant_id.
"""
from __future__ import annotations

import asyncio
import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .queue_service import IQueueService, QueueMessage
from .kv_pool_service import KVPoolService

logger = logging.getLogger("llaisys.scheduler")

TOPIC = "model-inference"

# Sentinel pushed onto a job's tokens_queue when it finishes.
DONE = object()


class Phase(enum.Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class RequestJob:
    request_id: str
    tenant_id: str
    input_ids: List[int]              # full prompt; rewritten only on preempt
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 4096

    # Set by the scheduler.
    block_id: Optional[int] = None
    kv_pos: int = 0
    output_tokens: List[int] = field(default_factory=list)
    pending_input_ids: List[int] = field(default_factory=list)  # transient per Execute
    tokens_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    phase: Phase = Phase.PREFILL
    finished: bool = False
    finish_reason: str = ""
    preempted_count: int = 0
    # Persists across preempts so max_tokens accounts for re-prefill output.
    total_output_count: int = 0

    queue_msg: Optional[QueueMessage] = None
    future: asyncio.Future = field(default_factory=asyncio.Future)
    created_at: float = field(default_factory=time.time)

    @property
    def tokens_generated(self) -> int:
        """Total output tokens across preempt cycles."""
        return self.total_output_count

    @property
    def is_prefilling(self) -> bool:
        return self.kv_pos < len(self.input_ids)


class ClipperScheduler:
    """Iteration-level batching scheduler. token_budget caps per-step tokens
    (Sarathi-Serve); max_preemptions_per_job bounds thrash."""

    def __init__(self, model_service: Any,
                 queue_service: Optional[IQueueService] = None,
                 kv_pool_service: Optional[KVPoolService] = None,
                 slo_ms: float = 200.0,
                 batch_wait_ms: float = 5.0,
                 token_budget: int = 4096,
                 max_preemptions_per_job: int = 2,
                 eos_token: Optional[int] = None):
        self.model_service = model_service
        self.queue_service = queue_service
        self.kv_pool = kv_pool_service
        self.eos_token = eos_token

        self.internal_queue: asyncio.Queue[RequestJob] = asyncio.Queue()
        self.pending_jobs: Dict[str, RequestJob] = {}      # in-flight queue messages
        self.active_requests: Dict[str, RequestJob] = {}   # currently batched

        self._work_event: asyncio.Event = asyncio.Event()

        self.running = False
        self.task: Optional[asyncio.Task] = None

        self.slo_ms = slo_ms
        self.batch_wait_ms = batch_wait_ms / 1000.0
        self.token_budget = token_budget
        self.max_preemptions_per_job = max_preemptions_per_job

        self.current_batch_size = 1.0
        self.max_batch_size = 32.0
        self.min_batch_size = 1.0
        self.success_count = 0
        self.aimd_additive_step = 1.0
        self.aimd_multiplicative_factor = 0.9

        self.total_iterations = 0
        self.latency_ema = 0.0
        self.alpha = 0.1
        self.prefill_count = 0
        self.decode_count = 0
        self.preempt_count = 0

    # ---------- lifecycle ----------

    async def start(self):
        self.running = True
        self.task = asyncio.create_task(self._run_loop())

    async def stop(self):
        self.running = False
        self._work_event.set()
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

    def set_eos_token(self, eos_token: int):
        """Returned tokens equal to this id mark the job as finished."""
        self.eos_token = int(eos_token)

    async def submit(self, job: RequestJob):
        if not job.tenant_id:
            raise ValueError("RequestJob.tenant_id is required")
        if self.queue_service:
            self.pending_jobs[job.request_id] = job
            msg = QueueMessage(
                id=job.request_id,
                payload={
                    "tenant_id": job.tenant_id,
                    "input_ids": list(job.input_ids),
                    "temperature": job.temperature,
                    "top_p": job.top_p,
                    "top_k": job.top_k,
                    "max_tokens": job.max_tokens,
                },
                metadata={"priority": 1},
            )
            try:
                await self.queue_service.enqueue(TOPIC, msg)
            except Exception:
                self.pending_jobs.pop(job.request_id, None)
                raise
        else:
            await self.internal_queue.put(job)
        self._work_event.set()

    # ---------- diagnostics ----------

    def stats(self) -> Dict[str, Any]:
        return {
            "active": len(self.active_requests),
            "current_batch_size": float(self.current_batch_size),
            "latency_ema_ms": self.latency_ema,
            "iterations": self.total_iterations,
            "prefill_steps": self.prefill_count,
            "decode_steps": self.decode_count,
            "preempts": self.preempt_count,
            "token_budget": self.token_budget,
            "pool": self.kv_pool.usage() if self.kv_pool else None,
        }

    # ---------- main loop ----------

    async def _run_loop(self):
        while self.running:
            try:
                self._mark_cancelled_jobs_finished()
                await self._cleanup_finished()
                await self._inject_new_requests()

                if not self.active_requests:
                    self._work_event.clear()
                    try:
                        await asyncio.wait_for(self._work_event.wait(), timeout=0.1)
                    except asyncio.TimeoutError:
                        pass
                    continue

                t0 = time.time()
                await self._execute_batch()
                latency_ms = (time.time() - t0) * 1000.0
                self._update_batch_size(latency_ms)
            except Exception as e:
                logger.error("scheduler loop error: %s", e, exc_info=True)
                await asyncio.sleep(0.05)

    def _mark_cancelled_jobs_finished(self):
        for job in self.active_requests.values():
            if not job.finished and job.future.cancelled():
                job.finished = True
                job.finish_reason = "cancelled"

    # ---------- Cleanup ----------

    async def _cleanup_finished(self):
        finished_ids = [rid for rid, j in self.active_requests.items() if j.finished]
        if not finished_ids:
            return

        ack_ids: List[str] = []
        fail_ids: List[str] = []

        for rid in finished_ids:
            job = self.active_requests.pop(rid)
            release_ok = True
            if self.kv_pool and job.block_id is not None:
                try:
                    # Commit so future same-tenant requests can prefix-match.
                    full_history = list(job.input_ids) + list(job.output_tokens)
                    self.kv_pool.commit(job.tenant_id, job.block_id,
                                        job.kv_pos, full_history)
                    self.kv_pool.release(job.tenant_id, job.block_id)
                except Exception as e:
                    logger.error("pool release failed for %s: %s", rid, e)
                    release_ok = False

            await job.tokens_queue.put(DONE)
            if not job.future.done():
                if job.finish_reason == "cancelled":
                    pass
                else:
                    job.future.set_result({
                        "tokens": list(job.output_tokens),
                        "finish_reason": job.finish_reason or "stop",
                    })

            if job.queue_msg is not None:
                if (job.future.cancelled()
                        or (job.future.done() and job.future.exception())
                        or job.finish_reason == "cancelled"
                        or not release_ok):
                    fail_ids.append(job.queue_msg.id)
                else:
                    ack_ids.append(job.queue_msg.id)

        if self.queue_service:
            if ack_ids:
                await self.queue_service.ack(TOPIC, ack_ids)
            if fail_ids:
                await self.queue_service.nack(TOPIC, fail_ids)

    # ---------- Inject ----------

    async def _inject_new_requests(self):
        capacity = max(0, int(self.current_batch_size) - len(self.active_requests))
        if capacity <= 0:
            return

        new_jobs = await self._lease_jobs(capacity)
        for job in new_jobs:
            if self.kv_pool is not None:
                ok = self._try_acquire_with_preemption(job)
                if not ok:
                    continue
            job.phase = Phase.PREFILL
            self.active_requests[job.request_id] = job

    def _try_acquire_with_preemption(self, job: RequestJob) -> bool:
        """pool.acquire with one retry through preemption on exhaustion."""
        try:
            block_id, matched_len = self.kv_pool.acquire(
                job.tenant_id, job.input_ids)
        except Exception as e:
            self._fail_job(job, e)
            return False

        if block_id < 0:
            if self._try_preempt():
                self.preempt_count += 1
                try:
                    block_id, matched_len = self.kv_pool.acquire(
                        job.tenant_id, job.input_ids)
                except Exception as e:
                    self._fail_job(job, e)
                    return False
            if block_id < 0:
                self._fail_job(job, RuntimeError(
                    "KV pool exhausted; preemption failed"))
                return False

        # Leave at least one token unfed so the next forward yields a sample.
        if matched_len >= len(job.input_ids):
            matched_len = max(0, len(job.input_ids) - 1)

        job.block_id = block_id
        job.kv_pos = matched_len
        return True

    def _try_preempt(self) -> bool:
        """Evict the oldest active job to free a pool block. Internal-queue
        mode only; the topic queue can't put messages at the front."""
        if not self.active_requests or self.kv_pool is None:
            return False

        candidates = [
            j for j in self.active_requests.values()
            if j.preempted_count < self.max_preemptions_per_job
        ]
        if not candidates:
            return False
        victim = min(candidates, key=lambda j: j.created_at)

        if victim.queue_msg is not None:
            return False  # queue-mode preemption unsupported

        combined_input = list(victim.input_ids) + list(victim.output_tokens)

        try:
            if victim.block_id is not None:
                self.kv_pool.commit(victim.tenant_id, victim.block_id,
                                    victim.kv_pos, combined_input)
                self.kv_pool.release(victim.tenant_id, victim.block_id)
        except Exception as e:
            logger.error("preempt commit/release failed for %s: %s",
                         victim.request_id, e)
            return False

        # Resume re-prefills, but the commit above lets the next acquire
        # skip the cached prefix.
        victim.input_ids = combined_input
        victim.block_id = None
        victim.kv_pos = 0
        victim.output_tokens = []
        victim.phase = Phase.PREFILL
        victim.preempted_count += 1

        del self.active_requests[victim.request_id]
        self.internal_queue.put_nowait(victim)
        logger.info("preempted job %s (tenant=%s, preempt #%d)",
                    victim.request_id, victim.tenant_id, victim.preempted_count)
        return True

    async def _lease_jobs(self, count: int) -> List[RequestJob]:
        jobs: List[RequestJob] = []
        if self.queue_service is not None:
            if self.active_requests:
                timeout_ms = 1
            else:
                timeout_ms = max(1, int(self.batch_wait_ms * 1000))
            msgs = await self.queue_service.lease(TOPIC, count=count, timeout_ms=timeout_ms)
            for msg in msgs:
                job = self.pending_jobs.pop(msg.id, None)
                if job is None:
                    job = RequestJob(
                        request_id=msg.id,
                        tenant_id=msg.payload.get("tenant_id", ""),
                        input_ids=list(msg.payload.get("input_ids", [])),
                        temperature=msg.payload.get("temperature", 1.0),
                        top_p=msg.payload.get("top_p", 0.9),
                        top_k=msg.payload.get("top_k", 50),
                        max_tokens=msg.payload.get("max_tokens", 4096),
                    )
                if not job.tenant_id:
                    self._fail_job(job, ValueError("missing tenant_id on queued job"))
                    continue
                job.queue_msg = msg
                jobs.append(job)
        else:
            deadline = time.time() + self.batch_wait_ms
            while len(jobs) < count:
                remaining = deadline - time.time()
                if remaining <= 0 and jobs:
                    break
                try:
                    timeout = max(remaining, 0.001) if jobs else max(remaining, 0.05)
                    job = await asyncio.wait_for(self.internal_queue.get(), timeout=timeout)
                    jobs.append(job)
                except asyncio.TimeoutError:
                    break
        return jobs

    # ---------- Execute ----------

    async def _execute_batch(self):
        batch_all = list(self.active_requests.values())
        if not batch_all:
            return

        # Decode jobs claim 1 token each; prefill jobs (FIFO) share the rest.
        # An oversized prompt waits for the next tick instead of pinning the GPU.
        decode_jobs = [j for j in batch_all if not j.is_prefilling]
        prefill_jobs = sorted(
            (j for j in batch_all if j.is_prefilling),
            key=lambda j: j.created_at,
        )

        remaining = max(0, self.token_budget - len(decode_jobs))
        batch_in_step: List[RequestJob] = []

        for j in decode_jobs:
            last_tok = j.output_tokens[-1] if j.output_tokens else j.input_ids[-1]
            j.pending_input_ids = [int(last_tok)]
            batch_in_step.append(j)

        for j in prefill_jobs:
            if remaining <= 0:
                continue  # skipped this tick; next tick will reconsider
            unfed = len(j.input_ids) - j.kv_pos
            feed = min(unfed, remaining)
            if feed <= 0:
                continue
            j.pending_input_ids = list(j.input_ids[j.kv_pos: j.kv_pos + feed])
            batch_in_step.append(j)
            remaining -= feed

        if not batch_in_step:
            return

        for j in batch_in_step:
            if j.is_prefilling:
                self.prefill_count += 1
            else:
                self.decode_count += 1

        try:
            results = await self.model_service.forward_batch(batch_in_step)
        except Exception as e:
            for j in batch_in_step:
                self._fail_job(j, e)
            return

        if results and isinstance(results[0], int):
            for j, tok in zip(batch_in_step, results):
                self._apply_token(j, tok)
        else:
            # Legacy stub path: each result is a complete string.
            for j, text in zip(batch_in_step, results):
                j.output_tokens.extend(list(text) if isinstance(text, str) else [])
                j.finished = True
                j.finish_reason = "stop"
                if not j.future.done():
                    j.future.set_result(text)

    def _apply_token(self, job: RequestJob, token):
        consumed = len(job.pending_input_ids)
        job.kv_pos += consumed

        if job.is_prefilling:
            # Mid-prefill chunk: sampled token is meaningless, drop it.
            job.phase = Phase.PREFILL
            return

        # First post-prefill sample, or a decode step.
        tok_int = int(token)
        job.output_tokens.append(tok_int)
        job.total_output_count += 1
        job.phase = Phase.DECODE

        try:
            job.tokens_queue.put_nowait(tok_int)
        except asyncio.QueueFull:
            logger.warning("tokens_queue full for %s, dropping token", job.request_id)

        if self.eos_token is not None and tok_int == int(self.eos_token):
            job.finished = True
            job.finish_reason = "stop"
        elif job.total_output_count >= job.max_tokens:
            job.finished = True
            job.finish_reason = "length"

    def _fail_job(self, job: RequestJob, error: Exception):
        job.finished = True
        job.finish_reason = "error"
        if not job.future.done():
            job.future.set_exception(error)
        try:
            job.tokens_queue.put_nowait(DONE)
        except Exception:
            pass

    # ---------- AIMD ----------

    def _update_batch_size(self, latency_ms: float):
        self.total_iterations += 1
        if self.total_iterations == 1:
            self.latency_ema = latency_ms
        else:
            self.latency_ema = self.alpha * latency_ms + (1 - self.alpha) * self.latency_ema

        if latency_ms <= self.slo_ms:
            self.success_count += 1
            if self.success_count >= 3:
                self.current_batch_size = min(
                    self.max_batch_size,
                    self.current_batch_size + self.aimd_additive_step,
                )
                self.success_count = 0
        else:
            self.current_batch_size = max(
                self.min_batch_size,
                self.current_batch_size * self.aimd_multiplicative_factor,
            )
            self.success_count = 0
