"""Mooncake-style PD-disaggregated scheduler.

Two cooperating loops share one paged KV store:

  PrefillWorker:
    Pulls jobs whose prefill is not done. Packs them into one prefill
    forward bounded by token_budget. Commits each finished prompt so its
    pages enter the global prefix index, then publishes a handoff.

  DecodeWorker:
    Pulls handed-off jobs, runs single-token decode forwards in batches
    bounded by max_batch_size, streams tokens until EOS or max_tokens.

The two workers share a KVStoreService that wraps a single paged pool.
Because the underlying prefix index is global content-addressed, the
Prefill-to-Decode handoff transfers only a block_id; the KV pages are
the same physical memory both workers see.

Contrast with combined batching ClipperScheduler in scheduler.py:
that one packs prefill chunks and decodes in one varlen forward to
avoid PD interference at the algorithmic level. PD disaggregation
solves the same interference at the system level by running prefill
and decode on separate workers possibly on separate hardware.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .kv_store_service import KVStoreService, PrefillHandoff
from .scheduler import RequestJob, Phase, DONE

logger = logging.getLogger("llaisys.pd_scheduler")


@dataclass
class PDStats:
    prefill_steps: int = 0
    decode_steps: int = 0
    handoffs: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    prefill_active: int = 0
    decode_active: int = 0


class PDScheduler:
    """Drives a PrefillWorker and a DecodeWorker that share a KV store.

    Use this as a drop-in alternative to ClipperScheduler when you want
    PD disaggregation. The public surface is the same:
      submit(job), stop(), stats(), set_eos_token(t).
    """

    def __init__(self, model_service: Any, kv_store: KVStoreService,
                 token_budget: int = 4096,
                 decode_batch_size: int = 16,
                 batch_wait_ms: float = 2.0,
                 eos_token: Optional[int] = None):
        self.model_service = model_service
        self.kv_store = kv_store
        self.token_budget = token_budget
        self.decode_batch_size = decode_batch_size
        self.batch_wait_s = batch_wait_ms / 1000.0
        self.eos_token = eos_token

        self._prefill_queue: asyncio.Queue[RequestJob] = asyncio.Queue()
        self._decode_queue: asyncio.Queue[RequestJob] = asyncio.Queue()

        self._prefill_active: Dict[str, RequestJob] = {}
        self._decode_active: Dict[str, RequestJob] = {}

        self._wake_prefill = asyncio.Event()
        self._wake_decode = asyncio.Event()

        self.running = False
        self._prefill_task: Optional[asyncio.Task] = None
        self._decode_task: Optional[asyncio.Task] = None
        self._stats = PDStats()

    # ---------- lifecycle ----------

    async def start(self):
        self.running = True
        self._prefill_task = asyncio.create_task(self._prefill_loop())
        self._decode_task = asyncio.create_task(self._decode_loop())

    async def stop(self):
        self.running = False
        self._wake_prefill.set()
        self._wake_decode.set()
        for t in (self._prefill_task, self._decode_task):
            if t is None:
                continue
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass

    def set_eos_token(self, eos_token: int):
        self.eos_token = int(eos_token)

    async def submit(self, job: RequestJob):
        if not job.tenant_id:
            raise ValueError("RequestJob.tenant_id is required")
        await self._prefill_queue.put(job)
        self._wake_prefill.set()

    def stats(self) -> Dict[str, Any]:
        return {
            "prefill_steps": self._stats.prefill_steps,
            "decode_steps": self._stats.decode_steps,
            "handoffs": self._stats.handoffs,
            "prefill_tokens": self._stats.prefill_tokens,
            "decode_tokens": self._stats.decode_tokens,
            "prefill_active": len(self._prefill_active),
            "decode_active": len(self._decode_active),
            "store": self.kv_store.usage(),
        }

    # ---------- Prefill worker ----------

    async def _prefill_loop(self):
        while self.running:
            try:
                await self._prefill_tick()
            except Exception as e:
                logger.error("prefill loop error: %s", e, exc_info=True)
                await asyncio.sleep(0.05)

    async def _prefill_tick(self):
        # Drain pending submissions into _prefill_active, bounded by capacity.
        await self._drain_into_prefill()
        if not self._prefill_active:
            self._wake_prefill.clear()
            try:
                await asyncio.wait_for(self._wake_prefill.wait(), timeout=0.1)
            except asyncio.TimeoutError:
                pass
            return

        # Pack a prefill batch under token_budget. FIFO by created_at.
        batch: List[RequestJob] = []
        remaining = self.token_budget
        for job in sorted(self._prefill_active.values(),
                          key=lambda j: j.created_at):
            if remaining <= 0:
                break
            unfed = len(job.input_ids) - job.kv_pos
            if unfed <= 0:
                continue
            feed = min(unfed, remaining)
            job.pending_input_ids = list(
                job.input_ids[job.kv_pos: job.kv_pos + feed])
            batch.append(job)
            remaining -= feed

        if not batch:
            await asyncio.sleep(self.batch_wait_s)
            return

        try:
            results = await self.model_service.forward_batch(batch)
        except Exception as e:
            for j in batch:
                self._fail(j, e)
            return

        self._stats.prefill_steps += 1
        for j, tok in zip(batch, results):
            self._stats.prefill_tokens += len(j.pending_input_ids)
            j.kv_pos += len(j.pending_input_ids)
            if j.is_prefilling:
                # Mid-prefill chunk, sample is junk, keep going next tick.
                continue
            # Prefill is done for this job. Commit, hand off to decode.
            self._handoff_to_decode(j, int(tok))

    async def _drain_into_prefill(self):
        # Pull at least one if idle; otherwise drain whatever's queued.
        if not self._prefill_active:
            try:
                job = await asyncio.wait_for(
                    self._prefill_queue.get(),
                    timeout=self.batch_wait_s)
                self._admit_for_prefill(job)
            except asyncio.TimeoutError:
                return
        while True:
            try:
                job = self._prefill_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._admit_for_prefill(job)

    def _admit_for_prefill(self, job: RequestJob):
        bid, matched = self.kv_store.reserve_prefill(
            job.tenant_id, job.input_ids)
        if bid < 0:
            self._fail(job, RuntimeError("KV store exhausted on prefill"))
            return
        # Leave at least one token unfed so a sample lands at prompt end.
        if matched >= len(job.input_ids):
            matched = max(0, len(job.input_ids) - 1)
        job.block_id = bid
        job.kv_pos = matched
        job.phase = Phase.PREFILL
        self._prefill_active[job.request_id] = job

    def _handoff_to_decode(self, job: RequestJob, first_token: int):
        # Commit the prompt so the global prefix index holds these pages.
        # Future requests with the same prompt will hit them directly.
        try:
            self.kv_store.commit_prefill(
                job.tenant_id, job.block_id, job.kv_pos,
                list(job.input_ids))
        except Exception as e:
            self._fail(job, e)
            return

        # Apply the first token as the prefill output.
        job.output_tokens.append(first_token)
        job.total_output_count += 1
        job.phase = Phase.DECODE
        try:
            job.tokens_queue.put_nowait(first_token)
        except asyncio.QueueFull:
            logger.warning("tokens_queue full for %s", job.request_id)

        # EOS or budget reached at the very first token, no decode needed.
        if self._should_finish(job, first_token):
            self._finish(job)
            return

        self._stats.handoffs += 1
        del self._prefill_active[job.request_id]

        handoff = PrefillHandoff(
            block_id=job.block_id,
            prefilled_tokens=job.kv_pos,
            first_decode_token=first_token,
        )
        self.kv_store.publish_handoff(job.request_id, handoff)
        # In our single-process design the Decode worker can read the job
        # directly. The handoff record above is mainly for observability
        # and would carry the block_id across a network boundary in a
        # true two-process deployment.
        self._decode_queue.put_nowait(job)
        self._wake_decode.set()

    # ---------- Decode worker ----------

    async def _decode_loop(self):
        while self.running:
            try:
                await self._decode_tick()
            except Exception as e:
                logger.error("decode loop error: %s", e, exc_info=True)
                await asyncio.sleep(0.05)

    async def _decode_tick(self):
        # Pull queued jobs into _decode_active.
        while True:
            try:
                job = self._decode_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self.kv_store.take_handoff(job.request_id)
            self._decode_active[job.request_id] = job

        if not self._decode_active:
            self._wake_decode.clear()
            try:
                await asyncio.wait_for(self._wake_decode.wait(), timeout=0.1)
            except asyncio.TimeoutError:
                pass
            return

        batch = list(self._decode_active.values())[: self.decode_batch_size]
        for j in batch:
            last = j.output_tokens[-1] if j.output_tokens else j.input_ids[-1]
            j.pending_input_ids = [int(last)]

        try:
            results = await self.model_service.forward_batch(batch)
        except Exception as e:
            for j in batch:
                self._fail(j, e)
            return

        self._stats.decode_steps += 1
        for j, tok in zip(batch, results):
            j.kv_pos += 1
            self._stats.decode_tokens += 1
            tok_int = int(tok)
            j.output_tokens.append(tok_int)
            j.total_output_count += 1
            try:
                j.tokens_queue.put_nowait(tok_int)
            except asyncio.QueueFull:
                logger.warning("tokens_queue full for %s", j.request_id)
            if self._should_finish(j, tok_int):
                self._finish(j)

    def _should_finish(self, job: RequestJob, tok: int) -> bool:
        if self.eos_token is not None and tok == int(self.eos_token):
            job.finish_reason = "stop"
            return True
        if job.total_output_count >= job.max_tokens:
            job.finish_reason = "length"
            return True
        return False

    def _finish(self, job: RequestJob):
        # Commit the full history so a later resubmit can prefix-recover.
        try:
            full = list(job.input_ids) + list(job.output_tokens)
            self.kv_store.commit_prefill(
                job.tenant_id, job.block_id, job.kv_pos, full)
        except Exception:
            pass
        try:
            self.kv_store.release(job.tenant_id, job.block_id)
        except Exception:
            pass
        job.finished = True
        try:
            job.tokens_queue.put_nowait(DONE)
        except Exception:
            pass
        if not job.future.done():
            job.future.set_result({
                "tokens": list(job.output_tokens),
                "finish_reason": job.finish_reason or "stop",
            })
        self._prefill_active.pop(job.request_id, None)
        self._decode_active.pop(job.request_id, None)

    def _fail(self, job: RequestJob, error: Exception):
        job.finished = True
        job.finish_reason = "error"
        if job.block_id is not None:
            try:
                self.kv_store.release(job.tenant_id, job.block_id)
            except Exception:
                pass
        if not job.future.done():
            job.future.set_exception(error)
        try:
            job.tokens_queue.put_nowait(DONE)
        except Exception:
            pass
        self._prefill_active.pop(job.request_id, None)
        self._decode_active.pop(job.request_id, None)
