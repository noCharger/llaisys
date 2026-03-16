import asyncio
import logging
from typing import List, Any
logger = logging.getLogger("llaisys.model_service")

class ModelService:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.is_loaded = True

    async def forward_batch(self, batch: List[Any]) -> List[Any]:
        await asyncio.sleep(0.05)
        results = []
        for job in batch:

            response_text = f"Hello from AI (batched). You requested max {job.max_tokens} tokens. "
            if job.max_tokens and job.max_tokens > 100:
                response_text += "Here is a much longer and more detailed explanation because you allowed more tokens. " * 5
            results.append(response_text)
        return results

    def forward(self, session_ptr: Any, input_ids: List[int], temperature: float, top_p: float, top_k: int) -> str:
        return "Hello from AI (sequential)"
