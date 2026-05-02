"""Tokenizer abstraction. CharTokenizer (tests) or HFTokenizer (real)."""
from __future__ import annotations

import logging
from typing import List, Optional, Protocol

logger = logging.getLogger("llaisys.tokenizer")


class Tokenizer(Protocol):
    def encode(self, text: str) -> List[int]: ...
    def decode(self, token_ids: List[int]) -> str: ...
    def decode_step(self, accumulated_ids: List[int], new_id: int) -> str: ...
    @property
    def eos_token_id(self) -> Optional[int]: ...


class CharTokenizer:
    """Each char → its codepoint."""

    def encode(self, text: str) -> List[int]:
        return [ord(c) for c in text]

    def decode(self, token_ids: List[int]) -> str:
        out = []
        for t in token_ids:
            try:
                out.append(chr(int(t)))
            except (ValueError, OverflowError):
                out.append("")
        return "".join(out)

    def decode_step(self, accumulated_ids: List[int], new_id: int) -> str:
        try:
            return chr(int(new_id))
        except (ValueError, OverflowError):
            return ""

    @property
    def eos_token_id(self) -> Optional[int]:
        return None


class HFTokenizer:
    """HuggingFace tokenizer with diff-based streaming decode."""

    def __init__(self, hf_tokenizer):
        self._tok = hf_tokenizer

    def encode(self, text: str) -> List[int]:
        # Chat template handles BOS/EOS.
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, token_ids: List[int]) -> str:
        return self._tok.decode(token_ids, skip_special_tokens=True)

    def decode_step(self, accumulated_ids: List[int], new_id: int) -> str:
        # Diff so BPE merges don't fragment the SSE stream.
        prev = self._tok.decode(accumulated_ids, skip_special_tokens=True)
        full = self._tok.decode(accumulated_ids + [int(new_id)],
                                skip_special_tokens=True)
        if full.startswith(prev):
            return full[len(prev):]
        return full

    @property
    def eos_token_id(self) -> Optional[int]:
        eid = getattr(self._tok, "eos_token_id", None)
        return int(eid) if eid is not None else None


def build_tokenizer(tokenizer_path: Optional[str]) -> Tokenizer:
    """HFTokenizer if path + transformers available, else CharTokenizer."""
    if not tokenizer_path:
        logger.info("No tokenizer_path; using CharTokenizer placeholder")
        return CharTokenizer()
    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.warning(
            "transformers not installed; falling back to CharTokenizer")
        return CharTokenizer()
    try:
        hf = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=False)
    except Exception as e:
        logger.error("Failed to load tokenizer from %s: %s; "
                     "falling back to CharTokenizer", tokenizer_path, e)
        return CharTokenizer()
    logger.info("Loaded HF tokenizer from %s (vocab=%s)",
                tokenizer_path, getattr(hf, "vocab_size", "?"))
    return HFTokenizer(hf)
