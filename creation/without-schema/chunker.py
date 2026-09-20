"""Sentence-aware, token-counted sliding-window text chunker."""
from __future__ import annotations

import re

import tiktoken


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation heuristics.

    Splits after ``.``, ``!``, or ``?`` when followed by whitespace and an
    uppercase letter or quote — a lightweight alternative to NLTK that avoids
    extra dependencies.
    """
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text.strip())
    return [s.strip() for s in raw if s.strip()]


class TextChunker:
    """Splits large text into overlapping, sentence-boundary-aligned chunks.

    Parameters
    ----------
    chunk_size:
        Maximum number of tokens per chunk.
    chunk_overlap:
        Number of overlap tokens carried forward from the previous chunk.
    encoding:
        tiktoken encoding name.  Defaults to ``cl100k_base`` (GPT-4 /
        text-embedding-3 compatible).

    Example
    -------
    >>> chunker = TextChunker(chunk_size=512, chunk_overlap=64)
    >>> chunks = chunker.split(long_text)
    """

    #: Approximate characters per token used when tiktoken cannot load its vocab.
    _CHARS_PER_TOKEN: int = 4

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        encoding: str = "cl100k_base",
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})."
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        try:
            self._enc = tiktoken.get_encoding(encoding)
            self._use_tiktoken = True
        except Exception:
            # Offline / sandboxed environments: fall back to character approximation.
            self._enc = None  # type: ignore[assignment]
            self._use_tiktoken = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(self, text: str) -> list[str]:
        """Return a list of overlapping text chunks, each within *chunk_size* tokens."""
        sentences = _split_sentences(text)
        if not sentences:
            return [text.strip()] if text.strip() else []

        chunks: list[str] = []
        current: list[str] = []
        current_tokens: int = 0

        for sentence in sentences:
            s_tokens = self._count_tokens(sentence)

            # If this sentence alone exceeds the limit, emit it as its own chunk
            if s_tokens >= self.chunk_size:
                if current:
                    chunks.append(" ".join(current))
                    current, current_tokens = [], 0
                chunks.append(sentence)
                continue

            # Flush the current chunk when adding this sentence would overflow
            if current and (current_tokens + s_tokens > self.chunk_size):
                chunks.append(" ".join(current))
                # Carry over a trailing window of sentences up to chunk_overlap tokens
                overlap, overlap_tokens = self._build_overlap(current)
                current, current_tokens = overlap, overlap_tokens

            current.append(sentence)
            current_tokens += s_tokens

        if current:
            chunks.append(" ".join(current))

        return chunks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _count_tokens(self, text: str) -> int:
        if self._use_tiktoken:
            return len(self._enc.encode(text))
        # Fallback: ~4 characters per token (BPE approximation)
        return max(1, len(text) // self._CHARS_PER_TOKEN)

    def _build_overlap(self, sentences: list[str]) -> tuple[list[str], int]:
        """Return the trailing sentences that fit within *chunk_overlap* tokens."""
        overlap: list[str] = []
        tokens: int = 0
        for sentence in reversed(sentences):
            t = self._count_tokens(sentence)
            if tokens + t > self.chunk_overlap:
                break
            overlap.insert(0, sentence)
            tokens += t
        return overlap, tokens
