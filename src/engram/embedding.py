"""Sentence-transformer embedding with lazy model loading and degradation."""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = logging.getLogger("engram.embedding")

MODEL_NAME = os.environ.get("ENGRAM_MODEL", "all-mpnet-base-v2")
MODEL_LOAD_TIMEOUT = int(os.environ.get("ENGRAM_MODEL_TIMEOUT", "30"))

_model: SentenceTransformer | None = None
_model_lock = threading.Lock()
_dimensions: int | None = None
_degraded = False

# Embedding result cache — OrderedDict for O(1) LRU eviction
_EMBED_CACHE_MAX = 512
_EMBED_CACHE_MAX_KEY_LEN = 1000  # Limit debug key preview length
_embed_cache: OrderedDict[str, list[float]] = OrderedDict()
_cache_lock = threading.Lock()


def is_degraded() -> bool:
    return _degraded


def try_recover() -> bool:
    """Attempt to exit degraded mode if model loads successfully."""
    global _degraded, _model
    with _model_lock:
        if not _degraded:
            return True
        try:
            from sentence_transformers import SentenceTransformer
            cache_dir = os.path.join(os.path.expanduser("~"), ".engram", "model_cache")
            device = os.environ.get("ENGRAM_EMBED_DEVICE", "cpu")
            m = SentenceTransformer(MODEL_NAME, cache_folder=cache_dir, local_files_only=True, device=device)
            m.encode("warmup", normalize_embeddings=True)
            _model = m
            _degraded = False
            log.info("Recovered from degraded mode — model loaded successfully")
            return True
        except Exception as e:
            log.warning("Recovery attempt failed: %s", e)
            return False


def _load_model_in_thread(result: dict, cache_dir: str):
    """Load model inside a thread so we can apply a timeout.
    
    DEPRECATED: This function is no longer used due to ARM SIGBUS crash.
    Model loading is now done sequentially in _get_model().
    """
    try:
        from sentence_transformers import SentenceTransformer
        try:
            m = SentenceTransformer(MODEL_NAME, cache_folder=cache_dir, local_files_only=True)
        except Exception:
            m = SentenceTransformer(MODEL_NAME, cache_folder=cache_dir)
        m.encode("warmup", normalize_embeddings=True)
        result["model"] = m
    except Exception as e:
        result["error"] = e


def _get_model() -> SentenceTransformer | None:
    global _model, _degraded
    if _degraded:
        return None
    if _model is None:
        with _model_lock:
            if _model is None and not _degraded:
                os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
                cache_dir = os.path.join(os.path.expanduser("~"), ".engram", "model_cache")
                os.makedirs(cache_dir, exist_ok=True)

                # FIX: Disable background thread loading to prevent ARM SIGBUS crash
                # Python 3.13 + ARM architecture has C extension concurrency issues
                # when loading embedding model and DuckDB VSS simultaneously
                try:
                    from sentence_transformers import SentenceTransformer
                    # Force CPU to avoid MPS segfault in async/multi-thread context
                    # MPS (Metal) is not thread-safe and crashes under uvloop + background threads
                    device = os.environ.get("ENGRAM_EMBED_DEVICE", "cpu")
                    try:
                        m = SentenceTransformer(MODEL_NAME, cache_folder=cache_dir, local_files_only=True, device=device)
                    except Exception:
                        m = SentenceTransformer(MODEL_NAME, cache_folder=cache_dir, device=device)
                    m.encode("warmup", normalize_embeddings=True)
                    _model = m
                    log.info("Embedding model loaded successfully (sequential mode)")
                except Exception as e:
                    log.error("Model loading failed: %s — entering degraded mode", e)
                    _degraded = True
                    return None
    return _model


def get_dimensions() -> int:
    global _dimensions
    if _dimensions is not None:
        return _dimensions
    with _model_lock:
        if _dimensions is not None:
            return _dimensions
        model = _get_model()
        if model is None:
            _dimensions = 768
        else:
            dim_fn = getattr(
                model, "get_embedding_dimension",
                model.get_sentence_embedding_dimension,
            )
            _dimensions = dim_fn()
    return _dimensions


def _cache_key(text: str) -> str:
    """Stable cache key to avoid collisions on truncated long text."""
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    preview = text[:_EMBED_CACHE_MAX_KEY_LEN]
    return f"{digest}:{preview}"


def embed(text: str) -> list[float]:
    cache_key = _cache_key(text)
    with _cache_lock:
        cached = _embed_cache.get(cache_key)
        if cached is not None:
            _embed_cache.move_to_end(cache_key)
            return cached
    model = _get_model()
    if model is None:
        log.warning("Embedding in degraded mode — returning zero vector")
        return [0.0] * get_dimensions()
    try:
        vec = model.encode(text, normalize_embeddings=True)
        result = vec.tolist()
        with _cache_lock:
            _embed_cache[cache_key] = result
            _embed_cache.move_to_end(cache_key)
            while len(_embed_cache) > _EMBED_CACHE_MAX:
                _embed_cache.popitem(last=False)
        return result
    except Exception as e:
        log.error("Embedding encode failed for text (%d chars): %s", len(text), e)
        return [0.0] * get_dimensions()
