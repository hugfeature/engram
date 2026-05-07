"""Sentence-transformer embedding with lazy model loading and degradation."""

from __future__ import annotations

import hashlib
import logging
import os
import threading
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

# Embedding result cache — avoids re-encoding identical text (e.g. during consolidation)
_EMBED_CACHE_MAX = 512
_EMBED_CACHE_MAX_KEY_LEN = 1000  # Limit debug key preview length
_embed_cache: dict[str, list[float]] = {}
_embed_cache_order: list[str] = []
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
            m = SentenceTransformer(MODEL_NAME, cache_folder=cache_dir, local_files_only=True)
            m.encode("warmup", normalize_embeddings=True)
            _model = m
            _degraded = False
            log.info("Recovered from degraded mode — model loaded successfully")
            return True
        except Exception as e:
            log.warning("Recovery attempt failed: %s", e)
            return False


def _load_model_in_thread(result: dict, cache_dir: str):
    """Load model inside a thread so we can apply a timeout."""
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

                result: dict = {}
                loader = threading.Thread(
                    target=_load_model_in_thread, args=(result, cache_dir), daemon=True
                )
                loader.start()
                loader.join(timeout=MODEL_LOAD_TIMEOUT)

                if loader.is_alive():
                    log.error(
                        "Model loading timed out after %ds — entering degraded mode",
                        MODEL_LOAD_TIMEOUT,
                    )
                    _degraded = True
                    # Don't return immediately — the daemon thread may still finish.
                    # If it completes later, try_recover() can restore normal mode.
                    return None

                if "error" in result:
                    log.error("Model loading failed: %s — entering degraded mode", result["error"])
                    _degraded = True
                    return None

                _model = result.get("model")
                if _model is None:
                    _degraded = True
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
            if cache_key in _embed_cache_order:
                _embed_cache_order.remove(cache_key)
            _embed_cache_order.append(cache_key)
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
            if cache_key in _embed_cache_order:
                _embed_cache_order.remove(cache_key)
            _embed_cache_order.append(cache_key)
            while len(_embed_cache) > _EMBED_CACHE_MAX and _embed_cache_order:
                oldest = _embed_cache_order.pop(0)
                _embed_cache.pop(oldest, None)
        return result
    except Exception as e:
        log.error("Embedding encode failed for text (%d chars): %s", len(text), e)
        return [0.0] * get_dimensions()
