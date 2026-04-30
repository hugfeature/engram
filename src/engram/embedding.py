"""Sentence-transformer embedding with lazy model loading."""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = logging.getLogger("engram.embedding")

MODEL_NAME = os.environ.get("ENGRAM_MODEL", "all-mpnet-base-v2")

_model: SentenceTransformer | None = None
_model_lock = threading.Lock()
_dimensions: int | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

                from sentence_transformers import SentenceTransformer

                cache_dir = os.path.join(os.path.expanduser("~"), ".engram", "model_cache")
                os.makedirs(cache_dir, exist_ok=True)
                try:
                    _model = SentenceTransformer(MODEL_NAME, cache_folder=cache_dir)
                    _warmup(_model)
                except Exception as e:
                    log.error("Failed to load embedding model %s: %s", MODEL_NAME, e)
                    raise
    return _model


def _warmup(model: "SentenceTransformer"):
    model.encode("warmup", normalize_embeddings=True)


def get_dimensions() -> int:
    global _dimensions
    if _dimensions is None:
        dim_fn = getattr(_get_model(), "get_embedding_dimension",
                         _get_model().get_sentence_embedding_dimension)
        _dimensions = dim_fn()
    return _dimensions


def embed(text: str) -> list[float]:
    model = _get_model()
    try:
        vec = model.encode(text, normalize_embeddings=True)
        return vec.tolist()
    except Exception as e:
        log.error("Embedding encode failed for text (%d chars): %s", len(text), e)
        raise
