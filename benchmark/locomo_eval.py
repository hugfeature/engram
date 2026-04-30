#!/usr/bin/env python3
"""Engram × LoCoMo Benchmark — evaluate retrieval quality against the
LoCoMo long-term conversational memory dataset (Snap Research).

Usage:
    # retrieval-only (no LLM needed)
    python benchmark/locomo_eval.py --mode turn --top-k 5

    # with LLM scoring
    python benchmark/locomo_eval.py --mode observation --top-k 5 \
        --llm GLM-5.1 --base-url https://api.example.com/v1

    # with API-based embedding model
    python benchmark/locomo_eval.py --mode turn --top-k 5 \
        --llm DeepSeek-V3.2 --base-url https://api.example.com/v1 \
        --embed-model bge-m3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")

DATA_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)
DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
CATEGORY_NAMES = {1: "Multi-Hop", 2: "Temporal", 3: "Open-Domain", 4: "Single-Hop"}

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from engram.db import MemoryDB
from engram.graph import MemoryGraph
from engram.embedding import embed as local_embed
from engram.retrieve import recall
import engram.embedding as _emb_module
import engram.retrieve as _ret_module


# ---------------------------------------------------------------------------
# API-based embedding (for bge-m3, Qwen3-Embedding, etc.)
# ---------------------------------------------------------------------------

_api_embed_fn = None
_api_embed_dim = None


_BATCH_SIZE = 10
_MAX_RETRIES = 5


def _api_call_with_retry(client, model: str, texts: list[str]) -> list[list[float]]:
    """Call embedding API with exponential backoff on 429."""
    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait = 2 ** attempt * 3
                print(f"    429 RPM limit, retry in {wait}s (attempt {attempt+1}/{_MAX_RETRIES})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Embedding API failed after {_MAX_RETRIES} retries")


def setup_api_embed(model: str, base_url: str, api_key: str):
    """Configure API-based embedding and detect dimensionality."""
    global _api_embed_fn, _api_embed_dim
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key)

    def _call(text: str) -> list[float]:
        return _api_call_with_retry(client, model, [text])[0]

    def _call_batch(texts: list[str]) -> list[list[float]]:
        all_vecs = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i:i + _BATCH_SIZE]
            all_vecs.extend(_api_call_with_retry(client, model, batch))
        return all_vecs

    probe = _call("hello")
    _api_embed_dim = len(probe)
    _api_embed_fn = _call
    _api_embed_fn.batch = _call_batch
    print(f"  API embed: {model} → {_api_embed_dim} dimensions (batch={_BATCH_SIZE})")

    _emb_module.embed = _call
    _emb_module.DIMENSIONS = _api_embed_dim
    _ret_module.embed = _call


def embed(text: str) -> list[float]:
    if _api_embed_fn:
        return _api_embed_fn(text)
    return local_embed(text)


def embed_batch(texts: list[str]) -> list[list[float]]:
    if _api_embed_fn and hasattr(_api_embed_fn, 'batch'):
        return _api_embed_fn.batch(texts)
    return [embed(t) for t in texts]


# ---------------------------------------------------------------------------
# Reranker (local cross-encoder, two-stage retrieval)
# ---------------------------------------------------------------------------

_reranker = None
_rerank_model_name = None


def setup_reranker(model_name: str = "BAAI/bge-reranker-v2-m3"):
    global _reranker, _rerank_model_name
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from sentence_transformers import CrossEncoder
    print(f"  Loading reranker: {model_name} …")
    _reranker = CrossEncoder(model_name)
    _rerank_model_name = model_name
    print(f"  Reranker ready")


def rerank(query: str, results: list, top_k: int) -> list:
    if not _reranker or not results:
        return results[:top_k]
    pairs = [(query, r.content) for r in results]
    scores = _reranker.predict(pairs)
    ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    return [r for r, _ in ranked[:top_k]]


# ---------------------------------------------------------------------------
# Flexible-dimension MemoryDB (overrides FLOAT[768] hardcodes)
# ---------------------------------------------------------------------------

class FlexDimDB(MemoryDB):
    """MemoryDB subclass that supports arbitrary embedding dimensions."""

    def __init__(self, db_path: str, dim: int = 768):
        self.dim = dim
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        import duckdb
        self.conn = duckdb.connect(db_path)
        self._init_flex_schema()

    def _init_flex_schema(self):
        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS memory_id_seq START 1")
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY DEFAULT nextval('memory_id_seq'),
                user_id VARCHAR DEFAULT 'default',
                content TEXT NOT NULL,
                embedding FLOAT[{self.dim}],
                importance FLOAT NOT NULL DEFAULT 0.5,
                category VARCHAR NOT NULL DEFAULT 'fact',
                recall_count INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP NOT NULL DEFAULT now(),
                last_accessed_at TIMESTAMP NOT NULL DEFAULT now()
            )
        """)
        try:
            self.conn.execute("INSTALL fts; LOAD fts;")
        except Exception:
            pass

    def insert(self, content, embedding, importance=0.5, category="fact", user_id="default"):
        result = self.conn.execute(
            f"""INSERT INTO memories (user_id, content, embedding, importance, category)
            VALUES (?, ?, ?::FLOAT[{self.dim}], ?, ?) RETURNING id""",
            [user_id, content, embedding, importance, category],
        ).fetchone()
        return result[0]

    def update(self, memory_id, content, embedding, importance=None):
        if importance is not None:
            self.conn.execute(
                f"""UPDATE memories SET content=?, embedding=?::FLOAT[{self.dim}],
                importance=?, last_accessed_at=now() WHERE id=?""",
                [content, embedding, importance, memory_id],
            )
        else:
            self.conn.execute(
                f"""UPDATE memories SET content=?, embedding=?::FLOAT[{self.dim}],
                last_accessed_at=now() WHERE id=?""",
                [content, embedding, memory_id],
            )

    def search_vector(self, query_embedding, user_id="default", top_k=20, threshold=0.20):
        rows = self.conn.execute(
            f"""SELECT id, user_id, content, importance, category,
                   recall_count, created_at, last_accessed_at,
                   array_cosine_similarity(embedding, ?::FLOAT[{self.dim}]) AS sim
            FROM memories WHERE user_id=?
              AND array_cosine_similarity(embedding, ?::FLOAT[{self.dim}]) >= ?
            ORDER BY sim DESC LIMIT ?""",
            [query_embedding, user_id, query_embedding, threshold, top_k],
        ).fetchall()
        from engram.db import MemoryRow
        results = []
        for r in rows:
            m = MemoryRow(*r[:8])
            m.similarity = r[8]
            results.append(m)
        return results


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def ensure_dataset() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "locomo10.json"
    if path.exists():
        return path
    print(f"Downloading locomo10.json …")
    import subprocess
    subprocess.run(["curl", "-sL", "-o", str(path), DATA_URL], check=True)
    print(f"  saved to {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def load_dataset(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def extract_turns(conv: dict) -> list[str]:
    """Flatten all session turns into memory strings."""
    turns = []
    for key in sorted(conv.keys()):
        if not key.startswith("session_") or key.endswith(("_date_time", "_observation")):
            continue
        session = conv[key]
        if not isinstance(session, list):
            continue
        dt_key = f"{key}_date_time"
        dt_str = conv.get(dt_key, "")
        for turn in session:
            speaker = turn.get("speaker", "Unknown")
            text = turn.get("text", "")
            dia_id = turn.get("dia_id", "")
            content = f"[{dt_str}] {speaker}: {text}"
            turns.append((dia_id, content))
    return turns


def extract_observations(sample: dict) -> list[str]:
    """Extract dataset-provided observation facts from top-level 'observation' key."""
    obs_data = sample.get("observation", {})
    obs = []
    for session_key in sorted(obs_data.keys()):
        session = obs_data[session_key]
        if isinstance(session, dict):
            for speaker, facts in session.items():
                if isinstance(facts, list):
                    for item in facts:
                        if isinstance(item, list) and len(item) >= 1:
                            obs.append(f"{speaker}: {item[0]}")
                        elif isinstance(item, str):
                            obs.append(f"{speaker}: {item}")
    return obs


def make_db(tmpdir: str) -> MemoryDB:
    """Create a DB with the right dimensions for the current embedding model."""
    dim = _api_embed_dim or 768
    if dim != 768:
        return FlexDimDB(db_path=os.path.join(tmpdir, "mem.duckdb"), dim=dim)
    return MemoryDB(db_path=os.path.join(tmpdir, "mem.duckdb"))


def ingest(contents: list[str], db: MemoryDB, graph: MemoryGraph) -> dict[int, list[float]]:
    """Embed and insert a list of memory strings. Returns {id: embedding}."""
    all_embs: dict[int, list[float]] = {}
    for i, content in enumerate(contents):
        vec = embed(content)
        mid = db.insert(content, vec, importance=1.0, category="fact")
        graph.index_memory(mid, vec, all_embs, importance=1.0, category="fact")
        all_embs[mid] = vec
        if (i + 1) % 100 == 0:
            print(f"  ingested {i+1}/{len(contents)}", flush=True)
    return all_embs


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Lower text, remove punctuation/articles/whitespace (LoCoMo official)."""
    s = str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall_score = n_common / len(gold_tokens)
    return 2 * precision * recall_score / (precision + recall_score)


def check_evidence_hit(retrieved: list, evidence_ids: list[str], turn_id_map: dict[str, int]) -> bool:
    """Check if any evidence turn was retrieved in top-k."""
    retrieved_ids = {r.id for r in retrieved}
    for eid in evidence_ids:
        if eid in turn_id_map and turn_id_map[eid] in retrieved_ids:
            return True
    return False


# ---------------------------------------------------------------------------
# Query expansion (LLM-based, for multi-hop improvement)
# ---------------------------------------------------------------------------

_query_expander = None


def setup_query_expander(model: str, base_url: str, api_key: str):
    global _query_expander
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key)

    def expand(question: str) -> list[str]:
        messages = [
            {"role": "system", "content": (
                "Generate 3 short, diverse search queries to find relevant conversation turns "
                "for answering the given question. Focus on different entities, time periods, "
                "or aspects mentioned. Output one query per line, no numbering or bullets."
            )},
            {"role": "user", "content": question},
        ]
        resp = client.chat.completions.create(
            model=model, messages=messages, max_tokens=200, temperature=0.3,
        )
        text = resp.choices[0].message.content or ""
        queries = [q.strip() for q in text.strip().split('\n') if q.strip()]
        return queries[:3]

    _query_expander = expand
    print(f"  Query expansion enabled (LLM: {model})")


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def make_llm_fn(model: str, base_url: str, api_key: str):
    """Return a function that calls an OpenAI-compatible chat API."""
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key)

    def call(question: str, context: str) -> str:
        messages = [
            {"role": "system", "content": (
                "You are a helpful assistant answering questions about past conversations. "
                "Use ONLY the provided context to answer. Be concise — answer in a few words or a short phrase. "
                "If the context doesn't contain the answer, say 'I don't know'."
            )},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"},
        ]
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=512,
            temperature=0,
        )
        content = resp.choices[0].message.content or ""
        return content.strip()

    return call


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(
    dataset: list[dict],
    mode: str = "turn",
    top_k: int = 5,
    llm_fn=None,
    dry_run: bool = False,
    max_conv: int | None = None,
):
    cat_f1: dict[int, list[float]] = defaultdict(list)
    cat_hits: dict[int, list[bool]] = defaultdict(list)
    total_qa = 0
    skipped = 0

    n_conv = max_conv or len(dataset)
    for ci, sample in enumerate(dataset[:n_conv]):
        conv = sample.get("conversation", {})
        qas = sample.get("qa", [])
        sample_id = sample.get("sample_id", ci)

        print(f"\n{'='*60}")
        print(f"Conversation {ci+1}/{n_conv} (sample_id={sample_id})")

        with tempfile.TemporaryDirectory() as tmpdir:
            db = make_db(tmpdir)
            graph = MemoryGraph(graph_path=os.path.join(tmpdir, "graph.pkl"))

            # Ingest
            turn_id_map: dict[str, int] = {}
            if mode == "turn":
                turns = extract_turns(conv)
                dia_ids = [t[0] for t in turns]
                contents = [t[1] for t in turns]
                print(f"  Ingesting {len(turns)} turns (batch embed) …")
                all_vecs = embed_batch(contents)
                all_embs: dict[int, list[float]] = {}
                for i, (dia_id, content, vec) in enumerate(zip(dia_ids, contents, all_vecs)):
                    mid = db.insert(content, vec, importance=1.0, category="fact")
                    graph.index_memory(mid, vec, all_embs, importance=1.0, category="fact")
                    all_embs[mid] = vec
                    turn_id_map[dia_id] = mid
                print(f"    {len(turns)}/{len(turns)} done", flush=True)
            else:
                obs = extract_observations(sample)
                print(f"  Ingesting {len(obs)} observations (batch embed) …")
                all_vecs = embed_batch(obs)
                all_embs: dict[int, list[float]] = {}
                for i, (content, vec) in enumerate(zip(obs, all_vecs)):
                    mid = db.insert(content, vec, importance=1.0, category="fact")
                    graph.index_memory(mid, vec, all_embs, importance=1.0, category="fact")
                    all_embs[mid] = vec
                print(f"    {len(obs)}/{len(obs)} done", flush=True)

            mem_count = db.count()
            print(f"  Memory count: {mem_count}")

            # Evaluate QAs
            qa_count = 0
            for qa in qas:
                cat = qa.get("category", 0)
                if cat == 5:
                    skipped += 1
                    continue

                question = qa["question"]
                answer = str(qa["answer"])
                evidence = qa.get("evidence", [])

                recall_k = top_k * 10 if _reranker else top_k

                if _query_expander:
                    sub_queries = _query_expander(question)
                    merged: dict[int, object] = {}
                    for q in [question] + sub_queries:
                        for r in recall(q, db, graph, top_k=recall_k):
                            if r.id not in merged or r.score > merged[r.id].score:
                                merged[r.id] = r
                    results = list(merged.values())
                else:
                    results = recall(question, db, graph, top_k=recall_k)

                if _reranker:
                    results = rerank(question, results, top_k)
                else:
                    results = sorted(results, key=lambda r: r.score, reverse=True)[:top_k]
                context = "\n".join(r.content for r in results)

                # Evidence hit
                hit = check_evidence_hit(results, evidence, turn_id_map) if mode == "turn" else False
                cat_hits[cat].append(hit)

                # F1
                if llm_fn:
                    try:
                        prediction = llm_fn(question, context)
                    except Exception as e:
                        prediction = ""
                        print(f"    LLM error: {e}")
                else:
                    prediction = context

                f1 = compute_f1(prediction, answer)
                cat_f1[cat].append(f1)
                total_qa += 1
                qa_count += 1

                if dry_run and total_qa >= 10:
                    break

            print(f"  Evaluated {qa_count} QAs")
            if dry_run and total_qa >= 10:
                break

    return cat_f1, cat_hits, total_qa, skipped


def print_results(cat_f1, cat_hits, total_qa, skipped, mode, top_k, llm_name, embed_model=None):
    print(f"\n{'='*60}")
    print(f"=== Engram × LoCoMo Benchmark ===")
    embed_label = embed_model or "all-mpnet-base-v2 (local)"
    print(f"Mode: {mode} | Top-K: {top_k} | LLM: {llm_name or 'none (retrieval-only)'}")
    print(f"Embed: {embed_label} ({_api_embed_dim or 768}d)")
    if _rerank_model_name:
        print(f"Reranker: {_rerank_model_name} (recall {top_k*10} → rerank {top_k})")
    if _query_expander:
        print(f"Query expansion: enabled (original + 3 sub-queries)")
    print(f"Total QA evaluated: {total_qa} | Skipped (adversarial): {skipped}")
    print()
    print(f"{'Category':<16} {'Count':>6} {'F1':>8} {'Hit@'+str(top_k):>8}")
    print("─" * 42)

    all_f1 = []
    for cat in sorted(CATEGORY_NAMES.keys()):
        name = CATEGORY_NAMES[cat]
        f1_list = cat_f1.get(cat, [])
        hit_list = cat_hits.get(cat, [])
        avg_f1 = sum(f1_list) / len(f1_list) if f1_list else 0
        hit_rate = sum(hit_list) / len(hit_list) * 100 if hit_list else 0
        all_f1.extend(f1_list)
        print(f"{name:<16} {len(f1_list):>6} {avg_f1:>8.4f} {hit_rate:>7.1f}%")

    print("─" * 42)
    overall = sum(all_f1) / len(all_f1) if all_f1 else 0
    all_hits = []
    for h in cat_hits.values():
        all_hits.extend(h)
    overall_hit = sum(all_hits) / len(all_hits) * 100 if all_hits else 0
    print(f"{'Overall':<16} {len(all_f1):>6} {overall:>8.4f} {overall_hit:>7.1f}%")
    print()

    return {
        "mode": mode,
        "top_k": top_k,
        "llm": llm_name,
        "embed_model": embed_model or "all-mpnet-base-v2",
        "embed_dim": _api_embed_dim or 768,
        "reranker": _rerank_model_name,
        "query_expand": _query_expander is not None,
        "total_qa": total_qa,
        "categories": {
            CATEGORY_NAMES[cat]: {
                "count": len(cat_f1.get(cat, [])),
                "f1": sum(cat_f1.get(cat, [])) / len(cat_f1[cat]) if cat_f1.get(cat) else 0,
                "hit_rate": sum(cat_hits.get(cat, [])) / len(cat_hits[cat]) if cat_hits.get(cat) else 0,
            }
            for cat in sorted(CATEGORY_NAMES.keys())
        },
        "overall_f1": overall,
        "overall_hit_rate": overall_hit / 100,
    }


def main():
    parser = argparse.ArgumentParser(description="Engram × LoCoMo Benchmark")
    parser.add_argument("--mode", choices=["turn", "observation"], default="turn")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--llm", type=str, default=None, help="LLM model name")
    parser.add_argument("--base-url", type=str, default=None, help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", type=str, default=None, help="API key (default: $OPENAI_API_KEY)")
    parser.add_argument("--dry-run", action="store_true", help="Only evaluate first 10 questions")
    parser.add_argument("--max-conv", type=int, default=None, help="Limit number of conversations")
    parser.add_argument("--embed-model", type=str, default=None,
                        help="API embedding model (e.g. bge-m3, Qwen3-Embedding-4B)")
    parser.add_argument("--rerank", action="store_true", help="Enable two-stage reranking")
    parser.add_argument("--rerank-model", type=str, default="BAAI/bge-reranker-v2-m3",
                        help="CrossEncoder reranker model")
    parser.add_argument("--query-expand", action="store_true",
                        help="Enable LLM query expansion (requires --llm)")
    args = parser.parse_args()

    dataset_path = ensure_dataset()
    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} conversations")

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    base_url = args.base_url or "https://api.openai.com/v1"

    if args.embed_model:
        if not api_key:
            print("Error: --embed-model requires --api-key or $OPENAI_API_KEY")
            sys.exit(1)
        setup_api_embed(args.embed_model, base_url, api_key)

    if args.rerank:
        setup_reranker(args.rerank_model)

    if args.query_expand:
        if not args.llm:
            print("Error: --query-expand requires --llm")
            sys.exit(1)
        setup_query_expander(args.llm, base_url, api_key)

    llm_fn = None
    if args.llm:
        if not api_key:
            print("Error: --llm requires --api-key or $OPENAI_API_KEY")
            sys.exit(1)
        print(f"Using LLM: {args.llm} @ {base_url}")
        llm_fn = make_llm_fn(args.llm, base_url, api_key)

    t0 = time.time()
    cat_f1, cat_hits, total_qa, skipped = run_benchmark(
        dataset, args.mode, args.top_k, llm_fn, args.dry_run, args.max_conv
    )
    elapsed = time.time() - t0

    results = print_results(cat_f1, cat_hits, total_qa, skipped, args.mode, args.top_k, args.llm, args.embed_model)
    results["elapsed_seconds"] = round(elapsed, 1)

    print(f"Time: {elapsed:.0f}s")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"locomo_{args.mode}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved: {out_path}")


if __name__ == "__main__":
    main()
