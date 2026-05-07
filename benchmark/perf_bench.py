"""Performance benchmark — store, recall, stats, consolidation at scale.

Usage:
    python -m benchmark.perf_bench [--count 1000] [--iterations 5]

Measures:
  - Store throughput (memories/sec)
  - Recall latency (ms, p50/p95)
  - Stats latency (ms)
  - Consolidation latency (ms)
"""

import argparse
import os
import sys
import time
import tempfile
import statistics

# Add parent to path so we can import engram
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engram.db import MemoryDB
from engram.graph import MemoryGraph
from engram.handlers import handle_store, handle_recall, handle_stats, handle_consolidate
from engram.embedding import embed, get_dimensions


def bench_store(db, graph, count, dim):
    """Store N memories and return timing."""
    fake_emb = [0.01] * dim
    times = []
    for i in range(count):
        category = ["fact", "assumption", "failure", "strategy"][i % 4]
        importance = 0.3 + (i % 7) * 0.1
        t0 = time.perf_counter()
        db.insert(
            f"Benchmark memory {i}: testing performance at scale",
            fake_emb,
            importance=importance,
            category=category,
            user_id="bench",
        )
        times.append(time.perf_counter() - t0)
    return times


def bench_recall(db, graph, dim, iterations=5):
    """Recall with a fake embedding and return latencies."""
    fake_emb = [0.01] * dim
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        db.search_vector(fake_emb, user_id="bench", top_k=10, threshold=0.1)
        times.append(time.perf_counter() - t0)
    return times


def bench_stats(db, iterations=5):
    """Stats via SQL aggregation and return latencies."""
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        handle_stats(db, user_id="bench")
        times.append(time.perf_counter() - t0)
    return times


def bench_consolidation(db, graph, iterations=1):
    """Run consolidation and return latencies."""
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        handle_consolidate(db, graph, user_id="bench")
        times.append(time.perf_counter() - t0)
    return times


def fmt_ms(seconds):
    return f"{seconds * 1000:.1f}ms"


def fmt_ops(seconds_list):
    if not seconds_list:
        return "N/A"
    return f"{1.0 / statistics.mean(seconds_list):.0f}/s"


def stats_summary(times):
    if not times:
        return {"p50": "N/A", "p95": "N/A", "mean": "N/A"}
    sorted_t = sorted(times)
    p50 = sorted_t[len(sorted_t) // 2]
    p95 = sorted_t[int(len(sorted_t) * 0.95)]
    return {
        "p50": fmt_ms(p50),
        "p95": fmt_ms(p95),
        "mean": fmt_ms(statistics.mean(times)),
    }


def main():
    parser = argparse.ArgumentParser(description="Engram performance benchmark")
    parser.add_argument("--count", type=int, default=1000, help="Number of memories to store")
    parser.add_argument("--recall-iters", type=int, default=10, help="Recall iterations")
    parser.add_argument("--stats-iters", type=int, default=10, help="Stats iterations")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "bench.duckdb")
        graph_path = os.path.join(tmpdir, "bench.json")

        db = MemoryDB(db_path)
        graph = MemoryGraph(graph_path)
        dim = db._dim

        print(f"=== Engram Performance Benchmark ===")
        print(f"DB: {db_path}")
        print(f"Embedding dim: {dim}")
        print(f"Memories to store: {args.count}")
        print()

        # Store
        print(f"[1/4] Storing {args.count} memories...")
        store_times = bench_store(db, graph, args.count, dim)
        s = stats_summary(store_times)
        print(f"  Total: {fmt_ms(sum(store_times))}")
        print(f"  Per-insert: p50={s['p50']}, p95={s['p95']}, mean={s['mean']}")
        print(f"  Throughput: {fmt_ops(store_times)}")
        print()

        # Recall
        print(f"[2/4] Recall ({args.recall_iters} iterations)...")
        recall_times = bench_recall(db, graph, dim, args.recall_iters)
        s = stats_summary(recall_times)
        print(f"  Per-recall: p50={s['p50']}, p95={s['p95']}, mean={s['mean']}")
        print()

        # Stats
        print(f"[3/4] Stats ({args.stats_iters} iterations)...")
        stats_times = bench_stats(db, args.stats_iters)
        s = stats_summary(stats_times)
        print(f"  Per-stats: p50={s['p50']}, p95={s['p95']}, mean={s['mean']}")
        print()

        # Consolidation
        print(f"[4/4] Consolidation (1 iteration)...")
        cons_times = bench_consolidation(db, graph, iterations=1)
        if cons_times:
            print(f"  Time: {fmt_ms(cons_times[0])}")
        print()

        # Summary
        total_count = db.count("bench")
        print(f"=== Summary ===")
        print(f"Memories in DB: {total_count}")
        print(f"Store throughput: {fmt_ops(store_times)}")
        print(f"Recall p50: {stats_summary(recall_times)['p50']}")
        print(f"Stats p50: {stats_summary(stats_times)['p50']}")

        db.close()
        graph.flush()


if __name__ == "__main__":
    main()