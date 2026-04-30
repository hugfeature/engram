"""One-time setup — download model, init DB, print config."""

import os
import sys
import json


def main():
    print("Engram Setup")
    print("=" * 40)

    data_dir = os.path.join(os.path.expanduser("~"), ".engram")
    os.makedirs(data_dir, exist_ok=True)

    print("\n[1/3] Downloading embedding model (all-mpnet-base-v2)...")
    print("      Using HF mirror: hf-mirror.com")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    try:
        from engram.embedding import embed
        vec = embed("setup test")
        print(f"      Model loaded OK ({len(vec)} dimensions)")
    except Exception as e:
        print(f"      ERROR: {e}", file=sys.stderr)
        print("      Try: HF_ENDPOINT=https://huggingface.co engram-setup")
        sys.exit(1)

    print("\n[2/3] Initializing DuckDB...")
    try:
        from engram.db import MemoryDB
        db = MemoryDB()
        count = db.count()
        print(f"      Database ready at {data_dir}/memories.duckdb ({count} memories)")
    except Exception as e:
        print(f"      ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n[3/3] Initializing graph...")
    try:
        from engram.graph import MemoryGraph
        graph = MemoryGraph()
        print(f"      Graph ready at {data_dir}/graph.json")
    except Exception as e:
        print(f"      ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    import shutil
    exe_path = shutil.which("engram") or ""
    if not exe_path:
        exe_path = "engram"

    print("\n" + "=" * 40)
    print("Setup complete!\n")
    print("Add to your MCP client config:\n")

    config = {
        "mcpServers": {
            "engram": {
                "command": exe_path,
                "env": {
                    "HF_ENDPOINT": "https://hf-mirror.com"
                }
            }
        }
    }
    print(json.dumps(config, indent=2))

    print(f"\nData directory: {data_dir}")
    print(f"Executable:     {exe_path}")


if __name__ == "__main__":
    main()
