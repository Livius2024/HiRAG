#!/usr/bin/env python3
"""Quick connectivity test for Neo4j using project's Neo4jStorage class.

This script loads `config.yaml` from the repo, creates a small `global_config`
and instantiates `Neo4jStorage` to call its index_start_callback (which verifies auth & connectivity).

Run this inside the dev container/workspace after installing dependencies:

    python scripts/test_neo4j_connect.py

"""
import asyncio
import yaml
import sys

from hirag._storage.gdb_neo4j import Neo4jStorage


def load_cfg():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


async def main():
    cfg = load_cfg()

    # Prepare a minimal global_config the storage expects
    global_config = {
        "working_dir": cfg.get("hirag", {}).get("working_dir", "./work_dir_test"),
        # cluster params required by Neo4jStorage methods
        "graph_cluster_seed": cfg.get("hirag", {}).get("graph_cluster_seed", 0xDEADBEEF),
        "max_graph_cluster_size": cfg.get("hirag", {}).get("max_graph_cluster_size", 10),
        # addon params used to pass Neo4j URL/auth
        "addon_params": {
            "neo4j_url": cfg.get("hirag", {}).get("neo4j_url", cfg.get("neo4j_url")),
            "neo4j_auth": cfg.get("hirag", {}).get("neo4j_auth", cfg.get("neo4j_auth")),
        },
    }

    print("Using global_config (partial):")
    print(global_config["addon_params"])

    storage = Neo4jStorage(namespace="connectivity_test", global_config=global_config)

    try:
        print("Verifying Neo4j connectivity and authentication...")
        await storage.index_start_callback()
        print("✅ Connected: Neo4j auth & connectivity verified.")
    except Exception as e:
        print("❌ Connectivity check failed:", e)
        raise
    finally:
        try:
            await storage.index_done_callback()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print("Exited with error", exc)
        sys.exit(1)
