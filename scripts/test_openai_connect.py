#!/usr/bin/env python3
"""Quick OpenAI connectivity test using project's `config.yaml`.

Usage: python scripts/test_openai_connect.py
"""
import yaml
import sys
from openai import OpenAI


def load_cfg():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_cfg()
    api_key = cfg.get("openai", {}).get("api_key")
    model = cfg.get("openai", {}).get("embedding_model", "text-embedding-ada-002")
    base_url = cfg.get("openai", {}).get("base_url", "https://api.openai.com")

    if not api_key or api_key.startswith("***"):
        print("OpenAI API key not found in config.yaml. Please add it and retry.")
        sys.exit(1)

    # Try using provided base_url first, fall back to default client if it fails
    print("Attempting a single embeddings request with model:", model)
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.embeddings.create(model=model, input=["HiRAG connectivity test"])
        emb = resp.data[0].embedding
        print("✅ Received embedding vector length:", len(emb))
        return
    except Exception as e:
        print("❗ First attempt (with base_url) failed:", e)

    # fallback: try default OpenAI client (no base_url), try embedding-3-small as backup model
    try:
        print("Retrying without base_url using fallback model text-embedding-3-small")
        client = OpenAI(api_key=api_key)
        fallback_model = "text-embedding-3-small"
        resp = client.embeddings.create(model=fallback_model, input=["HiRAG connectivity test"])
        emb = resp.data[0].embedding
        print("✅ Received embedding vector length (fallback):", len(emb))
        return
    except Exception as e:
        print("❌ OpenAI connectivity or request failed (fallback):", e)
        raise


if __name__ == "__main__":
    main()
