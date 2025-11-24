#!/usr/bin/env python3
"""Full demo (Neo4j + OpenAI): index a small text, run a hi-query, print result.

This script will:
 - load `config.yaml` for OpenAI + Neo4j credentials
 - instantiate HiRAG with Neo4jStorage and OpenAI embedding/LLM functions
 - insert a tiny sample document
 - run one query and print the answer

Usage:
    python scripts/full_demo_neo4j_openai.py

Note: this performs real calls to OpenAI and to the configured Neo4j Aura instance.
Make sure `config.yaml` contains valid credentials (already set in this workspace).
"""
import asyncio
import yaml
import logging

from hirag import HiRAG, QueryParam
from openai import AsyncOpenAI
from hirag._storage.gdb_neo4j import Neo4jStorage
from hirag._utils import wrap_embedding_func_with_attrs


logging.basicConfig(level=logging.INFO)


def load_cfg():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_openai_embedding_func(api_key, base_url, model_name="text-embedding-3-small"):
    @wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
    async def openai_embedding(texts: list[str]):
        # Try base_url first, fallback to default AsyncOpenAI if base_url fails
        last_exc = None
        if base_url:
            try:
                client = AsyncOpenAI(api_key=api_key, base_url=base_url)
                resp = await client.embeddings.create(model=model_name, input=texts, encoding_format="float")
                return __import__('numpy').array([d.embedding for d in resp.data])
            except Exception as e:
                last_exc = e

        try:
            client = AsyncOpenAI(api_key=api_key)
            resp = await client.embeddings.create(model=model_name, input=texts, encoding_format="float")
            return __import__('numpy').array([d.embedding for d in resp.data])
        except Exception:
            # raise the first exception if exists, otherwise raise a new error
            if last_exc:
                raise last_exc
            raise

    return openai_embedding


def make_openai_chat_func(api_key, base_url, model_name="gpt-4o-mini"):
    async def openai_chat(prompt, system_prompt=None, history_messages=None, **kwargs):
        # `hashing_kv` is passed by HiRAG (for caching) — remove it before calling the OpenAI client
        hashing_kv = kwargs.pop("hashing_kv", None)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        # Try calling with base_url if provided; if it fails, fall back to default client
        last_exc = None
        if base_url:
            try:
                client = AsyncOpenAI(api_key=api_key, base_url=base_url)
                resp = await client.chat.completions.create(model=model_name, messages=messages, **kwargs)
                return resp.choices[0].message.content
            except Exception as e:
                last_exc = e

        # fallback: try default client and a more widely available model
        try:
            client = AsyncOpenAI(api_key=api_key)
            fallback_model = "gpt-3.5-turbo"
            resp = await client.chat.completions.create(model=fallback_model, messages=messages, **kwargs)
            return resp.choices[0].message.content
        except Exception as e:
            # If both attempts failed, raise the first exception for debugging
            raise last_exc or e

    return openai_chat


async def main():
    cfg = load_cfg()

    openai_cfg = cfg.get("openai", {})
    api_key = openai_cfg.get("api_key")
    base_url = openai_cfg.get("base_url") or None

    hirag_cfg = cfg.get("hirag", {})
    neo4j_url = hirag_cfg.get("neo4j_url")
    neo4j_auth = hirag_cfg.get("neo4j_auth")

    if not api_key:
        print("OpenAI API key missing in config.yaml")
        return
    if not neo4j_url or not neo4j_auth:
        print("Neo4j connection info missing in config.yaml")
        return

    # create embedding and chat functions bound to the configured key/base_url
    embedding_func = make_openai_embedding_func(api_key, base_url)
    chat_func = make_openai_chat_func(api_key, base_url)

    print("Creating HiRAG instance (Neo4j storage + real LLM calls). This may take a while.")
    hr = HiRAG(
        working_dir="./tmp_demo_workdir_full",
        enable_llm_cache=True,
        embedding_func=embedding_func,
        best_model_func=chat_func,
        cheap_model_func=chat_func,
        enable_hierachical_mode=True,
        embedding_batch_num=4,
        embedding_func_max_async=4,
        enable_naive_rag=False,
        graph_storage_cls=Neo4jStorage,
        addon_params={"neo4j_url": neo4j_url, "neo4j_auth": neo4j_auth},
    )

    sample_doc = """
    Alice works at a company that makes electric cars. The company is based in Berlin.
    Bob is a researcher at the University of Cambridge working on battery technology.
    They collaborated on a study about charging networks and published it in 2024.
    """

    # Some Neo4j Aura instances do not have GDS procedures installed (used by clustering).
    # To make this demo work on such instances we replace the clustering method with a no-op.
    async def _noop_clustering(algorithm: str):
        logging.info("Skipping clustering in demo (no GDS available or not needed)")
        return None

    # patch storage to skip heavy GDS clustering if not available
    hr.chunk_entity_relation_graph.clustering = _noop_clustering

    # Some demo Neo4j instances won't have GDS and clustering fields populated.
    # To keep the demo simple we also skip community report generation which
    # expects clustering results (communityIds). Patch generate_community_report
    # to be a no-op for the demo run.
    import hirag._op as _op
    import hirag.hirag as _hirag

    async def _noop_generate_community_report(*args, **kwargs):
        logging.info("Skipping community report generation in demo")
        return None

    # Patch both the module function and the reference imported in the HiRAG module
    _op.generate_community_report = _noop_generate_community_report
    _hirag.generate_community_report = _noop_generate_community_report

    print("Inserting a small document into the graph (will call the extractor & LLMs)")
    # we are in async context - call ainsert directly
    await hr.ainsert(sample_doc)

    print("Running a hierarchical query:")
    q = "What are the main topics discussed and which people and organizations are involved?"
    answer = await hr.aquery(q, param=QueryParam(mode="hi"))
    print("\n--- Query Answer ---\n", answer)


if __name__ == "__main__":
    asyncio.run(main())
