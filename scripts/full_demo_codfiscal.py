#!/usr/bin/env python3
"""Index your `codfiscal.txt` and run a hierarchical query.

Safe defaults: runs in 'quick' mode which indexes only the first chunk or
a limited number of chunks to avoid large LLM costs and long runtimes.

Usage:
  python scripts/full_demo_codfiscal.py [--mode quick|full] [--limit N] [--query "..."] [--skip-cluster]

Examples:
  # Quick test (default) — index up to 1 chunk and run one query (fast, cheap)
  python scripts/full_demo_codfiscal.py

  # Full run (may be slow/costly) — requires explicit --mode full
  python scripts/full_demo_codfiscal.py --mode full --limit 0 --query "Summarize the main topics"

Notes:
 - `config.yaml` must contain OpenAI and Neo4j credentials (or set via env vars)
 - The script will avoid clustering / community reports on Neo4j instances
   that do not provide Graph Data Science (GDS) procedures.
"""
import argparse
import asyncio
import logging
import yaml
from pathlib import Path

from hirag import HiRAG, QueryParam
from openai import AsyncOpenAI
from hirag._storage.gdb_neo4j import Neo4jStorage
from hirag._utils import wrap_embedding_func_with_attrs


logging.basicConfig(level=logging.INFO)


def load_cfg():
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def make_openai_embedding_func(api_key, base_url, model_name='text-embedding-3-small'):
    @wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
    async def openai_embedding(texts: list[str]):
        # try base_url then fallback
        last_exc = None
        if base_url:
            try:
                client = AsyncOpenAI(api_key=api_key, base_url=base_url)
                resp = await client.embeddings.create(model=model_name, input=texts, encoding_format='float')
                return __import__('numpy').array([d.embedding for d in resp.data])
            except Exception as e:
                last_exc = e
        try:
            client = AsyncOpenAI(api_key=api_key)
            resp = await client.embeddings.create(model=model_name, input=texts, encoding_format='float')
            return __import__('numpy').array([d.embedding for d in resp.data])
        except Exception:
            if last_exc:
                raise last_exc
            raise

    return openai_embedding


def make_openai_chat_func(api_key, base_url, model_name='gpt-4o-mini'):
    async def openai_chat(prompt, system_prompt=None, history_messages=None, **kwargs):
        kwargs.pop('hashing_kv', None)
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({'role': 'user', 'content': prompt})

        last_exc = None
        if base_url:
            try:
                client = AsyncOpenAI(api_key=api_key, base_url=base_url)
                resp = await client.chat.completions.create(model=model_name, messages=messages, **kwargs)
                return resp.choices[0].message.content
            except Exception as e:
                last_exc = e
        try:
            client = AsyncOpenAI(api_key=api_key)
            resp = await client.chat.completions.create(model='gpt-3.5-turbo', messages=messages, **kwargs)
            return resp.choices[0].message.content
        except Exception as e:
            raise last_exc or e

    return openai_chat


async def run(args):
    cfg = load_cfg()
    openai_cfg = cfg.get('openai', {})
    api_key = openai_cfg.get('api_key')
    base_url = openai_cfg.get('base_url') or None
    hirag_cfg = cfg.get('hirag', {})
    neo4j_url = hirag_cfg.get('neo4j_url')
    neo4j_auth = hirag_cfg.get('neo4j_auth')

    if not api_key:
        raise SystemExit('OpenAI API key missing in config.yaml')
    if not neo4j_url or not neo4j_auth:
        raise SystemExit('Neo4j connection info missing in config.yaml')

    embedding_func = make_openai_embedding_func(api_key, base_url)
    chat_func = make_openai_chat_func(api_key, base_url)

    print('Creating HiRAG instance (Neo4jStorage + OpenAI)')
    hr = HiRAG(
        working_dir='./tmp_codfiscal_workdir',
        enable_llm_cache=True,
        embedding_func=embedding_func,
        best_model_func=chat_func,
        cheap_model_func=chat_func,
        enable_hierachical_mode=True,
        embedding_batch_num=8,
        embedding_func_max_async=4,
        enable_naive_rag=False,
        graph_storage_cls=Neo4jStorage,
        addon_params={'neo4j_url': neo4j_url, 'neo4j_auth': neo4j_auth},
    )

    # defensive: skip clustering / community report on Aura instances lacking GDS
    async def _noop_clustering(algorithm: str):
        logging.info('Skipping clustering for demo (GDS not available or skipped)')
        return None
    hr.chunk_entity_relation_graph.clustering = _noop_clustering

    # also patch community report to avoid requiring clustering outputs
    import hirag._op as _op
    import hirag.hirag as _hirag

    async def _noop_generate_community_report(*a, **k):
        logging.info('Skipping community report for demo')
        return None

    _op.generate_community_report = _noop_generate_community_report
    _hirag.generate_community_report = _noop_generate_community_report

    # read codfiscal.txt
    p = Path('codfiscal.txt')
    if not p.exists():
        raise SystemExit('codfiscal.txt not found in workspace root')

    text = p.read_text(encoding='utf-8')

    # Quick mode: index only the first N characters to limit LLM calls
    if args.mode == 'quick' and args.limit > 0:
        text_to_insert = text[: args.limit]
        print(f'Quick mode: inserting first {args.limit} characters')
    elif args.mode == 'quick' and args.limit == 0:
        # default quick to first 8192 chars if not specified
        text_to_insert = text[:8192]
        print('Quick mode: inserting first 8192 characters')
    elif args.mode == 'full':
        if not args.confirm:
            raise SystemExit('Full mode requires --confirm to avoid accidental heavy LLM use')
        text_to_insert = text
        print('Full mode: indexing entire codfiscal.txt (may be slow/costly)')
    else:
        raise SystemExit('Unsupported mode')

    print('Indexing document — this will extract entities and upsert into Neo4j (may call LLMs/embeddings).')
    await hr.ainsert(text_to_insert)

    if args.query:
        print('\nRunning query:')
        ans = await hr.aquery(args.query, param=QueryParam(mode='hi'))
        print('\n--- Answer ---\n', ans)
    else:
        print('\nIndexing complete. No query supplied. Use --query to run a retrieval.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick')
    parser.add_argument('--limit', type=int, default=8192, help='characters (quick mode)')
    parser.add_argument('--confirm', action='store_true', help='allow full indexing (use carefully)')
    parser.add_argument('--query', type=str, help='query to run after indexing')
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == '__main__':
    main()
