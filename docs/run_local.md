## How to run basic checks locally (dev container)

These quick checks help you confirm the project is installed and external services (Neo4j, OpenAI) are reachable.

1) Install project dependencies (in dev container):

```bash
python -m pip install -e .
```

2) Put secrets into `config.yaml` (DO NOT commit them to a public repo). Example keys used in tests are stored in `config.yaml` in this workspace for convenience.

3) Run the Neo4j connectivity test (will attempt to connect to the configured Neo4j URL):

```bash
python scripts/test_neo4j_connect.py
```

4) Run a single OpenAI embedding request to validate your OpenAI API key:

```bash
python scripts/test_openai_connect.py
```

5) Run a small offline smoke test that avoids LLM calls (uses dummy extractors):

```bash
python scripts/local_smoke_test.py
```

## Using your codfiscal.txt (end-to-end demo)

We added `scripts/full_demo_codfiscal.py` — a safe demo that indexes your `codfiscal.txt` and runs one hierarchical query.

- Default (quick) — indexes only the first ~8k characters (cheap / fast) and runs a query:

```bash
python scripts/full_demo_codfiscal.py --limit 4096 --query "Care sunt principalele teme din acest document?"
```

- Full mode (indexes the whole `codfiscal.txt`) — may perform many LLM calls and be slow/costly. You must explicitly pass `--mode full --confirm` to run it.

```bash
python scripts/full_demo_codfiscal.py --mode full --confirm --query "Summarize the main topics"
```

⚠️ Reminder: full mode can be expensive. If you want me to run full indexing for you now, say "Yes, run full index" and I will proceed (but confirm you accept any resulting API costs).

## Security: recommend switching to environment variables

Storing secrets in `config.yaml` is convenient for local tests, but not recommended for long-term or public repositories. Recommended patterns:

- Use environment variables (OPENAI_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD) and update scripts to read them.
- Add `.env.example` and update `.gitignore` to avoid committing secrets.

If you want, I can update the code to load credentials from environment variables and remove them from `config.yaml` automatically.

If all three scripts succeed (Neo4j connectivity, OpenAI embedding, offline smoke-test), the package and basic integrations are working in this environment.

If you want, I can now:
- Run a full example that inserts real data and performs retrieval (requires valid API/dataset access)
- Help you remove secrets from `config.yaml` and switch to environment variables
- Set up a small demo script which runs a full query pipeline and saves results
