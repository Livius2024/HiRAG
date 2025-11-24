"""Small offline smoke test for HiRAG package.

This script avoids any network/LLM calls by providing dummy extractor functions
that return no entities (so the insert() returns early and doesn't call LLMs).
It only tests import, instantiation and calling insert without external deps.
"""
from hirag import HiRAG


async def _dummy_extract(*args, **kwargs):
    """Accept any args/kwargs and return None to avoid LLM calls."""
    return None


def main():
    # Construct a HiRAG instance in a temp working directory and safe settings.
    hr = HiRAG(
        working_dir="./tmp_demo_workdir",
        enable_local=False,
        enable_naive_rag=False,
        enable_hierachical_mode=False,
        # override extraction to avoid LLMs during insert
        entity_extraction_func=_dummy_extract,
        hierarchical_entity_extraction_func=_dummy_extract,
        always_create_working_dir=True,
    )

    print("HiRAG instance created successfully:", type(hr))

    # run insert -> our dummy extractor returns None and ainsert should return early
    hr.insert("A short test document that will be ignored by the dummy extractor.")

    print("Insert ran (dummy extractor) — no external API calls performed.")


if __name__ == "__main__":
    main()
