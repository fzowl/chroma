import os
import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from chromadb.utils.embedding_functions.voyageai_embedding_function import (
    VoyageAIEmbeddingFunction,
    CONTEXT_CHUNK_SIZE,
)


def _make_ef(
    model_name: str = "voyage-3.5", input_type=None
) -> VoyageAIEmbeddingFunction:
    """Build an EF with the voyageai package and client stubbed out."""
    with patch.dict(sys.modules, {"voyageai": MagicMock()}):
        with patch.dict(os.environ, {"CHROMA_VOYAGE_API_KEY": "test-key"}, clear=False):
            ef = VoyageAIEmbeddingFunction(
                model_name=model_name, input_type=input_type
            )
    ef._client = MagicMock()
    return ef


# --- token-aware batching logic (pure, no client needed) --------------------


def test_batch_indices_splits_at_token_boundary() -> None:
    counts = [10, 10, 10, 10]
    batches = VoyageAIEmbeddingFunction._batch_indices(
        counts, max_tokens=20, max_items=1000
    )
    assert batches == [(0, 2), (2, 4)]


def test_batch_indices_single_oversized_text_goes_alone() -> None:
    counts = [5, 100, 5]
    batches = VoyageAIEmbeddingFunction._batch_indices(
        counts, max_tokens=10, max_items=1000
    )
    # The 100-token input exceeds the limit but must still be emitted alone.
    assert batches == [(0, 1), (1, 2), (2, 3)]


def test_batch_indices_respects_item_cap() -> None:
    counts = [1] * 2500
    batches = VoyageAIEmbeddingFunction._batch_indices(
        counts, max_tokens=10_000_000, max_items=1000
    )
    assert batches == [(0, 1000), (1000, 2000), (2000, 2500)]


def test_embed_batches_by_token_limit() -> None:
    ef = _make_ef(model_name="voyage-3.5")  # 320K token limit
    # Three inputs of 200K tokens each -> at most one per batch.
    ef._client.tokenize.return_value = [[0] * 200_000] * 3

    def fake_embed(texts, **kwargs):  # type: ignore[no-untyped-def]
        result = MagicMock()
        result.embeddings = [[1.0, 2.0] for _ in texts]
        return result

    ef._client.embed.side_effect = fake_embed

    out = ef(["x", "y", "z"])

    assert ef._client.embed.call_count == 3
    assert len(out) == 3
    ef._client.contextualized_embed.assert_not_called()


# --- contextualized embedding paths -----------------------------------------


def test_contextualized_document_path_uses_auto_chunking() -> None:
    ef = _make_ef(model_name="voyage-context-3", input_type=None)

    result_obj = MagicMock()
    r0 = MagicMock(index=0, embeddings=[[0.1, 0.2]])
    r1 = MagicMock(index=1, embeddings=[[0.3, 0.4]])
    result_obj.results = [r1, r0]  # returned out of order on purpose
    ef._client.contextualized_embed.return_value = result_obj

    out = ef(["a", "b"])

    kwargs = ef._client.contextualized_embed.call_args.kwargs
    assert kwargs["inputs"] == ["a", "b"]
    assert kwargs["enable_auto_chunking"] is True
    assert kwargs["chunk_size"] == CONTEXT_CHUNK_SIZE
    assert kwargs["input_type"] == "document"
    ef._client.embed.assert_not_called()

    # Results are reordered by index; one vector per input.
    assert len(out) == 2
    np.testing.assert_allclose(out[0], [0.1, 0.2])
    np.testing.assert_allclose(out[1], [0.3, 0.4])


def test_contextualized_query_path_disables_auto_chunking() -> None:
    ef = _make_ef(model_name="voyage-context-3", input_type="query")

    result_obj = MagicMock()
    result_obj.results = [MagicMock(index=0, embeddings=[[0.5, 0.6]])]
    ef._client.contextualized_embed.return_value = result_obj

    out = ef(["q"])

    kwargs = ef._client.contextualized_embed.call_args.kwargs
    assert kwargs["enable_auto_chunking"] is False
    assert "chunk_size" not in kwargs  # dropped when auto-chunking is off
    assert kwargs["input_type"] == "query"
    assert len(out) == 1
    np.testing.assert_allclose(out[0], [0.5, 0.6])


# --- live integration (requires the real package and an API key) ------------


def test_with_embedding_dimensions() -> None:
    pytest.importorskip("voyageai", reason="voyageai not installed")
    if os.environ.get("CHROMA_VOYAGE_API_KEY") is None:
        pytest.skip("CHROMA_VOYAGE_API_KEY not set")
    ef = VoyageAIEmbeddingFunction(api_key=os.environ["CHROMA_VOYAGE_API_KEY"])
    embeddings = ef(["hello world"])
    assert embeddings is not None
    assert len(embeddings) == 1
    # Default model voyage-3.5 produces 1024-dim vectors.
    assert len(embeddings[0]) == 1024
