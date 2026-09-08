import os
from unittest.mock import MagicMock
import pytest
from chromadb.utils.embedding_functions.voyageai_embedding_function import (
    VoyageAIEmbeddingFunction,
)

voyageai = pytest.importorskip("voyageai", reason="voyageai not installed")


def test_with_embedding_dimensions() -> None:
    if os.environ.get("CHROMA_VOYAGE_API_KEY") is None:
        pytest.skip("CHROMA_VOYAGE_API_KEY not set")
    ef = VoyageAIEmbeddingFunction(
        api_key=os.environ["CHROMA_VOYAGE_API_KEY"]
    )
    embeddings = ef(["hello world"])
    assert embeddings is not None
    assert len(embeddings) == 1
    assert len(embeddings[0]) > 0


def _make_ef(model_name: str, input_type: str = None) -> VoyageAIEmbeddingFunction:
    ef = VoyageAIEmbeddingFunction(
        api_key="fake-key", model_name=model_name, input_type=input_type
    )
    ef._client = MagicMock()
    return ef


def test_contextualized_document_path() -> None:
    ef = _make_ef("voyage-context-4")

    r0 = MagicMock(index=0, embeddings=[[0.1, 0.2]])
    r1 = MagicMock(index=1, embeddings=[[0.3, 0.4]])
    # Return out of order to prove results are reordered by index.
    ef._client.contextualized_embed.return_value = MagicMock(results=[r1, r0])

    out = ef(["doc a", "doc b"])

    ef._client.contextualized_embed.assert_called_once_with(
        inputs=["doc a", "doc b"],
        model="voyage-context-4",
        input_type="document",
        enable_auto_chunking=True,
        chunk_size=VoyageAIEmbeddingFunction.CONTEXT_CHUNK_SIZE,
    )
    assert VoyageAIEmbeddingFunction.CONTEXT_CHUNK_SIZE == 32000
    assert len(out) == 2
    assert out[0].tolist() == pytest.approx([0.1, 0.2])
    assert out[1].tolist() == pytest.approx([0.3, 0.4])
    ef._client.embed.assert_not_called()


def test_contextualized_query_path() -> None:
    ef = _make_ef("voyage-context-4", input_type="query")

    r0 = MagicMock(index=0, embeddings=[[0.5, 0.6]])
    ef._client.contextualized_embed.return_value = MagicMock(results=[r0])

    out = ef(["a query"])

    ef._client.contextualized_embed.assert_called_once_with(
        inputs=["a query"],
        model="voyage-context-4",
        input_type="query",
    )
    # Query path must not pass auto-chunking / chunk_size.
    _, kwargs = ef._client.contextualized_embed.call_args
    assert "enable_auto_chunking" not in kwargs
    assert "chunk_size" not in kwargs
    assert len(out) == 1
    assert out[0].tolist() == pytest.approx([0.5, 0.6])


def test_plain_model_uses_embed() -> None:
    ef = _make_ef("voyage-4")
    ef._client.embed.return_value = MagicMock(embeddings=[[1.0, 2.0]])

    out = ef(["hello"])

    ef._client.embed.assert_called_once()
    ef._client.contextualized_embed.assert_not_called()
    assert out[0].tolist() == pytest.approx([1.0, 2.0])
