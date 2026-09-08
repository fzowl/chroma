import os
from unittest.mock import MagicMock, patch

import pytest

from chromadb.utils.embedding_functions.voyageai_embedding_function import (
    VoyageAIEmbeddingFunction,
)

voyageai = pytest.importorskip("voyageai", reason="voyageai not installed")


def test_with_embedding_dimensions() -> None:
    if os.environ.get("CHROMA_VOYAGE_API_KEY") is None:
        pytest.skip("CHROMA_VOYAGE_API_KEY not set")
    # Pin the model explicitly so the assertion stays meaningful regardless of
    # the default. voyage-4-large returns 1024-dim embeddings by default.
    ef = VoyageAIEmbeddingFunction(
        api_key=os.environ["CHROMA_VOYAGE_API_KEY"],
        model_name="voyage-4-large",
    )
    embeddings = ef(["hello world"])
    assert embeddings is not None
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 1024


def test_contextualized_embed_wrapping() -> None:
    """Contextual models call contextualized_embed with each document wrapped
    as its own single-chunk group, and unwrap one embedding per document."""
    documents = ["first doc", "second doc"]

    # Fake contextualized_embed response: one result per group, each holding a
    # single-chunk embeddings list.
    result = MagicMock()
    result.results = [
        MagicMock(embeddings=[[0.1, 0.2, 0.3]]),
        MagicMock(embeddings=[[0.4, 0.5, 0.6]]),
    ]
    mock_client = MagicMock()
    mock_client.contextualized_embed.return_value = result

    with patch("voyageai.Client", return_value=mock_client):
        ef = VoyageAIEmbeddingFunction(
            api_key="fake-key",
            model_name="voyage-context-4",
            truncation=False,
        )
        embeddings = ef(documents)

    # Standard embed must not be used for contextual models.
    mock_client.embed.assert_not_called()
    call_kwargs = mock_client.contextualized_embed.call_args.kwargs
    assert call_kwargs["inputs"] == [["first doc"], ["second doc"]]
    assert call_kwargs["model"] == "voyage-context-4"
    assert call_kwargs["truncation"] is False

    # 1:1 document -> embedding mapping, one embedding unwrapped per group.
    assert len(embeddings) == 2
    assert [emb.tolist() for emb in embeddings] == [
        [pytest.approx(0.1), pytest.approx(0.2), pytest.approx(0.3)],
        [pytest.approx(0.4), pytest.approx(0.5), pytest.approx(0.6)],
    ]
