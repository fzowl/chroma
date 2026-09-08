from chromadb.api.types import EmbeddingFunction, Space, Embeddings, Documents
from chromadb.utils.embedding_functions.schemas import validate_config_schema
from typing import List, Dict, Any, Optional, Tuple
import os
import numpy as np
import warnings


# Maximum number of inputs VoyageAI accepts in a single request. Batches are also
# bounded by a per-model token limit (see VOYAGE_TOKEN_LIMITS below).
MAX_BATCH_SIZE = 1000

# Total token budget per request, keyed by model. Values mirror VoyageAI's
# documented per-model limits (see docs.voyageai.com model overview). Models not
# listed fall back to DEFAULT_TOKEN_LIMIT.
DEFAULT_TOKEN_LIMIT = 120_000
VOYAGE_TOKEN_LIMITS: Dict[str, int] = {
    # 1M-token batch models
    "voyage-4-lite": 1_000_000,
    "voyage-3.5-lite": 1_000_000,
    # 320K-token batch models
    "voyage-4": 320_000,
    "voyage-3.5": 320_000,
    "voyage-2": 320_000,
    # 120K-token batch models
    "voyage-4-large": 120_000,
    "voyage-4-nano": 120_000,
    "voyage-code-4": 120_000,
    "voyage-finance-2": 120_000,
    "voyage-law-2": 120_000,
    "voyage-context-4": 120_000,
    "voyage-context-3": 120_000,
    "voyage-3-large": 120_000,
    "voyage-code-3": 120_000,
    "voyage-3": 120_000,
    "voyage-3-lite": 120_000,
    "voyage-multilingual-2": 120_000,
    "voyage-large-2-instruct": 120_000,
    "voyage-large-2": 120_000,
    "voyage-code-2": 120_000,
}

# Chunk size (in tokens) used on the contextualized document path. Set to the
# model's maximum single-chunk length so that every input string resolves to
# exactly one chunk -> one embedding. See _embed_contextualized for the rationale.
CONTEXT_CHUNK_SIZE = 32_000


class VoyageAIEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    This class is used to generate embeddings for a list of texts using the VoyageAI API.

    Two model families are supported:

    * Plain text embedding models (e.g. ``voyage-3.5``, ``voyage-4``,
      ``voyage-code-4``). Inputs are embedded with ``client.embed`` using
      token-aware batching against the model's per-request token limit.
    * Contextualized embedding models (``voyage-context-*``). Each input string
      is embedded as its own independent document via ``client.contextualized_embed``.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "voyage-3.5",
        api_key_env_var: str = "CHROMA_VOYAGE_API_KEY",
        input_type: Optional[str] = None,
        truncation: bool = True,
    ):
        """
        Initialize the VoyageAIEmbeddingFunction.

        Args:
            api_key_env_var (str, optional): Environment variable name that contains your API key for the VoyageAI API.
                Defaults to "CHROMA_VOYAGE_API_KEY".
            model_name (str, optional): The name of the model to use for text embeddings.
                Defaults to "voyage-3.5". Contextualized models (``voyage-context-*``)
                are also supported.
            api_key (str, optional): API key for the VoyageAI API. If not provided, will look for it in the environment variable.
            input_type (str, optional): The type of input to use for the VoyageAI API
                (``"query"`` or ``"document"``). Defaults to None.
            truncation (bool): Whether to truncate the input text.
                Defaults to True.
        """
        try:
            import voyageai
        except ImportError:
            raise ValueError(
                "The voyageai python package is not installed. Please install it with `pip install voyageai`"
            )

        if api_key is not None:
            warnings.warn(
                "Direct api_key configuration will not be persisted. "
                "Please use environment variables via api_key_env_var for persistent storage.",
                DeprecationWarning,
            )

        if os.getenv("VOYAGE_API_KEY") is not None:
            self.api_key_env_var = "VOYAGE_API_KEY"
        else:
            self.api_key_env_var = api_key_env_var

        self.api_key = api_key or os.getenv(self.api_key_env_var)
        if not self.api_key:
            raise ValueError(
                f"The {self.api_key_env_var} environment variable is not set."
            )

        self.model_name = model_name
        self.input_type = input_type
        self.truncation = truncation
        self._client = voyageai.Client(api_key=self.api_key)

    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for the given documents.

        Args:
            input: Documents to generate embeddings for.

        Returns:
            Embeddings for the documents (one vector per input, in order).
        """
        texts = list(input)
        if not texts:
            return []

        if self._is_contextualized_model():
            return self._embed_contextualized(texts)
        return self._embed_texts(texts)

    def _is_contextualized_model(self) -> bool:
        return self.model_name.startswith("voyage-context")

    def _embed_texts(self, texts: List[str]) -> Embeddings:
        """Embed plain text with token-aware batching."""
        # Count tokens per input once, using the model's own tokenizer, and build
        # batches bounded by both the token budget and the item cap.
        token_counts = [
            len(encoding)
            for encoding in self._client.tokenize(texts, model=self.model_name)
        ]
        max_tokens = VOYAGE_TOKEN_LIMITS.get(self.model_name, DEFAULT_TOKEN_LIMIT)

        embeddings: List[Any] = []
        for start, end in self._batch_indices(token_counts, max_tokens, MAX_BATCH_SIZE):
            result = self._client.embed(
                texts=texts[start:end],
                model=self.model_name,
                input_type=self.input_type,
                truncation=self.truncation,
            )
            embeddings.extend(result.embeddings)

        return [np.array(embedding, dtype=np.float32) for embedding in embeddings]

    def _embed_contextualized(self, texts: List[str]) -> Embeddings:
        """Embed each input as its own independent contextualized document.

        The batch is passed as a flat ``list[str]``. On the document path we
        enable auto-chunking with ``chunk_size`` set to the model's maximum, so
        every input string resolves to exactly one chunk and therefore exactly
        one embedding -- deterministic, one vector per input, in input order.

        Inputs are embedded *independently*: we deliberately do NOT send the
        batch as a single document's chunks (``inputs=[texts]``). Generic
        ``embed_many`` callers pass unrelated texts, so cross-input
        contextualization would leak context between them. Each string is only
        contextualized within itself.

        VoyageAI rejects auto-chunking when ``input_type`` is ``"query"``, so the
        query path sends neither ``enable_auto_chunking`` nor ``chunk_size``.
        """
        enable_auto_chunking = self.input_type != "query"

        request: Dict[str, Any] = {
            "inputs": texts,
            "model": self.model_name,
            "enable_auto_chunking": enable_auto_chunking,
        }
        if enable_auto_chunking:
            # Auto-chunking requires input_type="document".
            request["input_type"] = "document"
            request["chunk_size"] = CONTEXT_CHUNK_SIZE
        else:
            request["input_type"] = self.input_type

        result = self._client.contextualized_embed(**request)

        # Each result corresponds to one input document; take its single chunk.
        ordered = sorted(result.results, key=lambda r: r.index)
        return [
            np.array(r.embeddings[0], dtype=np.float32)
            for r in ordered
        ]

    @staticmethod
    def _batch_indices(
        token_counts: List[int], max_tokens: int, max_items: int
    ) -> List[Tuple[int, int]]:
        """Group inputs into ``(start, end)`` slices bounded by token and item caps.

        A batch is closed when adding the next input would exceed ``max_tokens``
        (unless the batch is empty, so a single oversized input still goes through
        alone) or when it reaches ``max_items`` inputs.
        """
        batches: List[Tuple[int, int]] = []
        start = 0
        n = len(token_counts)
        while start < n:
            end = start
            batch_tokens = 0
            while end < n and (end - start) < max_items:
                next_tokens = token_counts[end]
                if end > start and batch_tokens + next_tokens > max_tokens:
                    break
                batch_tokens += next_tokens
                end += 1
            batches.append((start, end))
            start = end
        return batches

    @staticmethod
    def name() -> str:
        return "voyageai"

    def default_space(self) -> Space:
        return "cosine"

    def supported_spaces(self) -> List[Space]:
        return ["cosine", "l2", "ip"]

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "EmbeddingFunction[Documents]":
        api_key_env_var = config.get("api_key_env_var")
        model_name = config.get("model_name")
        input_type = config.get("input_type")
        truncation = config.get("truncation")

        if api_key_env_var is None or model_name is None:
            assert False, "This code should not be reached"

        return VoyageAIEmbeddingFunction(
            api_key_env_var=api_key_env_var,
            model_name=model_name,
            input_type=input_type,
            truncation=truncation if truncation is not None else True,
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "api_key_env_var": self.api_key_env_var,
            "model_name": self.model_name,
            "input_type": self.input_type,
            "truncation": self.truncation,
        }

    def validate_config_update(
        self, old_config: Dict[str, Any], new_config: Dict[str, Any]
    ) -> None:
        if "model_name" in new_config:
            raise ValueError(
                "The model name cannot be changed after the embedding function has been initialized."
            )

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """
        Validate the configuration using the JSON schema.

        Args:
            config: Configuration to validate

        Raises:
            ValidationError: If the configuration does not match the schema
        """
        validate_config_schema(config, "voyageai")
