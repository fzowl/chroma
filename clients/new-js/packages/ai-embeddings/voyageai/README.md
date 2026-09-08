# Voyage AI by MongoDB Embedding Function for Chroma

This package provides a Voyage AI by MongoDB embedding provider for Chroma.

## Installation

```bash
npm install @chroma-core/voyageai
```

## Usage

```typescript
import { ChromaClient } from 'chromadb';
import { VoyageAIEmbeddingFunction } from '@chroma-core/voyageai';

// Initialize the embedder
const embedder = new VoyageAIEmbeddingFunction({
  apiKey: 'your-api-key', // Or set VOYAGE_API_KEY env var
  modelName: 'voyage-4',
});

// Create a new ChromaClient
const client = new ChromaClient({
  path: 'http://localhost:8000',
});

// Create a collection with the embedder
const collection = await client.createCollection({
  name: 'my-collection',
  embeddingFunction: embedder,
});

// Add documents
await collection.add({
  ids: ["1", "2", "3"],
  documents: ["Document 1", "Document 2", "Document 3"],
});

// Query documents
const results = await collection.query({
  queryTexts: ["Sample query"],
  nResults: 2,
});
```

## Configuration

Set your Voyage AI API key as an environment variable:

```bash
export VOYAGE_API_KEY=your-api-key
```

Get your API key from [Voyage AI by MongoDB](https://www.voyageai.com/).

## Configuration Options

- **apiKey**: Your Voyage AI API key (or set via environment variable)
- **apiKeyEnvVar**: Environment variable name for API key (default: `VOYAGE_API_KEY`)
- **modelName**: Model to use for embeddings (required)

## Supported Models

Voyage AI by MongoDB offers high-quality embedding models (all with a
32,000-token context window and a default of 1024 dimensions):

- `voyage-4-large` - Highest general-purpose and multilingual retrieval quality
- `voyage-4` - General-purpose and multilingual retrieval (recommended default)
- `voyage-4-lite` - Latency- and cost-optimized general-purpose retrieval
- `voyage-code-4` - Optimized for code retrieval and coding-agent use cases
- `voyage-finance-2` - Optimized for finance retrieval and RAG
- `voyage-law-2` - Optimized for legal retrieval and RAG
- `voyage-context-4` - Contextualized chunk embeddings

Check the [Voyage AI by MongoDB documentation](https://docs.voyageai.com/docs/embeddings) for the complete list of available models and their specifications.

## Features

- **State-of-the-Art Quality**: High-performance embedding models optimized for retrieval
- **Domain-Specific Models**: Specialized models for code, legal, and other domains
- **Efficient**: Competitive pricing and fast inference times