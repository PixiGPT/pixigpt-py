"""Tests for embeddings and rerank API."""
import os
import pytest
from pixigpt import Client, EmbeddingRequest, RerankRequest


@pytest.fixture
def client():
    """Create client from environment variables."""
    base_url = os.getenv("PIXIGPT_BASE_URL")
    api_key = os.getenv("PIXIGPT_API_KEY")
    if not base_url or not api_key:
        pytest.skip("PIXIGPT_BASE_URL or PIXIGPT_API_KEY not set")
    return Client(base_url=base_url, api_key=api_key)


def test_single_embedding(client):
    """Test generating a single embedding."""
    response = client.create_embedding(
        EmbeddingRequest(input="The quick brown fox jumps over the lazy dog")
    )

    assert len(response.data) == 1
    assert len(response.data[0].embedding) > 0
    assert response.data[0].index == 0
    assert response.usage.prompt_tokens > 0
    assert response.usage.total_tokens > 0

    print(f"Embedding dimensions: {len(response.data[0].embedding)}")
    print(f"First 5 values: {response.data[0].embedding[:5]}")
    print(
        f"Usage: {response.usage.prompt_tokens} prompt tokens, {response.usage.total_tokens} total"
    )


def test_batch_embeddings(client):
    """Test generating multiple embeddings."""
    texts = [
        "Artificial intelligence is transforming technology",
        "Machine learning models process vast amounts of data",
        "Neural networks are inspired by biological neurons",
        "Deep learning requires significant computational resources",
    ]

    response = client.create_embedding(EmbeddingRequest(input=texts))

    assert len(response.data) == len(texts)
    for i, embedding_data in enumerate(response.data):
        assert embedding_data.index == i
        assert len(embedding_data.embedding) > 0

    print(f"Generated {len(response.data)} embeddings")
    print(f"Dimensions: {len(response.data[0].embedding)}")
    print(f"Usage: {response.usage.total_tokens} total tokens")


def test_rerank(client):
    """Test reranking documents."""
    query = "machine learning algorithms"
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on data-driven predictions",
        "Cats are popular pets known for their independence and playful nature",
        "Supervised learning algorithms learn from labeled training data to make predictions",
        "The weather forecast predicts rain tomorrow afternoon",
        "Neural networks use layers of interconnected nodes to process information",
        "Pizza is a traditional Italian dish with cheese and tomato sauce",
    ]

    response = client.rerank(
        RerankRequest(query=query, documents=documents, top_k=3)
    )

    assert len(response.results) > 0
    assert response.usage.total_tokens > 0

    # Verify results are sorted by score (descending)
    for i in range(1, len(response.results)):
        assert (
            response.results[i].relevance_score
            <= response.results[i - 1].relevance_score
        ), "Results should be sorted by relevance score (descending)"

    print(f"Top {len(response.results)} results:")
    for i, result in enumerate(response.results, 1):
        print(f"  {i}. [{result.relevance_score:.3f}] {result.document}")
    print(f"Usage: {response.usage.total_tokens} total tokens")
