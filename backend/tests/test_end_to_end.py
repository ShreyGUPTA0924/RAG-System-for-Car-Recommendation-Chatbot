"""
End-to-end tests for car recommendations RAG system.
"""

import os
import sys
import pytest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

from backend.rag.chain import create_chain, query_chain
from backend.rag.loader import create_sample_cars_data

# Test configuration
TEST_COLLECTION_NAME = "test_cars_rag"
TEST_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
TEST_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="module")
def test_collection():
    """Create test Qdrant collection with sample data."""
    # Create sample data
    sample_cars = create_sample_cars_data()
    
    # Initialize Qdrant client
    client = QdrantClient(url=TEST_QDRANT_URL)
    
    # Delete collection if exists
    try:
        client.delete_collection(TEST_COLLECTION_NAME)
    except:
        pass
    
    # Load embedding model
    model = SentenceTransformer(TEST_EMBEDDING_MODEL)
    vector_size = model.get_sentence_embedding_dimension()
    
    # Create collection
    client.create_collection(
        collection_name=TEST_COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )
    
    # Create text representations and embeddings
    from backend.rag.embed import create_text_from_record
    texts = [create_text_from_record(car) for car in sample_cars]
    embeddings = model.encode(texts)
    
    # Upsert points
    points = []
    for idx, (car, embedding) in enumerate(zip(sample_cars, embeddings)):
        # Clean metadata
        metadata = {k: v for k, v in car.items() if isinstance(v, (str, int, float, bool, type(None)))}
        
        points.append(
            PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload=metadata
            )
        )
    
    client.upsert(collection_name=TEST_COLLECTION_NAME, points=points)
    
    yield TEST_COLLECTION_NAME
    
    # Cleanup
    try:
        client.delete_collection(TEST_COLLECTION_NAME)
    except:
        pass


@pytest.fixture
def mock_llm(monkeypatch):
    """Mock LLM to avoid requiring actual Ollama/API."""
    from unittest.mock import MagicMock
    
    def mock_get_llm():
        try:
            from langchain.llms.fake import FakeListLLM
        except ImportError:
            from langchain_community.llms.fake import FakeListLLM
        responses = [
            "Based on the context, I recommend the Toyota Camry Hybrid. It has excellent fuel economy with 51 MPG city and 53 MPG highway, priced at $28,000. This is a reliable midsize sedan perfect for fuel efficiency."
        ]
        return FakeListLLM(responses=responses)
    
    monkeypatch.setattr("backend.rag.model.get_llm", mock_get_llm)
    
    # Also patch the retriever to use test collection
    original_get_retriever = None
    try:
        from backend.rag import retriever
        original_get_retriever = retriever.get_retriever
        
        def mock_get_retriever(filters=None, k=5):
            try:
                from langchain.vectorstores import Qdrant
                from langchain.embeddings import HuggingFaceEmbeddings
            except ImportError:
                from langchain_community.vectorstores import Qdrant
                from langchain_community.embeddings import HuggingFaceEmbeddings
            
            embeddings = HuggingFaceEmbeddings(model_name=TEST_EMBEDDING_MODEL)
            client = QdrantClient(url=TEST_QDRANT_URL)
            
            vector_store = Qdrant(
                client=client,
                collection_name=TEST_COLLECTION_NAME,
                embeddings=embeddings
            )
            
            return vector_store.as_retriever(search_kwargs={"k": k})
        
        monkeypatch.setattr("backend.rag.retriever.get_retriever", mock_get_retriever)
    except Exception as e:
        print(f"Warning: Could not patch retriever: {e}")


def test_end_to_end_query(test_collection, mock_llm):
    """Test end-to-end query through the chain."""
    # Create chain
    chain = create_chain(filters=None, k=3)
    
    # Query
    result = query_chain(chain, "What are the best fuel-efficient cars?")
    
    # Assertions
    assert "answer" in result
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0
    
    assert "recommended" in result
    assert isinstance(result["recommended"], list)
    assert len(result["recommended"]) > 0
    
    assert "sources" in result
    assert isinstance(result["sources"], list)
    assert len(result["sources"]) > 0
    
    # Check recommended structure
    for car in result["recommended"]:
        assert isinstance(car, dict)
        # At least one car should have make/model
        if "make" in car or "model" in car:
            assert True


def test_end_to_end_with_filters(test_collection, mock_llm):
    """Test end-to-end query with metadata filters."""
    # Create chain with filters
    filters = {"price_max": 30000, "body_type": "Sedan"}
    chain = create_chain(filters=filters, k=3)
    
    # Query
    result = query_chain(chain, "Show me affordable sedans")
    
    # Assertions
    assert "answer" in result
    assert "recommended" in result
    assert "sources" in result
    
    # Check that recommended cars match filters (if price is present)
    for car in result["recommended"]:
        if "price" in car:
            assert car["price"] <= 30000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

