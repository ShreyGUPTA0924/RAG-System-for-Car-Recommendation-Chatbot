"""
Qdrant-backed LangChain retriever with metadata filtering support.
"""

import logging
import os
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv
# Use langchain_community (more compatible with current qdrant-client)
try:
    from langchain_community.vectorstores import Qdrant as QdrantVectorStore
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_core.retrievers import BaseRetriever
except ImportError:
    # Fallback for older LangChain versions
    try:
        from langchain.vectorstores import Qdrant as QdrantVectorStore
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.schema import BaseRetriever
    except ImportError:
        from langchain_community.vectorstores import Qdrant as QdrantVectorStore
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain.schema import BaseRetriever
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from qdrant_client.http.models import NearestQuery
from typing import List, Any, Optional
import logging

logger = logging.getLogger(__name__)


def _add_search_method_to_client(client: QdrantClient) -> QdrantClient:
    """
    Add 'search' method to QdrantClient instance for LangChain compatibility.
    
    This is a robust, future-proof solution that:
    1. Works with current qdrant-client 1.16+
    2. Maintains compatibility with LangChain's Qdrant integration
    3. Handles all edge cases properly
    4. Won't break with future updates (only adds method if missing)
    
    Args:
        client: QdrantClient instance to patch
        
    Returns:
        The same client instance with search method added
    """
    # Only add if it doesn't exist (future-proof)
    if hasattr(client, 'search'):
        return client
    
    def search_method(
        self,
        collection_name: str = None, 
        query_vector: List[float] = None, 
        limit: int = 10, 
        **kwargs
    ) -> Any:
        """
        Compatibility method that wraps query_points to match LangChain's expected API.
        
        This method is added to QdrantClient instances to bridge the gap between
        qdrant-client 1.16+ (which uses query_points) and LangChain's expectation
        of a 'search' method.
        """
        try:
            # Handle both positional and keyword arguments
            # LangChain may call: search(collection_name=..., query_vector=..., limit=...)
            if collection_name is None and 'collection_name' in kwargs:
                collection_name = kwargs.pop('collection_name')
            if query_vector is None and 'query_vector' in kwargs:
                query_vector = kwargs.pop('query_vector')
            
            # Ensure query_vector is a list
            if query_vector is None:
                raise ValueError("query_vector is required")
            if not isinstance(query_vector, list):
                query_vector = list(query_vector)
            
            # Create NearestQuery for vector similarity search
            # NearestQuery expects: nearest=[...] (list of floats directly)
            nearest_query = NearestQuery(nearest=query_vector)
            
            # Build query parameters
            query_params = {
                "query": nearest_query,
                "limit": limit
            }
            
            # Add filter if provided (for metadata filtering)
            if "filter" in kwargs and kwargs["filter"] is not None:
                query_params["query_filter"] = kwargs["filter"]
            
            # Execute query using the modern query_points API
            results = client.query_points(
                collection_name=collection_name,
                **query_params
            )
            
            # LangChain expects an iterable of points directly
            # Return the points list directly (it's already iterable)
            return results.points
            
        except Exception as e:
            logger.error(f"Error in search compatibility method: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            # Return empty list on error to prevent crashes (LangChain expects iterable)
            return []
    
    # Bind the method to the client instance
    import types
    client.search = types.MethodType(search_method, client)
    
    return client

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Environment variables
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "cars_rag")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))


def get_qdrant_client() -> QdrantClient:
    """
    Get Qdrant client instance with LangChain compatibility.
    
    This function returns a QdrantClient instance with the 'search' method
    added for compatibility with LangChain's Qdrant integration.
    
    Returns:
        QdrantClient instance with search method added
    """
    if QDRANT_API_KEY:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    else:
        client = QdrantClient(url=QDRANT_URL)
    
    # Add search method for LangChain compatibility
    return _add_search_method_to_client(client)


def build_qdrant_filter(filters: Dict[str, Any]) -> Optional[Filter]:
    """
    Build Qdrant filter from metadata filters.
    
    Args:
        filters: Dictionary of filter conditions
            - price_max: Maximum price
            - price_min: Minimum price
            - body_type: Exact match for body type
            - fuel_type: Exact match for fuel type
            - year_min: Minimum year
            - year_max: Maximum year
            
    Returns:
        Qdrant Filter object or None
    """
    if not filters:
        return None
    
    conditions = []
    
    # Price filters
    if "price_max" in filters:
        conditions.append(
            FieldCondition(
                key="price",
                range=Range(lte=filters["price_max"])
            )
        )
    if "price_min" in filters:
        conditions.append(
            FieldCondition(
                key="price",
                range=Range(gte=filters["price_min"])
            )
        )
    
    # Body type filter
    if "body_type" in filters:
        conditions.append(
            FieldCondition(
                key="body_type",
                match=MatchValue(value=filters["body_type"])
            )
        )
    
    # Fuel type filter
    if "fuel_type" in filters:
        conditions.append(
            FieldCondition(
                key="fuel_type",
                match=MatchValue(value=filters["fuel_type"])
            )
        )
    
    # Year filters
    if "year_min" in filters:
        conditions.append(
            FieldCondition(
                key="year",
                range=Range(gte=filters["year_min"])
            )
        )
    if "year_max" in filters:
        conditions.append(
            FieldCondition(
                key="year",
                range=Range(lte=filters["year_max"])
            )
        )
    
    if not conditions:
        return None
    
    # If multiple conditions, combine with AND
    if len(conditions) == 1:
        return Filter(must=conditions)
    else:
        return Filter(must=conditions)


def get_retriever(filters: Optional[Dict[str, Any]] = None, k: int = None) -> BaseRetriever:
    """
    Get Qdrant retriever with optional metadata filters.
    
    Args:
        filters: Dictionary of metadata filters
        k: Number of documents to retrieve (defaults to RETRIEVAL_K env var)
        
    Returns:
        Qdrant retriever instance
    """
    if k is None:
        k = RETRIEVAL_K
    
    logger.info(f"Creating retriever with k={k}, filters={filters}")
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Get Qdrant client (now has search method via monkey-patch)
    client = get_qdrant_client()
    
    # Build filter
    qdrant_filter = build_qdrant_filter(filters) if filters else None
    
    # Create Qdrant vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
        embeddings=embeddings
    )
    
    # Create retriever with filter
    if qdrant_filter:
        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": k,
                "filter": qdrant_filter
            }
        )
    else:
        retriever = vector_store.as_retriever(
            search_kwargs={"k": k}
        )
    
    logger.info("Retriever created successfully")
    return retriever

