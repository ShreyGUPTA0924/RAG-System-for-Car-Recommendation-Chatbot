"""
Embedding generation and Qdrant vector database operations.
Computes embeddings using SentenceTransformers and upserts to Qdrant.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "backend" / "data" / "processed"

# Environment variables
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "cars_rag")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def create_text_from_record(record: Dict[str, Any]) -> str:
    """
    Create a searchable text representation from a car record.
    
    Args:
        record: Car record dictionary
        
    Returns:
        Concatenated text string for embedding
    """
    # Prioritize important fields
    text_parts = []
    
    if "make" in record:
        text_parts.append(f"Make: {record['make']}")
    if "model" in record:
        text_parts.append(f"Model: {record['model']}")
    if "description" in record:
        text_parts.append(record["description"])
    if "body_type" in record:
        text_parts.append(f"Body type: {record['body_type']}")
    if "fuel_type" in record:
        text_parts.append(f"Fuel type: {record['fuel_type']}")
    if "year" in record:
        text_parts.append(f"Year: {record['year']}")
    if "price" in record:
        text_parts.append(f"Price: ${record['price']}")
    if "mpg_city" in record or "mpg_highway" in record:
        mpg = f"MPG: City {record.get('mpg_city', 'N/A')}, Highway {record.get('mpg_highway', 'N/A')}"
        text_parts.append(mpg)
    
    # Add other fields
    for key, value in record.items():
        if key not in ["make", "model", "description", "body_type", "fuel_type", "year", "price", "mpg_city", "mpg_highway"]:
            if isinstance(value, (str, int, float)):
                text_parts.append(f"{key}: {value}")
    
    return " | ".join(text_parts)


def get_qdrant_client() -> QdrantClient:
    """
    Initialize and return Qdrant client.
    
    Returns:
        QdrantClient instance
    """
    if QDRANT_API_KEY:
        logger.info(f"Connecting to Qdrant cloud at {QDRANT_URL}")
        return QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )
    else:
        logger.info(f"Connecting to local Qdrant at {QDRANT_URL}")
        return QdrantClient(url=QDRANT_URL)


def create_collection(client: QdrantClient, collection_name: str, vector_size: int, recreate: bool = False):
    """
    Create or recreate Qdrant collection.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection
        vector_size: Size of embedding vectors
        recreate: If True, delete existing collection first
    """
    try:
        collections = client.get_collections().collections
        collection_exists = any(c.name == collection_name for c in collections)
        
        if collection_exists:
            if recreate:
                logger.info(f"Deleting existing collection: {collection_name}")
                client.delete_collection(collection_name)
            else:
                logger.info(f"Collection {collection_name} already exists. Use --recreate to rebuild.")
                return
        
        logger.info(f"Creating collection: {collection_name} with vector size {vector_size}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        logger.info(f"Collection {collection_name} created successfully")
        
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        raise


def load_processed_data(filename: str = "cars_processed.json") -> List[Dict[str, Any]]:
    """Load processed data from JSON file."""
    file_path = DATA_DIR / filename
    if not file_path.exists():
        logger.warning(f"Processed data file not found: {file_path}")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def embed_and_upsert(
    records: List[Dict[str, Any]],
    model: SentenceTransformer,
    client: QdrantClient,
    collection_name: str,
    batch_size: int = 100
):
    """
    Generate embeddings and upsert to Qdrant.
    
    Args:
        records: List of car records
        model: SentenceTransformer model
        client: QdrantClient instance
        collection_name: Collection name
        batch_size: Batch size for upserting
    """
    logger.info(f"Generating embeddings for {len(records)} records")
    
    # Generate texts for embedding
    texts = [create_text_from_record(record) for record in records]
    
    # Generate embeddings in batches
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        logger.info(f"Embedding batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")
        embeddings = model.encode(batch_texts, show_progress_bar=True)
        all_embeddings.extend(embeddings.tolist())
    
    logger.info("Embeddings generated. Upserting to Qdrant...")
    
    # Prepare points for upsert
    points = []
    for idx, (record, embedding, text) in enumerate(zip(records, all_embeddings, texts)):
        # Extract metadata (exclude embedding)
        metadata = {k: v for k, v in record.items() if k != "embedding"}
        
        # Ensure metadata values are JSON-serializable
        clean_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                clean_metadata[k] = v
            else:
                clean_metadata[k] = str(v)
        
        # Add page_content for LangChain compatibility
        # LangChain Qdrant expects the text content in payload
        clean_metadata["page_content"] = text
        
        points.append(
            PointStruct(
                id=idx,
                vector=embedding,
                payload=clean_metadata
            )
        )
    
    # Upsert in batches
    for i in range(0, len(points), batch_size):
        batch_points = points[i:i + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch_points
        )
        logger.info(f"Upserted batch {i // batch_size + 1}/{(len(points) + batch_size - 1) // batch_size}")
    
    logger.info(f"Successfully upserted {len(points)} points to collection {collection_name}")


def main(recreate: bool = False):
    """Main function to embed and upsert data."""
    # Load model
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    vector_size = model.get_sentence_embedding_dimension()
    logger.info(f"Model loaded. Vector size: {vector_size}")
    
    # Initialize Qdrant client
    client = get_qdrant_client()
    
    # Create collection
    create_collection(client, QDRANT_COLLECTION_NAME, vector_size, recreate=recreate)
    
    # Load data
    cars_data = load_processed_data("cars_processed.json")
    faq_data = load_processed_data("faq_processed.json")
    
    if not cars_data and not faq_data:
        logger.error("No processed data found. Run loader.py first.")
        return
    
    # Combine and embed
    all_records = cars_data + faq_data
    
    if all_records:
        embed_and_upsert(all_records, model, client, QDRANT_COLLECTION_NAME)
        logger.info("Embedding and upsert complete!")
    else:
        logger.warning("No records to embed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings and upsert to Qdrant")
    parser.add_argument("--recreate", action="store_true", help="Recreate collection if it exists")
    args = parser.parse_args()
    
    main(recreate=args.recreate)

