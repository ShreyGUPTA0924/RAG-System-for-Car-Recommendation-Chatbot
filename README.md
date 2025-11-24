# Car Recommendations RAG Chatbot Backend

A production-style RAG (Retrieval-Augmented Generation) chatbot backend for car recommendations using FastAPI, LangChain, Qdrant, and SentenceTransformers.

## Tech Stack

- **Backend**: FastAPI
- **Vector DB**: Qdrant (local Docker or cloud)
- **Embeddings**: SentenceTransformers
- **LLM**: Ollama (local) or hosted Llama API
- **Framework**: LangChain

## Project Structure

```
backend/
├── main.py                 # FastAPI application
├── rag/
│   ├── loader.py          # Data ingestion from Excel/CSV
│   ├── embed.py           # Embedding generation and Qdrant upsert
│   ├── retriever.py       # Qdrant retriever with metadata filters
│   ├── model.py           # LLM wrapper (Ollama/hosted)
│   └── chain.py           # ConversationalRetrievalChain setup
├── prompts/
│   └── base_prompt.txt    # Prompt template
├── data/
│   └── raw/
│       ├── cars.xlsx      # Car dataset (350 rows, ~80 columns)
│       └── faq.csv        # FAQ dataset
└── tests/
    └── test_end_to_end.py # End-to-end tests
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Required environment variables:
- `QDRANT_URL`: Qdrant server URL (default: http://localhost:6333)
- `QDRANT_API_KEY`: API key for cloud Qdrant (optional for local)
- `OLLAMA_URL`: Ollama server URL (default: http://localhost:11434)
- `OLLAMA_MODEL`: Model name (default: llama2)
- `HF_API_KEY`: HuggingFace API key (if using hosted model)
- `EMBEDDING_MODEL`: Embedding model name (default: sentence-transformers/all-MiniLM-L6-v2)

### 3. Start Qdrant (Local)

If using local Qdrant, run:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 4. Start Ollama (Local)

If using local Ollama:

```bash
ollama serve
ollama pull llama2
```

## Usage

### Data Ingestion

Run the ingestion pipeline to load and embed data:

**Windows:**
```bash
py -3 backend/rag/loader.py
py -3 backend/rag/embed.py --recreate
```

**Linux/Mac:**
```bash
python backend/rag/loader.py
python backend/rag/embed.py --recreate
```

Or use the Cursor task:
- Task: `ingest`

### Start Server

**Windows:**
```bash
py -3 -m uvicorn backend.main:app --reload --port 8000
```

**Linux/Mac:**
```bash
uvicorn backend.main:app --reload --port 8000
```

Or use the Cursor task:
- Task: `start`

### Run Tests

```bash
py -3 -m pytest -q
```

Or use the Cursor task:
- Task: `test`

## API Endpoints

### POST `/chat`

Chat endpoint for car recommendations.

**Request:**
```json
{
  "query": "I need a fuel-efficient sedan under $30,000",
  "filters": {
    "price_max": 30000,
    "body_type": "Sedan",
    "fuel_type": "Hybrid"
  },
  "session_id": "user-123"
}
```

**Response:**
```json
{
  "answer": "Based on your requirements, here are some recommendations...",
  "recommended": [
    {
      "make": "Toyota",
      "model": "Camry Hybrid",
      "price": 28000,
      ...
    }
  ],
  "sources": [
    {
      "make": "Toyota",
      "model": "Camry Hybrid",
      "score": 0.95
    }
  ]
}
```

## Cursor Tasks

The project includes Cursor tasks for common operations:

1. **ingest**: Run data ingestion and embedding pipeline
2. **start**: Start the FastAPI server
3. **test**: Run pytest tests

## Development

### Adding New Data

Place new Excel files in `backend/data/raw/` and run the ingestion pipeline.

### Tuning Parameters

Edit `.env` to adjust:
- `RETRIEVAL_K`: Number of documents to retrieve
- `LLM_TEMPERATURE`: LLM temperature
- `LLM_TOP_P`: LLM top-p sampling

## Troubleshooting

- **Qdrant connection errors**: Ensure Qdrant is running and `QDRANT_URL` is correct
- **Ollama errors**: Ensure Ollama is running and model is pulled
- **Embedding errors**: Check internet connection for model download
- **Import errors**: Ensure all dependencies are installed

