# Car Recommendations RAG Chatbot - Setup Summary

## ✅ Project Complete

All components have been successfully created and are ready for use. The project is a production-style RAG (Retrieval-Augmented Generation) chatbot backend for car recommendations.

## 📁 Repository Structure

```
MINOR 1 PROJECT/
├── backend/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application with /chat endpoint
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── loader.py             # Excel/CSV data loader with normalization
│   │   ├── embed.py              # Embedding generation and Qdrant upsert
│   │   ├── retriever.py          # Qdrant retriever with metadata filters
│   │   ├── model.py              # LLM wrapper (Ollama/hosted)
│   │   └── chain.py               # ConversationalRetrievalChain (custom for LangChain 1.0+)
│   ├── prompts/
│   │   └── base_prompt.txt        # Prompt template for grounding
│   ├── data/
│   │   ├── raw/                   # Place your cars.xlsx and faq.csv here
│   │   └── processed/             # Processed JSON files (auto-generated)
│   └── tests/
│       ├── __init__.py
│       └── test_end_to_end.py     # End-to-end tests
├── .vscode/
│   └── tasks.json                 # Cursor/VS Code tasks
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variables template
├── README.md                      # Full documentation
└── .gitignore                     # Git ignore rules
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note for Windows:** If `python` doesn't work, use `py -3` instead.

### 2. Set Up Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

Edit `.env` with your configuration:
- `QDRANT_URL`: Default is `http://localhost:6333` (for local Qdrant)
- `OLLAMA_URL`: Default is `http://localhost:11434` (for local Ollama)
- `OLLAMA_MODEL`: Default is `llama2`

### 3. Start Qdrant (Required)

**Option A: Docker (Recommended)**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

**Option B: Cloud Qdrant**
- Set `QDRANT_URL` and `QDRANT_API_KEY` in `.env`

### 4. Start Ollama (Required for LLM)

**Option A: Local Ollama**
```bash
# Install Ollama from https://ollama.ai
ollama serve
ollama pull llama2
```

**Option B: Hosted API**
- Set `HF_API_KEY` and `HF_MODEL_ENDPOINT` in `.env`

### 5. Run Data Ingestion

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

**Or use Cursor task:** `ingest`

### 6. Start the Server

**Windows:**
```bash
py -3 -m uvicorn backend.main:app --reload --port 8000
```

**Linux/Mac:**
```bash
uvicorn backend.main:app --reload --port 8000
```

**Or use Cursor task:** `start`

### 7. Test the API

The server will be available at `http://localhost:8000`

**Test endpoint:**
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the best fuel-efficient cars?",
    "filters": {"price_max": 30000},
    "session_id": "test-123"
  }'
```

### 8. Run Tests

```bash
py -3 -m pytest -q
```

**Or use Cursor task:** `test`

## 📋 Cursor Tasks

The project includes three Cursor tasks (defined in `.vscode/tasks.json`):

1. **ingest**: Run data ingestion and embedding pipeline
2. **start**: Start the FastAPI server
3. **test**: Run pytest tests

Access these via Cursor's task runner (Ctrl+Shift+P → "Tasks: Run Task")

## 🔧 Key Features

### Data Loading (`backend/rag/loader.py`)
- Loads Excel files (`cars.xlsx`) and CSV files (`faq.csv`)
- Normalizes column names and converts numeric types
- Creates sample data if files are not found
- Saves processed JSON files

### Embedding (`backend/rag/embed.py`)
- Uses SentenceTransformers for embeddings
- Upserts to Qdrant vector database
- Supports local and cloud Qdrant
- `--recreate` flag to rebuild collection

### Retrieval (`backend/rag/retriever.py`)
- Qdrant-backed semantic search
- Metadata filtering (price, body_type, fuel_type, year)
- Configurable `k` parameter

### LLM Integration (`backend/rag/model.py`)
- Supports local Ollama
- Supports hosted Llama/HuggingFace API
- Automatic fallback between options

### Chain (`backend/rag/chain.py`)
- Custom ConversationalRetrievalChain for LangChain 1.0+
- Maintains conversation history
- Returns answer, recommended cars, and sources

### API (`backend/main.py`)
- FastAPI with `/chat` endpoint
- Accepts query, filters, and session_id
- Returns answer, recommended cars, and sources
- Session-based chain management

## 📝 API Usage

### POST `/chat`

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
  "answer": "Based on your requirements...",
  "recommended": [
    {
      "make": "Toyota",
      "model": "Camry Hybrid",
      "price": 28000,
      "body_type": "Sedan",
      "fuel_type": "Hybrid"
    }
  ],
  "sources": [
    {
      "content": "Make: Toyota | Model: Camry Hybrid...",
      "metadata": {...}
    }
  ]
}
```

## ⚠️ Important Notes

1. **Python Version**: Requires Python 3.10+. Tested with Python 3.13.

2. **Windows Compatibility**: Use `py -3` instead of `python` on Windows if needed.

3. **Qdrant Must Be Running**: The embedding and retrieval steps require Qdrant to be accessible.

4. **LLM Must Be Available**: Either Ollama must be running locally, or you must configure a hosted API.

5. **Sample Data**: If `cars.xlsx` and `faq.csv` are not found, the loader will create sample data for testing.

6. **LangChain 1.0+**: The project uses LangChain 1.0+ with a custom chain implementation for compatibility.

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'pandas'"
- Make sure you've installed dependencies: `pip install -r requirements.txt`
- On Windows, try: `py -3 -m pip install -r requirements.txt`

### "Connection refused" (Qdrant)
- Ensure Qdrant is running: `docker run -p 6333:6333 qdrant/qdrant`
- Check `QDRANT_URL` in `.env`

### "No LLM configured"
- Start Ollama: `ollama serve` and `ollama pull llama2`
- Or configure `HF_API_KEY` and `HF_MODEL_ENDPOINT` in `.env`

### Import errors with LangChain
- The project includes fallback imports for different LangChain versions
- If issues persist, check your LangChain version: `pip show langchain`

## 📚 Next Steps

1. **Add Your Data**: Place `cars.xlsx` (350 rows, ~80 columns) and `faq.csv` in `backend/data/raw/`

2. **Run Ingestion**: Execute the `ingest` task or run the scripts manually

3. **Start Server**: Execute the `start` task

4. **Test**: Use the `/chat` endpoint or run the test suite

5. **Customize**: Adjust prompts, retrieval parameters, and LLM settings in `.env`

## ✨ Summary

The project is fully functional and ready to use. All components are implemented:
- ✅ Data loading and normalization
- ✅ Embedding generation and vector storage
- ✅ Retrieval with metadata filtering
- ✅ LLM integration (Ollama/hosted)
- ✅ Conversational chain with history
- ✅ FastAPI endpoint
- ✅ End-to-end tests
- ✅ Cursor tasks for automation

Enjoy building your car recommendation chatbot! 🚗

