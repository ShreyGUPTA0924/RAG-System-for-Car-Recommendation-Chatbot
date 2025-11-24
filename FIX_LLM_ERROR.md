# Fixing the 410 Gone Error

## Problem
The HuggingFace Inference API models are returning `410 Gone`, meaning the models are no longer available on their free inference API.

## Solution Options

### Option 1: Install Ollama (RECOMMENDED - Most Reliable)

1. **Download Ollama**: https://ollama.ai
2. **Install and start Ollama**:
   ```powershell
   # After installation, Ollama should start automatically
   # If not, run:
   ollama serve
   ```
3. **Download a model**:
   ```powershell
   ollama pull llama2
   # Or try other models:
   ollama pull mistral
   ollama pull codellama
   ```
4. **Update .env** (if needed):
   ```env
   OLLAMA_URL=http://localhost:11434
   OLLAMA_MODEL=llama2
   ```

The system will **automatically detect and use Ollama** once it's running!

### Option 2: Use a Different HuggingFace Model

1. Visit https://huggingface.co/models?pipeline_tag=text-generation
2. Find a model that supports inference API
3. Update `.env`:
   ```env
   HF_MODEL_ENDPOINT=https://api-inference.huggingface.co/models/MODEL_NAME
   ```

**Note**: Many free models on HuggingFace Inference API are being deprecated, so Option 1 (Ollama) is more reliable.

### Option 3: Use HuggingFace Text Generation Inference (Advanced)

For production, consider hosting your own model using HuggingFace's TGI server.

## Current Status

The system is configured to:
1. **Try Ollama first** (if available)
2. **Fall back to HuggingFace API** (if Ollama not available)
3. **Show clear error messages** with instructions

## Testing

After installing Ollama, test with:
```powershell
.\.venv\Scripts\Activate.ps1
python test_llm_fix.py
```

Or start the server:
```powershell
uvicorn backend.main:app --reload --port 8000
```


