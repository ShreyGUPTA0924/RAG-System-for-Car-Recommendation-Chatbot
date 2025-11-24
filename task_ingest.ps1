# Task: Ingest - Run data ingestion and embedding pipeline
# Usage: .\task_ingest.ps1

Write-Host "=== Running Data Ingestion ===" -ForegroundColor Cyan

# Activate venv
if (Test-Path .venv\Scripts\Activate.ps1) {
    .\.venv\Scripts\Activate.ps1
} else {
    Write-Host "Warning: .venv not found. Using system Python." -ForegroundColor Yellow
}

# Run loader
Write-Host "Step 1: Loading data..." -ForegroundColor Green
python backend/rag/loader.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: loader.py failed" -ForegroundColor Red
    exit 1
}

# Run embed
Write-Host "Step 2: Generating embeddings and upserting to Qdrant..." -ForegroundColor Green
python backend/rag/embed.py --recreate
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: embed.py failed" -ForegroundColor Red
    exit 1
}

Write-Host "=== Ingestion Complete ===" -ForegroundColor Cyan





