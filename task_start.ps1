# Task: Start - Start the FastAPI server
# Usage: .\task_start.ps1

Write-Host "=== Starting FastAPI Server ===" -ForegroundColor Cyan

# Activate venv
if (Test-Path .venv\Scripts\Activate.ps1) {
    .\.venv\Scripts\Activate.ps1
} else {
    Write-Host "Warning: .venv not found. Using system Python." -ForegroundColor Yellow
}

# Check if Qdrant is reachable
try {
    $response = Invoke-WebRequest -Uri "http://localhost:6333/collections" -TimeoutSec 2 -ErrorAction Stop
    Write-Host "Qdrant is reachable" -ForegroundColor Green
} catch {
    Write-Host "Warning: Qdrant is not reachable at http://localhost:6333" -ForegroundColor Yellow
    Write-Host "Please ensure Qdrant is running (docker run -p 6333:6333 qdrant/qdrant)" -ForegroundColor Yellow
}

# Start server
Write-Host "Starting server on http://localhost:8000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
python -m uvicorn backend.main:app --reload --port 8000





