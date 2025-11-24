# Task: Status - Check Qdrant and server status
# Usage: .\task_status.ps1

Write-Host "=== System Status Check ===" -ForegroundColor Cyan

# Check Qdrant
Write-Host "`n1. Checking Qdrant..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:6333/collections" -TimeoutSec 2 -ErrorAction Stop
    $collections = ($response.Content | ConvertFrom-Json).result.collections
    Write-Host "   Qdrant: OK (HTTP $($response.StatusCode))" -ForegroundColor Green
    Write-Host "   Collections: $($collections.Count)" -ForegroundColor Green
    if ($collections.Count -gt 0) {
        foreach ($col in $collections) {
            Write-Host "     - $($col.name): $($col.points_count) points" -ForegroundColor Gray
        }
    }
} catch {
    Write-Host "   Qdrant: NOT REACHABLE" -ForegroundColor Red
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "   Action: Start Qdrant with: docker run -p 6333:6333 qdrant/qdrant" -ForegroundColor Yellow
}

# Check FastAPI server
Write-Host "`n2. Checking FastAPI server..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 2 -ErrorAction Stop
    Write-Host "   Server: OK (HTTP $($response.StatusCode))" -ForegroundColor Green
    
    # Test /chat endpoint
    Write-Host "`n3. Testing /chat endpoint..." -ForegroundColor Yellow
    $body = @{
        query = "What are fuel-efficient cars?"
        filters = @{}
        session_id = "test-status"
    } | ConvertTo-Json
    
    try {
        $chatResponse = Invoke-WebRequest -Uri "http://localhost:8000/chat" -Method POST -Body $body -ContentType "application/json" -TimeoutSec 10 -ErrorAction Stop
        $chatData = $chatResponse.Content | ConvertFrom-Json
        Write-Host "   /chat endpoint: OK" -ForegroundColor Green
        Write-Host "   Answer length: $($chatData.answer.Length) chars" -ForegroundColor Gray
        Write-Host "   Recommendations: $($chatData.recommended.Count)" -ForegroundColor Gray
    } catch {
        Write-Host "   /chat endpoint: ERROR" -ForegroundColor Red
        Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Red
    }
} catch {
    Write-Host "   Server: NOT RUNNING" -ForegroundColor Red
    Write-Host "   Action: Start server with: .\task_start.ps1" -ForegroundColor Yellow
}

Write-Host "`n=== Status Check Complete ===" -ForegroundColor Cyan





