Write-Host "Starting API server..." -ForegroundColor Cyan
uvicorn src.app_api:app --host 0.0.0.0 --port 8000 --reload
