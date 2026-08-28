@echo off
REM Start Enterprise GraphRAG application with Docker Compose
REM Double-click this file to start Neo4j, FastAPI backend, and Streamlit UI

echo Starting Enterprise GraphRAG...
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker Desktop is not installed or not in PATH
    echo Please download Docker Desktop from: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

REM Start services
docker compose up -d

if errorlevel 1 (
    echo ERROR: Failed to start services
    pause
    exit /b 1
)

echo.
echo SUCCESS: Services started!
echo.
echo Services starting up:
echo   - Neo4j (localhost:7687)
echo   - FastAPI (localhost:8000)
echo   - Streamlit UI (localhost:8501)
echo.
echo Waiting for services to be ready... (this takes ~30 seconds)
timeout /t 10 /nobreak

REM Open browser to Streamlit UI
echo Opening browser to http://localhost:8501...
start http://localhost:8501

echo.
echo To stop the application, run: stop.bat
echo.
pause
