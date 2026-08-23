@echo off
REM Stop Enterprise GraphRAG application
REM Double-click this file to cleanly shut down all services

echo Stopping Enterprise GraphRAG...
echo.

docker compose down

if errorlevel 1 (
    echo ERROR: Failed to stop services
    pause
    exit /b 1
)

echo SUCCESS: All services stopped.
echo.
echo - Neo4j stopped
echo - FastAPI backend stopped
echo - Streamlit UI stopped
echo.
pause
