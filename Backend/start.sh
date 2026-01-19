#!/bin/bash
# Start Gunicorn with Uvicorn workers
# Use the PORT environment variable provided by Render, or default to 8000
PORT="${PORT:-8000}"
exec gunicorn api:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind "0.0.0.0:$PORT" --timeout 120
