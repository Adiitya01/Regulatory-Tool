#!/bin/bash
# Start Gunicorn with Uvicorn workers
exec gunicorn api:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 120
