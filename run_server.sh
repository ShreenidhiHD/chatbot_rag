#!/bin/bash
# Run the FastAPI server for the RAG Chat API

cd "$(dirname "$0")"
source .venv/bin/activate
python run_api.py
