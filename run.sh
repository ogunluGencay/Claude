#!/bin/bash

# Course Materials RAG System - UV Runner

# Create necessary directories
mkdir -p docs

# Check if backend directory exists
if [ ! -d "backend" ]; then
    echo "Error: backend directory not found"
    exit 1
fi

# Sync dependencies with UV
echo "Syncing dependencies with UV..."
uv sync

echo ""
echo "Starting Course Materials RAG System..."
echo "Make sure you have set your ANTHROPIC_API_KEY in .env"
echo ""

# Start the server using UV
cd backend && uv run uvicorn app:app --reload --port 8000
