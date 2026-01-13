# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Run Commands

```bash
# Install dependencies
uv sync

# Run the application (from project root)
./run.sh
# Or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

The web interface runs at `http://localhost:8000` and API docs at `http://localhost:8000/docs`.

## Environment Setup

Requires a `.env` file in the root directory with:
```
ANTHROPIC_API_KEY=your_key_here
```

## Architecture Overview

This is a RAG (Retrieval-Augmented Generation) chatbot for querying course materials. The system uses ChromaDB for vector storage, sentence-transformers for embeddings, and Anthropic's Claude for response generation.

### Request Flow

1. **User Query** → FastAPI endpoint (`/api/query`) → `RAGSystem.query()`
2. **RAGSystem** orchestrates the flow:
   - Retrieves conversation history from `SessionManager`
   - Calls `AIGenerator` with the query and registered tools
3. **AIGenerator** sends request to Claude with tool definitions
4. **Claude decides** whether to use the `search_course_content` tool
5. If tool is used → `ToolManager.execute_tool()` → `CourseSearchTool.execute()` → `VectorStore.search()`
6. Tool results are sent back to Claude for final response generation
7. Response and sources returned to user; conversation history updated

### Key Components

- **`RAGSystem`** (`rag_system.py`): Main orchestrator that wires together all components
- **`VectorStore`** (`vector_store.py`): ChromaDB wrapper with two collections:
  - `course_catalog`: Course metadata for semantic name matching
  - `course_content`: Chunked course content for RAG retrieval
- **`AIGenerator`** (`ai_generator.py`): Handles Claude API calls with tool use loop
- **`CourseSearchTool`** (`search_tools.py`): Tool that Claude can invoke to search course content
- **`DocumentProcessor`** (`document_processor.py`): Parses course documents into structured chunks

### Document Format

Course documents (`.txt`, `.pdf`, `.docx` in `docs/` folder) are expected to follow this structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 0: [lesson title]
Lesson Link: [url]
[content...]

Lesson 1: [lesson title]
...
```

### Data Models (`models.py`)

- `Course`: Title, link, instructor, list of lessons
- `Lesson`: Number, title, link
- `CourseChunk`: Content chunk with course/lesson metadata for vector storage

### Configuration (`config.py`)

Key settings: `CHUNK_SIZE` (800), `CHUNK_OVERLAP` (100), `MAX_RESULTS` (5), `MAX_HISTORY` (2 messages), embedding model (`all-MiniLM-L6-v2`), Claude model (`claude-sonnet-4-20250514`).
