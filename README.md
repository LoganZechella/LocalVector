# Local Documentation Vector Database

This project provides a simple, self-hosted vector database designed for ingesting documentation (from URLs, local files, or GitHub repositories) and performing semantic searches. It uses ChromaDB for the vector store, OpenAI embeddings, and exposes functionality via a FastAPI service and a command-line interface (CLI).

## Features

*   **Multi-Source Ingestion:**
    *   Load documents from web URLs.
    *   Ingest local PDF and plain text files.
    *   Clone and process public **GitHub Repositories** locally, avoiding API rate limits.
*   **Vector Storage:** Uses ChromaDB for efficient local vector storage and retrieval.
*   **Embeddings:** Leverages OpenAI's `text-embedding-3-large` (with fallback to `small`) for generating text embeddings.
*   **Code-Aware Chunking:** Intelligently chunks code files (Python, JavaScript) and Markdown based on structure (functions, classes, headers) for better context.
*   **Semantic Search:** Query ingested documents based on semantic similarity.
*   **API:** FastAPI service provides endpoints for ingestion and search.
*   **CLI:** A command-line interface for interacting with the service.
*   **Dockerized:** Easy deployment using Docker and Docker Compose.

## Prerequisites

*   Python 3.10+
*   Docker and Docker Compose
*   `git` command-line tool installed (required by the Docker container for cloning)
*   An OpenAI API Key

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd LocalVector
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the project root and add your OpenAI API key:
    ```dotenv
    OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # Optional: Override default ChromaDB path, chunk size, etc.
    # CHROMA_DB_DIR=./chroma_data
    # CHUNK_SIZE=1000
    # CHUNK_OVERLAP=200
    # API_URL=http://127.0.0.1:8000
    ```

## Running the Service (Docker)

Run the service using Docker Compose. Use `--build` the first time or when changes are made to the code or `Dockerfile`.

```bash
# Build image (if needed, use --no-cache to force rebuild)
docker-compose build [--no-cache]

# Start the service
docker-compose up
```

This will build the Docker image (if it doesn't exist or if `--build` is used) and start the FastAPI service. The service will be accessible at `http://127.0.0.1:8000` (use this IP instead of localhost if localhost doesn't resolve). The `chroma_db` directory will be mounted into the container for persistence.

To stop the service:
```bash
docker-compose down
```

## Using the CLI

Ensure your virtual environment is active (`source .venv/bin/activate`). The CLI interacts with the running FastAPI service (defaults to `http://127.0.0.1:8000`).

*   **Add a source (using generic `add` command):**
    ```bash
    # Add from a URL (auto-detects type=url)
    python cli.py add https://docs.trychroma.com/getting-started

    # Add from a PDF file (specify type or let it auto-detect)
    python cli.py add ./path/to/your/document.pdf --type pdf

    # Add from a text file
    python cli.py add ./path/to/your/notes.txt --type text

    # Add a GitHub repository (auto-detects type=github)
    python cli.py add https://github.com/langchain-ai/langchain

    # Add a GitHub repository (specific branch)
    python cli.py add https://github.com/langchain-ai/langchain --type github --branch release-candidate
    ```

*   **Add a GitHub repository (using dedicated `add_github` command):**
    ```bash
    python cli.py add_github https://github.com/anthropics/anthropic-sdk-python --branch main
    ```

*   **Search for documents:**
    ```bash
    python cli.py search "How do I install ChromaDB?" --k 3

    # Search with a metadata filter
    python cli.py search "What is an agent?" --filter '{"source_type": "github"}'
    ```
    Replace the query and optionally adjust the number of results (`-k`) or add a metadata filter.

## API Endpoints

When the service is running, you can access the interactive API documentation (Swagger UI) at `http://127.0.0.1:8000/docs`.

*   `POST /add_source`: Ingests a new document source (URL, file, GitHub repo, Docs site).
    *   Body: `SourceInput` schema (see `/docs` for details). Handles auto-detection.
*   `POST /add_github_repo`: Specifically ingests a GitHub repository.
    *   Body: `GitHubRepoInput` schema (`repo_url`, `branch`, `keep_repo`).
*   `POST /query`: Performs a similarity search.
    *   Body: `QueryInput` schema (`query`, `k`, `filter`).
*   `GET /health`: Health check endpoint.

## Project Structure

```
.
├── .env                  # Environment variables (OpenAI key, config)
├── .gitignore
├── Dockerfile            # Defines the service container
├── docker-compose.yml    # Orchestrates the Docker container
├── requirements.txt      # Python dependencies
├── cli.py                # Command Line Interface tool
├── README.md             # This file
├── app/                  # FastAPI service code
│   ├── __init__.py
│   ├── main.py           # FastAPI app definition, endpoints
│   ├── ingestion.py      # Document loading, splitting, embedding, storing logic
│   ├── schemas.py        # Pydantic models for API requests/responses
│   ├── config.py         # Loads configuration from .env
│   ├── search.py         # Search logic implementation
│   ├── github_handler.py # Handles cloning/updating Git repos
│   ├── repository_explorer.py # Explores local repo files
│   └── code_chunker.py   # Chunks code/text files intelligently
├── chroma_db/            # Default directory for persistent ChromaDB data (mounted via Docker)
└── tests/                # Placeholder for tests
```

## Configuration

Configuration is managed via the `.env` file:

*   `OPENAI_API_KEY`: **Required**. Your OpenAI API key.
*   `CHROMA_DB_DIR`: Path where ChromaDB persists data (Default: `./chroma_db`). Inside Docker, this is mapped to `/app/chroma_db_volume`.
*   `CHUNK_SIZE`: Target size for text chunks (Default: 1000).
*   `CHUNK_OVERLAP`: Overlap between text chunks (Default: 200).
*   `API_URL`: Base URL for the API service (Default: `http://127.0.0.1:8000`). Used by the CLI. 