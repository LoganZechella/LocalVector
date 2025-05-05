# Local Documentation Vector Database

This project provides a simple, self-hosted vector database designed for ingesting documentation (from URLs or local files) and performing semantic searches. It uses ChromaDB for the vector store, OpenAI embeddings, and exposes functionality via a FastAPI service and a command-line interface (CLI).

## Features

*   **Document Ingestion:** Load documents from web URLs, PDF files, or plain text files.
*   **Vector Storage:** Uses ChromaDB for efficient local vector storage and retrieval.
*   **Embeddings:** Leverages OpenAI's `text-embedding-3-large` (with fallback to `small`) for generating text embeddings.
*   **Semantic Search:** Query ingested documents based on semantic similarity.
*   **API:** FastAPI service provides endpoints for ingestion and search.
*   **CLI:** A command-line interface for interacting with the service.
*   **Dockerized:** Easy deployment using Docker and Docker Compose.

## Prerequisites

*   Python 3.10+
*   Docker and Docker Compose
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

The easiest way to run the service is using Docker Compose:

```bash
docker-compose up --build
```

This will build the Docker image (if it doesn't exist) and start the FastAPI service. The service will be accessible at `http://localhost:8000`. The `chroma_db` directory will be mounted into the container for persistence.

To stop the service:
```bash
docker-compose down
```

## Using the CLI

Ensure your virtual environment is active (`source .venv/bin/activate`). The CLI interacts with the running FastAPI service.

*   **Add a document source:**
    ```bash
    # Add from a URL (default type)
    python cli.py add https://docs.trychroma.com/getting-started

    # Add from a PDF file
    python cli.py add ./path/to/your/document.pdf --type pdf

    # Add from a text file
    python cli.py add ./path/to/your/notes.txt --type text
    ```

*   **Search for documents:**
    ```bash
    python cli.py search "How do I install ChromaDB?" --k 3
    ```
    Replace the query and optionally adjust the number of results (`-k`).

## API Endpoints

When the service is running, you can access the interactive API documentation (Swagger UI) at `http://localhost:8000/docs`.

*   `POST /add_source`: Ingests a new document source.
    *   Body: `{"source": "...", "type": "..."}`
*   `POST /query`: Performs a similarity search.
    *   Body: `{"query": "...", "k": 5, "filter": null}`
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
│   └── search.py         # Search logic implementation
├── chroma_db/            # Default directory for persistent ChromaDB data (mounted via Docker)
└── tests/                # Placeholder for tests
```

## Configuration

Configuration is managed via the `.env` file:

*   `OPENAI_API_KEY`: **Required**. Your OpenAI API key.
*   `CHROMA_DB_DIR`: Path where ChromaDB persists data (Default: `./chroma_db`).
*   `CHUNK_SIZE`: Target size for text chunks (Default: 1000).
*   `CHUNK_OVERLAP`: Overlap between text chunks (Default: 200).
*   `API_URL`: Base URL for the API service (Default: `http://127.0.0.1:8000`). Used by the CLI. 