# FastAPI application entry point 
import logging
import os # Added for path operations
from urllib.parse import urlparse # Added for source type detection

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import SourceInput, QueryInput, QueryResponse, GitHubRepoInput
from app.ingestion import ingest, ingest_github_repository, ingest_documentation_site
from app.search import search
from app.config import API_TITLE, API_VERSION

# Configure logging (if not already configured globally)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=API_TITLE,
    description="API for ingesting and querying local documentation.",
    version=API_VERSION,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for simplicity, restrict in production
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods
    allow_headers=["*"], # Allow all headers
)

# --- Exception Handlers ---
@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    logger.error(f"ValueError: {exc}")
    return JSONResponse(
        status_code=400,
        content={"message": f"Invalid input: {exc}"},
    )

@app.exception_handler(FileNotFoundError)
async def file_not_found_error_handler(request: Request, exc: FileNotFoundError):
    logger.error(f"FileNotFoundError: {exc}")
    return JSONResponse(
        status_code=404,
        content={"message": f"File not found: {exc}"},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception: {exc}") # Log the full traceback
    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected internal server error occurred."},
    )

# --- Helper Function for Source Type Detection ---
def detect_source_type(source: str) -> str:
    """Attempt to automatically detect the source type."""
    logger.debug(f"Attempting to detect source type for: {source}")
    parsed_url = urlparse(source)

    # Check for GitHub URLs
    if parsed_url.netloc == "github.com":
        logger.info(f"Detected GitHub source type for: {source}")
        return "github"

    # Check for local file paths
    if os.path.exists(source):
        _, ext = os.path.splitext(source.lower())
        if ext == '.pdf':
            logger.info(f"Detected PDF file type for: {source}")
            return "pdf"
        elif ext in ['.txt', '.md', '.py', '.js', '.html', '.css']: # Common text file types
            logger.info(f"Detected text file type for: {source}")
            return "text"
        else:
            logger.warning(f"Existing file with unrecognized extension, treating as generic 'file': {source}")
            return "file" # Or maybe raise an error?

    # Check for general URLs (assume documentation site or simple web page)
    if parsed_url.scheme in ["http", "https"]:
        # Basic heuristic: if it looks like a docs site path, assume 'docs'
        # This is very basic and could be improved
        if any(part in parsed_url.path.lower() for part in ["/docs", "/documentation", "/api", "/guide"]):
             logger.info(f"Detected potential 'docs' source type for URL: {source}")
             return "docs"
        else:
             logger.info(f"Detected generic 'url' source type for: {source}")
             return "url"

    logger.warning(f"Could not automatically determine source type for: {source}. Defaulting to 'url'.")
    return "url" # Default fallback

# --- API Endpoints ---

# New endpoint specifically for GitHub repos
@app.post("/add_github_repo")
async def add_github_repo(payload: GitHubRepoInput):
    """Endpoint to ingest a GitHub repository."""
    logger.info(f"Received request to add GitHub repository: {payload.repo_url}, branch: {payload.branch}")
    try:
        # Call the specific ingestion function
        count = ingest_github_repository(repo_url=payload.repo_url, branch=payload.branch, keep_repo=payload.keep_repo)

        if count > 0:
            return {"status": "success", "message": f"Successfully ingested {count} chunks from {payload.repo_url}."}
        elif count == 0:
             return {"status": "success", "message": f"Repository {payload.repo_url} processed, but no new content was added (potentially empty or filtered)."}
        else: # count == -1 indicates an unexpected error during ingest
             logger.error(f"GitHub ingestion function returned failure code for {payload.repo_url}")
             raise HTTPException(status_code=500, detail=f"Ingestion failed for GitHub repository {payload.repo_url}. Check server logs.")

    except RuntimeError as e:
        # Catch specific runtime errors from the handler (e.g., git clone failed)
        logger.error(f"Runtime error during GitHub ingestion for {payload.repo_url}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error in /add_github_repo endpoint for {payload.repo_url}")
        raise HTTPException(status_code=500, detail=f"Internal server error during GitHub ingestion: {str(e)}")


# Updated endpoint for generic sources
@app.post("/add_source")
async def add_source(payload: SourceInput):
    """Endpoint to ingest a new source document (URL, file path, GitHub repo, Docs site)."""
    logger.info(f"Received request to add source: {payload.source} (type: {payload.type}) Options: branch={payload.branch}, keep={payload.keep_repo}, depth={payload.max_depth}")

    source_type = payload.type
    if source_type == "auto":
        source_type = detect_source_type(payload.source)
        logger.info(f"Auto-detected source type as: {source_type}")

    try:
        count = -1 # Default to error/unhandled

        if source_type == "github":
            # Use defaults from schema if not provided in payload
            branch = payload.branch or "main"
            keep_repo = payload.keep_repo if payload.keep_repo is not None else False
            count = ingest_github_repository(repo_url=payload.source, branch=branch, keep_repo=keep_repo)

        elif source_type == "docs":
            max_depth = payload.max_depth or 3 # Default crawl depth
            count = ingest_documentation_site(doc_url=payload.source, max_depth=max_depth)

        elif source_type in ["url", "pdf", "text", "file"]:
            # Use the original ingest function for these types
            count = ingest(source=payload.source, loader_type=source_type, token=payload.token)

        else:
            # If auto-detection failed or invalid type provided
            raise ValueError(f"Unsupported source type: '{source_type}'")

        # Process result
        if count > 0:
            return {"status": "success", "message": f"Successfully ingested {count} chunks from {payload.source} (type: {source_type})."}
        elif count == 0:
             return {"status": "success", "message": f"Source {payload.source} (type: {source_type}) processed, but no new content was added."}
        else: # count == -1 indicates an unexpected error
             logger.error(f"Ingestion function returned failure code for {payload.source} (type: {source_type})" )
             raise HTTPException(status_code=500, detail=f"Ingestion failed for {payload.source}. Check server logs.")

    except (FileNotFoundError, ValueError, PermissionError, RuntimeError) as e:
        # Re-raise specific errors for FastAPI exception handlers
        logger.warning(f"{type(e).__name__} during ingestion for {payload.source}: {e}")
        if isinstance(e, FileNotFoundError):
            raise HTTPException(status_code=404, detail=str(e))
        else: # ValueError, PermissionError, RuntimeError (e.g., git failed)
             raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error adding source {payload.source}: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected internal server error occurred: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_docs(payload: QueryInput):
    """Endpoint to query the vector database."""
    logger.info(f"Received query: '{payload.query}' (k={payload.k}, filter={payload.filter})")
    try:
        results = search(query=payload.query, k=payload.k, filter=payload.filter)
        return QueryResponse(results=results)
    except Exception as e:
        logger.exception(f"Unexpected error in /query endpoint for query '{payload.query}'")
        raise HTTPException(status_code=500, detail="Internal server error during query.")

@app.get("/health", status_code=200)
async def health():
    """Health check endpoint."""
    # Basic health check, could be expanded (e.g., check DB connection)
    return {"status": "ok"} 