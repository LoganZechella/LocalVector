# FastAPI application entry point 
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import SourceInput, QueryInput, QueryResponse
from app.ingestion import ingest
from app.search import search

# Configure logging (if not already configured globally)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Local Documentation Vector DB",
    description="API for ingesting and querying local documentation.",
    version="0.1.0",
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

# --- API Endpoints ---
@app.post("/add_source", status_code=201)
async def add_source(payload: SourceInput):
    """Endpoint to ingest a new source document (URL, file path)."""
    logger.info(f"Received request to add source: {payload.source} (type: {payload.type})")
    try:
        count = ingest(source=payload.source, loader_type=payload.type)
        if count > 0:
            return {"status": "success", "message": f"Successfully ingested {count} chunks from {payload.source}."}
        elif count == 0:
             return {"status": "success", "message": f"Source {payload.source} loaded, but no new chunks were added (possibly empty or already processed)."}
        else: # count == -1 indicates an unexpected error during ingest
             raise HTTPException(status_code=500, detail=f"Ingestion failed for {payload.source}. Check server logs.")

    except (ValueError, FileNotFoundError) as e:
        # These are re-raised from ingest and caught by exception handlers
        raise e
    except Exception as e:
        logger.exception(f"Unexpected error in /add_source endpoint for {payload.source}")
        raise HTTPException(status_code=500, detail="Internal server error during ingestion.")

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