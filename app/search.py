import logging
from typing import List, Dict, Any
from app.ingestion import vector_db # Import the initialized vector_db
from app.schemas import SearchResult

logger = logging.getLogger(__name__)

def search(query: str, k: int, filter: Dict[str, Any] = None) -> List[SearchResult]:
    """Perform similarity search on the vector database."""
    logger.info(f"Performing search for query: '{query}', k={k}, filter={filter}")
    try:
        # Use the vector_db instance directly
        docs = vector_db.similarity_search(
            query=query,
            k=k,
            filter=filter
        )

        results = [
            SearchResult(content=d.page_content, metadata=d.metadata)
            for d in docs
        ]
        logger.info(f"Found {len(results)} results for query: '{query}'")
        return results

    except Exception as e:
        logger.exception(f"Error during similarity search for query '{query}': {e}")
        # Re-raise or return an empty list/error status
        # Returning empty list for now
        return [] 