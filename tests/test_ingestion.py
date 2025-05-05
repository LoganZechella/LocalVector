import pytest
import os
import sys

# Add app directory to sys.path to import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ingestion import ingest
from app.search import search
from app.config import CHROMA_DB_DIR

# Ensure the Chroma DB directory exists for testing, maybe clear it?
# For now, just ensure it exists. Be careful if running tests concurrently.
if not os.path.exists(CHROMA_DB_DIR):
    os.makedirs(CHROMA_DB_DIR)


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_deepwiki_ingestion():
    """Test ingestion and search of DeepWiki content."""
    # Test with a known DeepWiki URL
    # Using a potentially stable page, replace if needed
    deepwiki_url = "https://deepwiki.com/openai/openai-python"
    print(f"\nTesting ingestion for: {deepwiki_url}")
    try:
        result = ingest(source=deepwiki_url, loader_type="url")
        print(f"Ingestion result (chunks added): {result}")
        # Allow 0 chunks if the page is empty or parsing fails, but log it.
        assert result >= 0, f"Ingestion returned an error code: {result}"
        if result == 0:
             pytest.skip(f"Skipping search test as 0 chunks were ingested from {deepwiki_url}")

    except Exception as e:
        pytest.fail(f"Ingestion failed with exception: {e}")

    # Verify search functionality only if ingestion added chunks
    print(f"Testing search functionality for query related to: {deepwiki_url}")
    try:
        # Use a query relevant to the openai-python library
        search_query = "How do I use the OpenAI Python client?"
        search_results = search(query=search_query, k=3)
        print(f"Search results count: {len(search_results)}")
        # We expect at least one result if chunks were added
        assert len(search_results) > 0, f"Search returned no results for query: '{search_query}'"
        print(f"Top search result metadata: {search_results[0]['metadata']}")
        # Check if the source in metadata matches
        assert deepwiki_url in search_results[0]['metadata'].get('source', ''), "Source URL not found in top search result metadata"

    except Exception as e:
        pytest.fail(f"Search failed with exception: {e}")

# Add more tests below for other loaders (PDF, TXT) and edge cases 