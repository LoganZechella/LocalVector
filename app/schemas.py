# Pydantic schemas for API 
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class SourceInput(BaseModel):
    source: str = Field(..., description="The source identifier (URL, file path)")
    type: str = Field(default="url", description="Type of the source (e.g., 'url', 'pdf', 'text', 'file')")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "source": "https://docs.trychroma.com/getting-started",
                    "type": "url"
                },
                {
                    "source": "./documents/my_document.pdf",
                    "type": "pdf"
                }
            ]
        }
    }

class QueryInput(BaseModel):
    query: str = Field(..., description="The text query for similarity search")
    k: int = Field(default=5, description="Number of results to return")
    filter: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata filter")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "How to install Chroma?",
                    "k": 3
                },
                {
                    "query": "Explain Chroma persistence",
                    "k": 5,
                    "filter": {"source": "https://docs.trychroma.com/"}
                }
            ]
        }
    }

# Schema for search results
class SearchResult(BaseModel):
    content: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    results: List[SearchResult] 