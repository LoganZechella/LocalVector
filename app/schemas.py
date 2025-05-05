# Pydantic schemas for API 
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

# --- GitHub Specific Input ---
class GitHubRepoInput(BaseModel):
    repo_url: str = Field(..., description="URL of the GitHub repository")
    branch: str = Field(default="main", description="Branch to checkout")
    keep_repo: bool = Field(default=False, description="Whether to keep the cloned repository after ingestion")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "repo_url": "https://github.com/anthropics/anthropic-sdk-python",
                    "branch": "main",
                    "keep_repo": False
                }
            ]
        }
    }

# --- Generic Source Input (Updated) ---
class SourceInput(BaseModel):
    source: str = Field(..., description="The source identifier (URL, file path, GitHub repository URL, documentation site URL)")
    type: str = Field(default="auto", description="Type of the source (github, docs, url, pdf, text, file, auto)")
    # GitHub specific options
    branch: Optional[str] = Field(default=None, description="Branch to checkout (for GitHub repositories, defaults to 'main')")
    keep_repo: Optional[bool] = Field(default=None, description="Whether to keep the cloned repository (for GitHub repositories, defaults to False)")
    # Documentation site specific options
    max_depth: Optional[int] = Field(default=None, description="Maximum crawl depth (for documentation sites, defaults to 3)")
    # Optional token (might be useful for private repos/sites later)
    token: Optional[str] = Field(default=None, description="Optional authentication token")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "source": "https://github.com/openai/openai-python",
                    "type": "github",
                    "branch": "main"
                },
                {
                    "source": "https://docs.python.org/3/",
                    "type": "docs",
                    "max_depth": 2
                },
                {
                    "source": "https://example.com/article.html",
                    "type": "url"
                },
                {
                    "source": "./data/report.pdf",
                    "type": "pdf"
                },
                {
                    "source": "https://github.com/another/repo",
                    "type": "auto" # Let the backend detect it's GitHub
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