# Command Line Interface 
import click
import requests
import json
import os
from urllib.parse import urlparse

# Assuming config is now in app.config
# Need to make sure Python path allows importing from app
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from app.config import API_URL
except ImportError:
    # Fallback if running CLI without package install or specific env setup
    API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Configure basic logging for the CLI
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_source_type(source: str) -> str:
    """Auto-detect source type based on the source string."""
    parsed_url = urlparse(source)
    # Check for GitHub URLs
    if parsed_url.netloc == "github.com":
        return "github"
    # Check for local file paths
    if os.path.exists(source): # Use os.path.exists which works for files and dirs
        if os.path.isfile(source):
            _, ext = os.path.splitext(source.lower())
            if ext == ".pdf":
                return "pdf"
            elif ext in ['.txt', '.md', '.py', '.js', '.ts', '.rst', '.html', '.css']: # Common text types
                return "text"
            else:
                # Treat unknown file extensions as generic file
                click.echo(f"Warning: Treating file with unknown extension '{ext}' as generic 'file'.", err=True)
                return "file"
        elif os.path.isdir(source):
            # Could potentially add directory ingestion type later
            click.echo(f"Warning: Source '{source}' is a directory. Directory ingestion not yet implemented.", err=True)
            # What should we return here? Maybe raise error? For now, fallback.
            return "unknown_dir"
    # Check for general URLs
    if parsed_url.scheme in ["http", "https"]:
         # Assume 'docs' or 'url' based on simple heuristics (can be refined)
         if any(part in parsed_url.path.lower() for part in ["/docs", "/documentation", "/api", "/guide"]):
              return "docs"
         else:
              return "url"

    # Fallback if none of the above match
    click.echo(f"Warning: Could not determine type for '{source}'. Please specify type explicitly.", err=True)
    return "unknown"

@click.group()
def cli():
    """Command Line Interface for the Local Vector Database Service."""
    pass

@cli.command()
@click.argument("source")
@click.option("--type", default="auto",
              type=click.Choice(['auto', 'github', 'docs', 'url', 'pdf', 'text', 'file'], case_sensitive=False),
              help="Source type (auto, github, docs, url, pdf, text, file)." )
# GitHub specific options
@click.option("--branch", default=None, help="Branch for GitHub repos.")
@click.option("--keep-repo", is_flag=True, help="Keep cloned GitHub repo.")
# Docs specific options
@click.option("--max-depth", default=None, type=int, help="Crawl depth for docs sites.")
# General options
@click.option("--token", default=None, help="Optional auth token (usage depends on source type).")
def add(source, type, branch, keep_repo, max_depth, token):
    """Ingest a new source (detects type or uses specified type)."""
    click.echo(f"Attempting to add source: {source} (specified type: {type})", nl=False)

    detected_type = type
    if type == 'auto':
        detected_type = detect_source_type(source)
        if detected_type in ["unknown", "unknown_dir"]:
             click.echo(f" -> Auto-detection failed or yielded unsupported type: '{detected_type}'", err=True)
             click.echo("Please specify a valid type using --type.", err=True)
             return
        click.echo(f" -> auto-detected as: {detected_type}", nl=True)
    else:
        click.echo("") # Newline if type was specified

    # Resolve file paths if applicable
    if detected_type in ['pdf', 'text', 'file'] and not os.path.isabs(source):
        abs_path = os.path.abspath(source)
        if not os.path.exists(abs_path):
             click.echo(f"Error: File not found at resolved path: {abs_path}", err=True)
             return
        source = abs_path # Use the absolute path for the payload
        click.echo(f"Using absolute path for file: {source}")

    # Build payload for the generic /add_source endpoint
    payload = {"source": source, "type": detected_type}
    options_added = []
    if detected_type == "github":
        if branch: payload["branch"] = branch; options_added.append(f"branch={branch}")
        if keep_repo: payload["keep_repo"] = keep_repo; options_added.append("keep_repo=True")
    elif detected_type == "docs":
        if max_depth is not None: payload["max_depth"] = max_depth; options_added.append(f"max_depth={max_depth}")

    if token:
        payload["token"] = token
        options_added.append("token=******")

    if options_added:
         click.echo(f"Applying options: { ", ".join(options_added)}")

    endpoint = f"{API_URL}/add_source"
    click.echo(f"Sending request to {endpoint}...")

    try:
        # Use a long timeout suitable for various ingestion types
        response = requests.post(endpoint, json=payload, timeout=600) # 10 minutes timeout
        response.raise_for_status()

        click.echo("--- API Response ---")
        click.echo(json.dumps(response.json(), indent=2))

    except requests.exceptions.RequestException as e:
        handle_api_error(e)
    except Exception as e:
        click.echo(f"\nAn unexpected error occurred: {e}", err=True)

@cli.command()
@click.argument("query")
@click.option("-k", "--k", default=3, type=int, help="Number of results to return.")
@click.option("--filter", default=None, help="Optional JSON string for metadata filtering (e.g., '{\"source\":\"url\"}').")
def search(query, k, filter):
    """Search the vector database for a given query."""
    click.echo(f"Searching for: '{query}' (k={k})")
    endpoint = f"{API_URL}/query"

    payload = {"query": query, "k": k}
    if filter:
        try:
            payload["filter"] = json.loads(filter)
            click.echo(f"Applying filter: {payload['filter']}")
        except json.JSONDecodeError:
            click.echo(f"Error: Invalid JSON format for filter: {filter}", err=True)
            return

    click.echo(f"Sending request to {endpoint}...")
    try:
        response = requests.post(endpoint, json=payload, timeout=60) # 60 sec timeout for search
        response.raise_for_status()
        results = response.json().get("results", [])

        click.echo("--- Search Results ---")
        if not results:
            click.echo("No results found.")
            return

        for i, r in enumerate(results, 1):
            click.echo(f"\n--- Result {i} ---")
            metadata_str = json.dumps(r.get('metadata', {}))
            click.echo(f"Metadata: {metadata_str}")
            click.echo(f"Content: \n{r.get('content', '')[:500]}...")

    except requests.exceptions.RequestException as e:
        handle_api_error(e)
    except Exception as e:
        click.echo(f"\nAn unexpected error occurred during search: {e}", err=True)

# --- Utility for Handling API Errors --- 
def handle_api_error(e: requests.exceptions.RequestException):
    """Provides consistent error reporting for API calls."""
    if isinstance(e, requests.exceptions.ConnectionError):
        click.echo(f"\nError: Could not connect to the API service at {API_URL}. Is it running?", err=True)
    elif isinstance(e, requests.exceptions.Timeout):
        timeout = e.request.timeout if hasattr(e.request, 'timeout') else 'unknown'
        click.echo(f"\nError: Request timed out after {timeout} seconds. Ingestion/Search might be taking too long or the service is unresponsive.", err=True)
    elif isinstance(e, requests.exceptions.HTTPError):
        click.echo(f"\nError: API request failed with status {e.response.status_code}.", err=True)
        try:
            # Try to print the detail from the JSON error response
            error_detail = e.response.json().get("detail", "No detail provided.")
            click.echo(f"Detail: {error_detail}", err=True)
        except json.JSONDecodeError:
            # Fallback if response is not JSON
            click.echo(f"Response body: {e.response.text}", err=True)
    else:
        # Catch-all for other request exceptions
        click.echo(f"\nAn unexpected network or request error occurred: {e}", err=True)

# --- ADD GITHUB Command (New) ---
@cli.command()
@click.argument("repo_url")
@click.option("--branch", default="main", help="Branch to checkout.")
@click.option("--keep-repo", is_flag=True, help="Keep the cloned repository after processing.")
def add_github(repo_url, branch, keep_repo):
    """Ingest a GitHub repository directly via the dedicated endpoint."""
    click.echo(f"Adding GitHub repository: {repo_url}, branch: {branch}")
    endpoint = f"{API_URL}/add_github_repo"
    payload = {"repo_url": repo_url, "branch": branch, "keep_repo": keep_repo}

    click.echo(f"Sending request to {endpoint}...")
    try:
        # Use a long timeout suitable for cloning/processing large repos
        response = requests.post(endpoint, json=payload, timeout=600) # 10 minutes timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        click.echo("--- API Response ---")
        click.echo(json.dumps(response.json(), indent=2))

    except requests.exceptions.RequestException as e:
        handle_api_error(e)
    except Exception as e:
        click.echo(f"\nAn unexpected error occurred: {e}", err=True)

if __name__ == "__main__":
    cli() 