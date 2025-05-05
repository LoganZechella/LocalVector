# Command Line Interface 
import click
import requests
import json
import logging
from app.config import API_URL # Import API URL from config

# Configure basic logging for the CLI
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """CLI tool to interact with the Local Documentation Vector DB API."""
    pass

@cli.command()
@click.argument("source")
@click.option("--type", default="url", help="Type of the source (url, pdf, text, file)")
def add(source, type):
    """Ingest a new document source (URL or local file path)."""
    click.echo(f"Attempting to add source: {source} (type: {type})")
    endpoint = f"{API_URL}/add_source"
    payload = {"source": source, "type": type}
    try:
        response = requests.post(endpoint, json=payload, timeout=60) # Increased timeout for potential ingestion time
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        click.echo("--- API Response ---")
        click.echo(json.dumps(response.json(), indent=2))
    except requests.exceptions.Timeout:
        click.echo(f"Error: Request timed out connecting to {endpoint}", err=True)
        logger.error(f"Timeout error for add command: {endpoint}")
    except requests.exceptions.ConnectionError:
        click.echo(f"Error: Could not connect to API at {endpoint}. Is the server running?", err=True)
        logger.error(f"Connection error for add command: {endpoint}")
    except requests.exceptions.HTTPError as e:
        click.echo(f"Error: API returned status code {e.response.status_code}", err=True)
        try:
            # Try to print the error detail from the API response
            click.echo(f"Detail: {json.dumps(e.response.json(), indent=2)}", err=True)
        except json.JSONDecodeError:
            click.echo(f"Response body: {e.response.text}", err=True)
        logger.error(f"HTTP error for add command: {e}")
    except requests.exceptions.RequestException as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
        logger.error(f"Unexpected request error for add command: {e}")

@cli.command()
@click.argument("query")
@click.option("--k", default=5, type=int, help="Number of results to return.")
# Placeholder for filter option - requires parsing string to dict
# @click.option("--filter", default=None, help='Metadata filter (JSON string e.g., '{"source": "some_url"}')')
def search(query, k):
    """Query the vector database for similar documents."""
    click.echo(f"Searching for: '{query}' (k={k})")
    endpoint = f"{API_URL}/query"
    payload = {"query": query, "k": k} # Add filter logic here if implemented

    try:
        response = requests.post(endpoint, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])

        if not results:
            click.echo("No results found.")
            return

        click.echo("--- Search Results ---")
        for i, r in enumerate(results, 1):
            click.echo(f"\n[{i}] Match Score (if available, lower is better): {r.get('distance', 'N/A')}")
            click.echo(f"    Source: {r.get('metadata', {}).get('source', 'Unknown')}")
            # Print other metadata if needed
            # click.echo(f"    Metadata: {json.dumps(r.get('metadata', {}), indent=6)}")
            click.echo(f"    Content: \n      {"".join(r.get('content', '').splitlines(True)[:5])}...") # Show first 5 lines
            click.echo("---")

    except requests.exceptions.Timeout:
        click.echo(f"Error: Request timed out connecting to {endpoint}", err=True)
        logger.error(f"Timeout error for search command: {endpoint}")
    except requests.exceptions.ConnectionError:
        click.echo(f"Error: Could not connect to API at {endpoint}. Is the server running?", err=True)
        logger.error(f"Connection error for search command: {endpoint}")
    except requests.exceptions.HTTPError as e:
        click.echo(f"Error: API returned status code {e.response.status_code}", err=True)
        try:
            click.echo(f"Detail: {json.dumps(e.response.json(), indent=2)}", err=True)
        except json.JSONDecodeError:
            click.echo(f"Response body: {e.response.text}", err=True)
        logger.error(f"HTTP error for search command: {e}")
    except requests.exceptions.RequestException as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
        logger.error(f"Unexpected request error for search command: {e}")

if __name__ == "__main__":
    cli() 