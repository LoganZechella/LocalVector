# Document ingestion logic 
import logging
import re # Added for DeepWiki cleanup
import time # Added for rate limiting
import base64 # Added for GitHub content decoding
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Set, Tuple, Optional
import uuid # Added for generating IDs
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from app.config import CHROMA_DB_DIR, OPENAI_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib
import os
import chromadb # Added for version logging
from urllib.parse import urlparse, urljoin # Added urljoin for crawler

# Import new modules
from app.github_handler import GitRepositoryHandler
from app.repository_explorer import RepositoryExplorer
from app.code_chunker import CodeChunker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base Loader Class
class BaseLoader(ABC):
    @abstractmethod
    def load(self) -> List[Document]:
        """Load documents from a source."""
        pass

# Web Loader
class WebLoader(BaseLoader):
    def __init__(self, url: str):
        self.url = url

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _fetch_html(self) -> str:
        try:
            response = requests.get(self.url, timeout=15) # Increased timeout slightly
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            # Try to detect encoding, default to utf-8
            response.encoding = response.apparent_encoding or 'utf-8'
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {self.url}: {e}")
            raise

    # Enhanced _parse_html method from notepad
    def _parse_html(self, html_content: str) -> str:
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove common boilerplate tags
        for tag in soup(['nav', 'footer', 'aside', 'script', 'style', 'header']):
            tag.decompose()

        # Specific handling for DeepWiki content
        is_deepwiki = "deepwiki.com" in self.url
        if is_deepwiki:
            logger.info(f"Detected DeepWiki URL ({self.url}), applying specialized parsing")

            # Look for DeepWiki-specific content containers
            main_content = (
                soup.find('div', class_='wiki-content') or
                soup.find('main', class_='documentation-content') or
                soup.find('article') or
                soup.find('div', {'id': 'content'}) or
                soup.find('div', class_='content') # Added common content class
            )

            # Special handling for code blocks and headers within the main content
            if main_content:
                # Preserve code blocks formatting
                for code_block in main_content.find_all(['pre', 'code']):
                    # Check if it's a block or inline - handle differently?
                    # For now, treat all as blocks for simplicity
                    if code_block.parent.name != 'pre': # Avoid double wrapping pre > code
                         code_text = code_block.get_text(strip=False) # Keep internal whitespace
                         # Basic check to avoid empty code blocks
                         if code_text.strip():
                              code_block.replace_with(f"\n```\n{code_text}\n```\n")
                         else:
                              code_block.decompose() # Remove empty code tags
                    elif code_block.name == 'pre':
                         code_text = code_block.get_text(strip=False)
                         if code_text.strip():
                              code_block.replace_with(f"\n```\n{code_text}\n```\n")
                         else:
                              code_block.decompose()

                # Find headers and add spacing
                for header in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                    header_text = header.get_text(strip=True)
                    # Add more spacing before larger headers
                    if header.name in ['h1', 'h2', 'h3']:
                        header.replace_with(f"\n\n## {header_text}\n\n") # Use markdown style
                    else:
                        header.replace_with(f"\n### {header_text}\n\n")
            else:
                 logger.warning(f"Could not find main content container for DeepWiki URL: {self.url}")
                 # Fallback to body if specific containers fail
                 main_content = soup.body
        else:
            # Default content extraction for regular web pages
            main_content = soup.find('main') or soup.find('article') or soup.find('div', role='main') or soup.body

        if main_content:
            # Extract text, trying to preserve paragraphs
            text = main_content.get_text(separator='\n', strip=True)
        else:
            # Absolute fallback
            text = soup.get_text(separator='\n', strip=True)

        # Clean up the text: remove excessive newlines and whitespace
        lines = (line.strip() for line in text.splitlines())
        # Join lines, but preserve blank lines between paragraphs to some extent
        cleaned_text = '\n'.join(lines)
        # Reduce multiple newlines to a maximum of two
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)

        # Additional DeepWiki-specific cleanup (applied AFTER general cleaning)
        if is_deepwiki:
            logger.debug(f"Applying DeepWiki specific regex cleanup to: {self.url}")
            # Remove common DeepWiki boilerplate text patterns more carefully
            patterns_to_remove = [
                r'Want a great API Testing tool.*?maximum productivity\?', # Apidog ad
                r'Subscribe to our newsletter.*?Apidog\.com', # Newsletter signup
                r'Improve documentation on GitHub', # Link to GitHub
                r'Table of Contents' # Often redundant if headers are parsed
            ]
            for pattern in patterns_to_remove:
                cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE | re.DOTALL).strip()
            # Reduce multiple newlines again after removals
            cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text).strip()

        return cleaned_text

    def load(self) -> List[Document]:
        logger.info(f"Loading content from URL: {self.url}")
        # Add DeepWiki specific logging here
        if "deepwiki.com" in self.url:
            logger.info(f"Processing DeepWiki URL: {self.url}")
            # Extract GitHub repo info for better context
            try:
                # Attempt to extract repo info assuming pattern deepwiki.com/user/repo/...
                path_part = self.url.split("deepwiki.com/", 1)[1]
                repo_info = "/".join(path_part.split("/")[:2]) # Get first two parts after domain
                logger.info(f"Associated Repository (estimated): {repo_info}")
            except IndexError:
                logger.warning(f"Could not parse repository info from DeepWiki URL: {self.url}")

        try:
            html_content = self._fetch_html()
            text_content = self._parse_html(html_content)
            if not text_content:
                 logger.warning(f"No text content extracted from {self.url}")
                 return []
            metadata = {"source": self.url, "type": "url"}
            return [Document(page_content=text_content, metadata=metadata)]
        except Exception as e:
            logger.error(f"Failed to load and parse {self.url}: {e}")
            return []

# Text File Loader
class TextFileLoader(BaseLoader):
    def __init__(self, path: str):
        self.path = path

    def load(self) -> List[Document]:
        logger.info(f"Loading content from text file: {self.path}")
        try:
            loader = TextLoader(self.path)
            return loader.load()
        except Exception as e:
            logger.error(f"Failed to load text file {self.path}: {e}")
            return []

# PDF Loader
class PDFLoader(BaseLoader):
    def __init__(self, path: str):
        self.path = path

    def load(self) -> List[Document]:
        logger.info(f"Loading content from PDF file: {self.path}")
        try:
            loader = PyPDFLoader(self.path)
            # PyPDFLoader loads pages as separate documents by default
            return loader.load()
        except Exception as e:
            logger.error(f"Failed to load PDF file {self.path}: {e}")
            return []

# Initialize Embeddings with fallback
try:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=OPENAI_API_KEY
    )
    logger.info("Using OpenAI model: text-embedding-3-large")
except Exception as e:
    logger.warning(f"Failed to initialize text-embedding-3-large: {e}. Falling back to text-embedding-3-small.")
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=OPENAI_API_KEY
        )
        logger.info("Using OpenAI model: text-embedding-3-small")
    except Exception as fallback_e:
        logger.error(f"Failed to initialize fallback model text-embedding-3-small: {fallback_e}")
        # Consider raising an exception or using a default local model here
        raise RuntimeError("Could not initialize any OpenAI embedding model.") from fallback_e

# Initialize ChromaDB Client
vector_db = Chroma(
    persist_directory=CHROMA_DB_DIR,
    embedding_function=embeddings
)

# Add version information logging at initialization
try:
    chroma_version = chromadb.__version__
except AttributeError:
    chroma_version = "unknown"
try:
    vector_db_class_name = type(vector_db).__name__
    vector_db_module = type(vector_db).__module__
except Exception:
    vector_db_class_name = "unknown"
    vector_db_module = "unknown"

logger.info(f"ChromaDB version: {chroma_version}")
logger.info(f"Using Chroma vector store class: {vector_db_module}.{vector_db_class_name}")
logger.info(f"ChromaDB persistence directory: {CHROMA_DB_DIR}")

# Loader Factory
def loader_factory(loader_type: str, source: str) -> BaseLoader | None: # Adjusted return type hint
    logger.debug(f"Creating loader of type '{loader_type}' for source: {source}")
    if loader_type == "url":
        return WebLoader(url=source)
    elif loader_type == "pdf":
        if not os.path.exists(source):
             raise FileNotFoundError(f"PDF file not found: {source}")
        return PDFLoader(path=source)
    elif loader_type == "text" or loader_type == "file": # Allow 'file' as alias
        if not os.path.exists(source):
             raise FileNotFoundError(f"Text file not found: {source}")
        return TextFileLoader(path=source)
    # elif loader_type in ["github", "docs"]:
    #     # These types are handled directly in the ingest function now or dedicated functions
    #     # logger.warning(f"Loader type '{loader_type}' should be handled by specific ingestion function.")
    #     # return None # Return None or raise error?
    else:
        # logger.error(f"Unsupported loader type: {loader_type}")
        # raise ValueError(f"Unsupported loader type: {loader_type}")
        # Let the main ingest handle this case if type detection fails
        return None

def _generate_doc_id(doc: Document) -> str:
    """Generate a consistent ID for a document chunk based on content and source."""
    source = doc.metadata.get("source", "unknown_source")
    # Use content hash for uniqueness within a source
    content_hash = hashlib.sha256(doc.page_content.encode('utf-8')).hexdigest()[:16]
    # Include chunk index if available for code chunks
    chunk_index = doc.metadata.get("chunk_index")
    if chunk_index is not None:
        return f"{source}::{content_hash}::chunk_{chunk_index}"
    else:
        # For documents loaded whole (like single web pages initially)
        return f"{source}::{content_hash}"

# Updated Text Splitter Initialization (more generic)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    add_start_index=True, # Add start index to metadata
)

# --- Existing Ingest Function (Modified slightly if needed) ---
def ingest(source: str, loader_type: str, token: Optional[str] = None) -> int:
    """Ingests a document source using the appropriate loader and adds it to the vector store.
    Handles URL, PDF, Text files. GitHub and Docs should use specific functions.
    Args:
        source: The source identifier (URL or file path).
        loader_type: The type of loader to use ('url', 'pdf', 'text', 'file').
        token: Optional token (e.g., for future authenticated sources).

    Returns:
        Number of chunks added to the database, or -1 on failure.
    """
    logger.info(f"Starting ingestion for source: {source} (type: {loader_type})")

    if loader_type in ["github", "docs"]:
        logger.error(f"Ingestion type '{loader_type}' requires specific function. Use /add_github_repo or /add_doc_site instead.")
        # Or call the specific functions here if desired, but API routes handle it now
        raise ValueError(f"Ingestion type '{loader_type}' not supported by generic ingest function.")

    try:
        # 1. Load Document(s)
        loader = loader_factory(loader_type, source)
        if not loader:
             # Try to auto-detect if factory returns None (e.g., file exists check?)
             if os.path.exists(source):
                 _, ext = os.path.splitext(source.lower())
                 if ext == '.pdf':
                     loader = PDFLoader(path=source)
                     loader_type = 'pdf'
                 elif ext in ['.txt', '.md', '.py', '.js', '.html', '.css']: # Add more text types
                     loader = TextFileLoader(path=source)
                     loader_type = 'text'
                 else:
                     raise ValueError(f"Could not auto-detect loader type for existing file: {source}")
             else:
                 raise ValueError(f"Unsupported or invalid source/type: {source} / {loader_type}")

        documents = loader.load()
        if not documents:
            logger.warning(f"No documents loaded from source: {source}")
            return 0 # Indicate success but no content added

        logger.info(f"Loaded {len(documents)} document(s) from {source} using {loader_type} loader.")

        # 2. Split Documents
        # Use the globally defined text_splitter
        split_docs = text_splitter.split_documents(documents)
        if not split_docs:
            logger.warning(f"No chunks generated after splitting for source: {source}")
            return 0

        logger.info(f"Split into {len(split_docs)} chunks.")

        # 3. Generate IDs and Prepare for Vector Store
        ids = [_generate_doc_id(doc) for doc in split_docs]
        texts = [doc.page_content for doc in split_docs]
        metadatas = [doc.metadata for doc in split_docs] # Metadata includes source, start_index etc.

        # 4. Add to Vector Store
        if texts:
            logger.info(f"Adding {len(texts)} chunks to ChromaDB...")
            vector_db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            # Persist is handled automatically by ChromaDB client usually, but explicit call can ensure writes
            # vector_db.persist() # Re-evaluate if needed based on Chroma client behavior
            logger.info(f"Successfully added {len(texts)} chunks from {source}.")
            return len(texts)
        else:
            logger.warning(f"No text content found to add for source: {source}")
            return 0

    except FileNotFoundError as e:
        logger.error(f"File not found during ingestion: {e}")
        raise e # Re-raise for API handler
    except ValueError as e:
        logger.error(f"Value error during ingestion: {e}")
        raise e # Re-raise for API handler
    except Exception as e:
        logger.exception(f"An unexpected error occurred during ingestion of {source}: {e}")
        return -1 # Indicate failure

# --- New GitHub Ingestion Function ---
def ingest_github_repository(repo_url: str, branch: str = "main", keep_repo: bool = False) -> int:
    """Ingest a GitHub repository by cloning it and processing its contents.

    Args:
        repo_url: URL of the GitHub repository
        branch: Branch to checkout (default: main)
        keep_repo: Whether to keep the cloned repository (default: False)

    Returns:
        Number of chunks ingested, or -1 on failure.
    """
    try:
        logger.info(f"Ingesting GitHub repository: {repo_url}, branch: {branch}")

        # Initialize handlers
        # Consider making CHUNK_SIZE/OVERLAP directly configurable in CodeChunker if needed
        git_handler = GitRepositoryHandler() # Uses temp dir by default
        repo_path = git_handler.clone_repository(repo_url, branch)

        repo_explorer = RepositoryExplorer() # Uses default ignore patterns
        files = repo_explorer.explore_repository(repo_path)
        logger.info(f"Explorer found {len(files)} processable files in {repo_url}")

        chunker = CodeChunker(max_chunk_size=CHUNK_SIZE, min_chunk_size=200, overlap=CHUNK_OVERLAP)

        # Process all files
        all_chunks = []
        processed_files_count = 0
        for file_info in files:
            if file_info: # Ensure file_info is not None (error occurred during processing)
                chunks = chunker.chunk_file(file_info)
                all_chunks.extend(chunks)
                processed_files_count += 1
            else:
                 # Error logged by explorer's _process_file
                 pass
        logger.info(f"Processed {processed_files_count} files, generating {len(all_chunks)} total chunks.")


        if not all_chunks:
             logger.warning(f"No chunks generated from repository: {repo_url}")
             # Clean up even if no chunks generated
             if not keep_repo:
                 git_handler.cleanup()
             return 0

        # Prepare data for vector store
        texts = [chunk["content"] for chunk in all_chunks]
        metadatas = [chunk["metadata"] for chunk in all_chunks]
        # Generate unique IDs for each chunk
        # Using uuid for simplicity, could also use hash of content + path + chunk_index
        ids = [str(uuid.uuid4()) for _ in all_chunks]

        # Add to vector database (using the global vector_db instance)
        logger.info(f"Adding {len(all_chunks)} chunks from {repo_url} to ChromaDB...")
        vector_db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        # vector_db.persist() # Evaluate if needed

        # Clean up if not keeping repo
        if not keep_repo:
            git_handler.cleanup()

        logger.info(f"Successfully ingested {len(all_chunks)} chunks from {repo_url}")
        return len(all_chunks)

    except Exception as e:
        logger.exception(f"Error ingesting GitHub repository {repo_url}: {e}")
        # Attempt cleanup even on error
        try:
            if not keep_repo and 'git_handler' in locals() and git_handler.repos_dir and os.path.exists(git_handler.repos_dir):
                logger.info("Attempting cleanup after error...")
                git_handler.cleanup()
        except Exception as cleanup_e:
            logger.error(f"Error during cleanup after failed ingestion: {cleanup_e}")
        return -1 # Indicate failure

# --- Existing Documentation Crawler Class (keep as is for now) ---
class DocumentationCrawler:
    """Crawls a documentation website and extracts text content."""

    def __init__(self, start_url: str, max_depth: int = 2, rate_limit_seconds: float = 0.5):
        self.start_url = start_url
        self.max_depth = max_depth
        self.rate_limit = rate_limit_seconds
        self.allowed_domain = urlparse(start_url).netloc
        self.visited_urls: Set[str] = set()
        logger.info(f"Initialized crawler for {start_url} (domain: {self.allowed_domain}, max_depth: {max_depth})")

    def _normalize_url(self, url: str, base_url: str) -> Optional[str]:
        """Normalize URL, ensure it's within the same domain and is HTTP/HTTPS."""
        try:
            # Join relative URLs with the base URL of the page they were found on
            abs_url = urljoin(base_url, url.strip())
            parsed_url = urlparse(abs_url)

            # Remove fragment identifiers
            abs_url_no_fragment = parsed_url._replace(fragment="").geturl()

            # Check scheme and domain
            if parsed_url.scheme not in self.allowed_schemes:
                logger.debug(f"Skipping URL with invalid scheme: {abs_url_no_fragment}")
                return None
            if parsed_url.netloc != self.base_domain:
                logger.debug(f"Skipping URL from different domain: {abs_url_no_fragment}")
                return None

            return abs_url_no_fragment
        except Exception as e:
            logger.warning(f"Error normalizing URL '{url}' based on '{base_url}': {e}")
            return None

    def _is_valid_content_type(self, headers: Dict[str, str]) -> bool:
        """Check if the content type is likely HTML."""
        content_type = headers.get("content-type", "").lower()
        return "html" in content_type

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
    def _fetch_page(self, url: str) -> str | None:
        """Fetches the content of a single page."""
        try:
            response = requests.get(url, timeout=15, headers={'User-Agent': 'LocalVectorCrawler/0.1'})
            response.raise_for_status()
            if not self._is_valid_content_type(response.headers):
                logger.debug(f"Skipping non-HTML content at {url} (Content-Type: {response.headers.get('content-type')})")
                return None
            response.encoding = response.apparent_encoding or 'utf-8'
            return response.text
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return None

    def _extract_content(self, html_content: str, url: str) -> str:
        """Extracts the main textual content from HTML (uses WebLoader's logic)."""
        # Reuse the parsing logic from WebLoader for consistency
        # Create a temporary WebLoader instance for this
        temp_loader = WebLoader(url=url)
        return temp_loader._parse_html(html_content)

    def _extract_links(self, html_content: str, base_url: str) -> Set[str]:
        """Extracts valid, normalized links from HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')
        links = set()
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            normalized_url = self._normalize_url(href, base_url)
            if normalized_url:
                links.add(normalized_url)
        return links

    def crawl(self) -> List[Document]:
        """Performs the crawl and returns extracted documents."""
        logger.info(f"Starting crawl from: {self.start_url}")
        documents = []
        queue: List[Tuple[str, int]] = [(self.start_url, 0)] # URL, depth

        while queue:
            current_url, current_depth = queue.pop(0)

            if current_url in self.visited_urls or current_depth > self.max_depth:
                continue

            logger.info(f"Crawling [Depth {current_depth}]: {current_url}")
            self.visited_urls.add(current_url)

            # Rate limiting
            time.sleep(self.rate_limit)

            html_content = self._fetch_page(current_url)
            if not html_content:
                continue

            # Extract content from the current page
            text_content = self._extract_content(html_content, current_url)
            if text_content:
                metadata = {"source": current_url, "type": "docs"}
                documents.append(Document(page_content=text_content, metadata=metadata))

            # Extract links if not at max depth
            if current_depth < self.max_depth:
                links = self._extract_links(html_content, current_url)
                for link in links:
                    if link not in self.visited_urls:
                        queue.append((link, current_depth + 1))

        logger.info(f"Crawl finished. Extracted content from {len(documents)} pages. Visited {len(self.visited_urls)} URLs.")
        return documents 