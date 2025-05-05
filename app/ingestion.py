# Document ingestion logic 
import logging
from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from app.config import CHROMA_DB_DIR, OPENAI_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib
import os

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
            response = requests.get(self.url, timeout=10)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {self.url}: {e}")
            raise

    def _parse_html(self, html_content: str) -> str:
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove common boilerplate tags
        for tag in soup(['nav', 'footer', 'aside', 'script', 'style', 'header']):
            tag.decompose()

        # Get main content text, trying common main content containers
        main_content = soup.find('main') or soup.find('article') or soup.find('div', role='main') or soup.body
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
        else:
            text = soup.get_text(separator='\n', strip=True) # Fallback

        # Simple cleaning (can be expanded)
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        cleaned_text = '\n'.join(chunk for chunk in chunks if chunk)
        return cleaned_text

    def load(self) -> List[Document]:
        logger.info(f"Loading content from URL: {self.url}")
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

logger.info(f"ChromaDB initialized. Persistence directory: {CHROMA_DB_DIR}")

# Loader Factory
def loader_factory(loader_type: str, source: str) -> BaseLoader:
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
    # Add DirectoryLoader later if needed
    else:
        logger.error(f"Unsupported loader type: {loader_type}")
        raise ValueError(f"Unsupported loader type: {loader_type}")

# Ingestion Function
def ingest(source: str, loader_type: str):
    """Load, split, and ingest documents into the vector store."""
    try:
        loader = loader_factory(loader_type, source)
        docs = loader.load()
        if not docs:
            logger.warning(f"No documents loaded from source: {source}")
            return 0

        chunks = text_splitter.split_documents(docs)
        if not chunks:
            logger.warning(f"No chunks created after splitting for source: {source}")
            return 0

        logger.info(f"Prepared {len(chunks)} chunks for ingestion from {source}")

        # Prepare for upsert
        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]

        # Generate deterministic IDs based on source and chunk content
        ids = []
        for i, chunk in enumerate(chunks):
            content_hash = hashlib.sha256(chunk.page_content.encode('utf-8')).hexdigest()[:8] # Short hash
            # Combine source (or filename) and content hash for a more unique ID
            source_identifier = os.path.basename(source) if loader_type != 'url' else source
            deterministic_id = f"{source_identifier}::{i}::{content_hash}"
            ids.append(deterministic_id)
            # Update metadata with the generated ID for potential future use
            if 'id' not in metadatas[i]: # Avoid overwriting existing ID if any
                 metadatas[i]['id'] = deterministic_id

        logger.debug(f"Generated {len(ids)} deterministic IDs for upsert.")

        # Upsert into Chroma
        vector_db.upsert(ids=ids, documents=texts, metadatas=metadatas)
        # Persist changes immediately after upsert
        # vector_db.persist() # Persist is often handled automatically by Chroma client in recent versions, but explicit call ensures it.
        # Re-checking Chroma docs/behavior - Langchain wrapper might require explicit persist.
        # Let's keep it explicit for safety for now.
        vector_db.persist()
        logger.info(f"Successfully upserted {len(ids)} chunks from {source}. Persistence triggered.")

        return len(ids)

    except FileNotFoundError as e:
         logger.error(f"Ingestion failed: {e}")
         raise # Re-raise specific expected errors
    except ValueError as e:
         logger.error(f"Ingestion failed: {e}")
         raise # Re-raise specific expected errors
    except Exception as e:
        logger.exception(f"An unexpected error occurred during ingestion for source {source}: {e}")
        # Depending on requirements, you might want to raise e or return an error status
        return -1 # Indicate failure

# Initialize Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    is_separator_regex=False,
    separators=["\n\n", "\n", ". ", " ", ""] # Added common separators
)
logger.info(f"Text splitter initialized with chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}") 