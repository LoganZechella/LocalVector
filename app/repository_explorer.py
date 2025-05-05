import os
from pathlib import Path
import logging
import chardet
import mimetypes
import re

logger = logging.getLogger(__name__)

class RepositoryExplorer:
    """Explorer for traversing and extracting content from repositories."""

    # Common file patterns to ignore
    DEFAULT_IGNORE_PATTERNS = [
        r"\.git/",
        r"__pycache__/",
        r"\.pytest_cache/",
        r"\.venv/",
        r"venv/",
        r"node_modules/",
        r"\.DS_Store",
        r"\.env",
        r"\.(jpg|jpeg|png|gif|bmp|ico|svg)$",
        r"\.(zip|tar|gz|rar|7z)$",
        r"\.(pyc|pyo|pyd)$",
        r"\.(so|dll|exe)$",
    ]

    # Maximum file size to process (in bytes)
    MAX_FILE_SIZE = 1024 * 1024  # 1MB

    def __init__(self, ignore_patterns=None):
        """Initialize with optional ignore patterns."""
        self.ignore_patterns = ignore_patterns or self.DEFAULT_IGNORE_PATTERNS
        self.ignore_regexes = [re.compile(pattern) for pattern in self.ignore_patterns]

    def explore_repository(self, repo_path):
        """Explore repository and extract content from all relevant files.

        Args:
            repo_path: Path to the repository

        Returns:
            List of dictionaries with file information
        """
        files = []

        # Walk through the repository
        for root, dirs, filenames in os.walk(repo_path):
            # Skip directories that match ignore patterns
            dirs[:] = [d for d in dirs if not self._should_ignore(os.path.join(root, d))]

            for filename in filenames:
                file_path = os.path.join(root, filename)

                # Skip files that match ignore patterns
                if self._should_ignore(file_path):
                    continue

                # Skip files that are too large
                if os.path.getsize(file_path) > self.MAX_FILE_SIZE:
                    logger.info(f"Skipping large file: {file_path}")
                    continue

                try:
                    # Process the file
                    file_info = self._process_file(file_path, repo_path)
                    if file_info:
                        files.append(file_info)
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")

        return files

    def _should_ignore(self, path):
        """Check if a path should be ignored."""
        rel_path = os.path.normpath(path)
        # Use os.path.sep for cross-platform compatibility in regex matching
        rel_path_for_regex = rel_path.replace(os.path.sep, '/')
        return any(regex.search(rel_path_for_regex) for regex in self.ignore_regexes)

    def _process_file(self, file_path, repo_root):
        """Process a file to extract its content and metadata."""
        try:
            # Detect file type
            mime_type, _ = mimetypes.guess_type(file_path)

            # Get relative path from repository root
            rel_path = os.path.relpath(file_path, repo_root)

            # Read file content with proper encoding detection
            content = self._read_file_content(file_path)

            return {
                "path": rel_path,
                "mime_type": mime_type,
                "content": content,
                "size": os.path.getsize(file_path),
                "extension": os.path.splitext(file_path)[1].lower(),
            }
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None

    def _read_file_content(self, file_path):
        """Read file content with encoding detection."""
        with open(file_path, 'rb') as f:
            raw_content = f.read()

        # Try to detect encoding
        result = chardet.detect(raw_content)
        encoding = result['encoding'] or 'utf-8'

        try:
            # Try to decode with detected encoding
            content = raw_content.decode(encoding)
        except UnicodeDecodeError:
            # Fall back to latin-1 which can decode any byte sequence
            logger.warning(f"Could not decode {file_path} with detected encoding {encoding}. Falling back to latin-1.")
            content = raw_content.decode('latin-1')

        return content 