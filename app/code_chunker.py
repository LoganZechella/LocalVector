import logging
from typing import List, Dict, Any, Optional
import re

logger = logging.getLogger(__name__)

class CodeChunker:
    """Chunk code files into meaningful segments."""

    def __init__(self, max_chunk_size=1500, min_chunk_size=200, overlap=50):
        """Initialize with chunk size parameters."""
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap
        logger.info(f"Initializing CodeChunker with max_chunk_size={max_chunk_size}, min_chunk_size={min_chunk_size}, overlap={overlap}")

    def chunk_file(self, file_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks from a file based on its content and type."""
        file_path = file_info["path"]
        content = file_info["content"]
        ext = file_info.get("extension", "")
        logger.debug(f"Chunking file: {file_path} (extension: {ext})")

        # Choose chunking strategy based on file type
        if ext == '.py':
            chunks = self._chunk_python(content, file_info)
        elif ext in ['.js', '.jsx', '.ts', '.tsx']:
            chunks = self._chunk_javascript(content, file_info)
        elif ext in ['.md', '.markdown']:
            chunks = self._chunk_markdown(content, file_info)
        else:
            # Default chunking for other text-based files
            if file_info.get("mime_type") and file_info["mime_type"].startswith("text/"):
                logger.debug(f"Using text chunker for {file_path} (mime: {file_info['mime_type']})")
                chunks = self._chunk_text(content, file_info)
            elif not file_info.get("mime_type"):
                 logger.warning(f"MIME type unknown for {file_path}, attempting text chunking.")
                 chunks = self._chunk_text(content, file_info)
            else:
                 logger.info(f"Skipping chunking for non-text file: {file_path} (mime: {file_info['mime_type']})")
                 chunks = [] # Don't process non-text files by default

        logger.info(f"Generated {len(chunks)} chunks for file: {file_path}")
        return chunks

    def _create_chunk_metadata(self, file_info: Dict[str, Any], chunk_index: int, start_line: int, end_line: int, chunk_type: str = "generic") -> Dict[str, Any]:
        """Create metadata for a chunk, ensuring values are Chroma-compatible."""
        # Ensure mime_type is a string, defaulting to 'unknown' if None
        mime_type = file_info.get("mime_type") or "unknown"

        return {
            "path": file_info["path"],
            # "mime_type": file_info.get("mime_type"),
            "mime_type": mime_type, # Use the sanitized mime_type
            "chunk_index": chunk_index,
            "chunk_type": chunk_type,
            "start_line": start_line,
            "end_line": end_line,
            "extension": file_info.get("extension", ""),
            "source_type": "github" # Add source type explicitly
        }

    def _chunk_python(self, content: str, file_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk Python file by classes and functions."""
        logger.debug(f"Chunking Python file: {file_info['path']}")
        chunks = []
        lines = content.splitlines()

        # Find class and function definitions (simplified regex)
        class_pattern = re.compile(r'^class\s+(\w+)')
        function_pattern = re.compile(r'^def\s+(\w+)')

        current_chunk_lines = []
        current_type = "module_level"
        start_line = 1
        in_block = False # Are we inside a class/function block?
        block_indent = 0

        for line_num, line in enumerate(lines, 1):
            stripped_line = line.lstrip()
            current_indent = len(line) - len(stripped_line)

            # Check for end of block (dedent to or below block start)
            if in_block and current_indent <= block_indent and stripped_line:
                if len("\n".join(current_chunk_lines)) >= self.min_chunk_size:
                    chunk_content = "\n".join(current_chunk_lines)
                    chunks.append({
                        "content": chunk_content,
                        "metadata": self._create_chunk_metadata(
                            file_info, len(chunks), start_line, line_num - 1, current_type
                        )
                    })
                current_chunk_lines = []
                in_block = False
                current_type = "module_level"
                start_line = line_num # Start new potential module-level chunk here

            # Check for new class/function definition
            new_block = False
            if not in_block:
                class_match = class_pattern.match(stripped_line)
                func_match = function_pattern.match(stripped_line)

                if class_match:
                    # Finalize previous module-level chunk if any
                    if current_chunk_lines:
                         chunk_content = "\n".join(current_chunk_lines)
                         chunks.append({
                             "content": chunk_content,
                             "metadata": self._create_chunk_metadata(
                                 file_info, len(chunks), start_line, line_num - 1, "module_level"
                             )
                         })
                    current_type = f"python_class_{class_match.group(1)}"
                    new_block = True
                elif func_match:
                     # Finalize previous module-level chunk if any
                    if current_chunk_lines:
                         chunk_content = "\n".join(current_chunk_lines)
                         chunks.append({
                             "content": chunk_content,
                             "metadata": self._create_chunk_metadata(
                                 file_info, len(chunks), start_line, line_num - 1, "module_level"
                             )
                         })
                    current_type = f"python_function_{func_match.group(1)}"
                    new_block = True

                if new_block:
                    current_chunk_lines = [line]
                    start_line = line_num
                    in_block = True
                    block_indent = current_indent
                    continue # Skip adding line again

            # Add line to the current chunk (either block or module-level)
            current_chunk_lines.append(line)

        # Add the last chunk (either block or module-level)
        if current_chunk_lines:
            chunk_content = "\n".join(current_chunk_lines)
            # Only add if it meets min size criteria or it's the only content
            if len(chunk_content) >= self.min_chunk_size or len(chunks) == 0:
                chunks.append({
                    "content": chunk_content,
                    "metadata": self._create_chunk_metadata(
                        file_info, len(chunks), start_line, len(lines), current_type
                    )
                })

        # If no structural chunks found, fall back to text chunking
        if not chunks and content:
            logger.debug(f"Falling back to text chunking for Python file: {file_info['path']}")
            return self._chunk_text(content, file_info)

        return chunks

    def _chunk_markdown(self, content: str, file_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk markdown by headers (simplistic approach)."""
        logger.debug(f"Chunking Markdown file: {file_info['path']}")
        chunks = []
        lines = content.splitlines()

        # Find all headers (lines starting with #)
        header_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
        header_indices = [-1] # Start with a pseudo-header before the first line
        header_levels = [0]

        for i, line in enumerate(lines):
            match = header_pattern.match(line)
            if match:
                header_indices.append(i)
                header_levels.append(len(match.group(1)))

        # Create chunks between headers
        for i in range(len(header_indices)):
            start_index = header_indices[i] + 1
            end_index = header_indices[i + 1] if i + 1 < len(header_indices) else len(lines)

            if start_index < end_index:
                section_lines = lines[start_index:end_index]
                section_content = "\n".join(section_lines).strip()
                start_line_num = start_index + 1
                end_line_num = end_index

                if section_content: # Avoid empty chunks
                     # Use the level of the header *starting* this section, or 'text' if it's the initial part
                    chunk_type = f"markdown_h{header_levels[i]}" if i > 0 else "markdown_intro"
                    chunks.append({
                        "content": section_content,
                        "metadata": self._create_chunk_metadata(
                            file_info, len(chunks), start_line_num, end_line_num, chunk_type
                        )
                    })

        # If no headers found or content remains, fall back to text chunking for the whole file
        if not chunks and content:
            logger.debug(f"Falling back to text chunking for Markdown file: {file_info['path']}")
            return self._chunk_text(content, file_info)

        # Further split large sections if needed
        final_chunks = []
        for chunk in chunks:
            if len(chunk['content']) > self.max_chunk_size:
                logger.debug(f"Splitting large Markdown chunk (type: {chunk['metadata']['chunk_type']}) from {file_info['path']}")
                # Create a temporary file_info for the sub-chunking
                temp_file_info = file_info.copy()
                temp_file_info['path'] = f"{chunk['metadata']['path']}#section_{chunk['metadata']['chunk_index']}" # Modify path for context
                sub_chunks = self._chunk_text(chunk['content'], temp_file_info)
                # Adjust metadata for sub_chunks (optional, maybe just keep original section metadata?)
                for sub_chunk in sub_chunks:
                     sub_chunk['metadata']['chunk_type'] = f"{chunk['metadata']['chunk_type']}_split"
                     sub_chunk['metadata']['path'] = chunk['metadata']['path'] # Revert path
                     sub_chunk['metadata']['original_start_line'] = chunk['metadata']['start_line'] # Keep original context
                     sub_chunk['metadata']['original_end_line'] = chunk['metadata']['end_line']
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _chunk_text(self, content: str, file_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk text files by character count with line awareness."""
        logger.debug(f"Chunking text file: {file_info['path']}")
        chunks = []
        lines = content.splitlines()
        if not lines:
            return []

        current_chunk_lines = []
        current_length = 0
        start_line_num = 1

        for i, line in enumerate(lines, 1):
            line_length = len(line) + 1  # +1 for the newline character

            # If adding this line exceeds max size AND current chunk is big enough
            if current_length + line_length > self.max_chunk_size and current_length >= self.min_chunk_size:
                chunk_text = "\n".join(current_chunk_lines)
                chunks.append({
                    "content": chunk_text,
                    "metadata": self._create_chunk_metadata(
                        file_info, len(chunks), start_line_num, i - 1, "text"
                    )
                })

                # Start new chunk with overlap (find suitable overlap point)
                overlap_line_index = len(current_chunk_lines) - 1
                overlap_char_count = 0
                while overlap_line_index > 0 and overlap_char_count < self.overlap:
                     overlap_char_count += len(current_chunk_lines[overlap_line_index]) + 1
                     overlap_line_index -= 1

                # Ensure overlap_line_index is valid
                overlap_line_index = max(0, overlap_line_index)

                current_chunk_lines = current_chunk_lines[overlap_line_index:]
                current_length = sum(len(l) + 1 for l in current_chunk_lines)
                start_line_num = start_line_num + overlap_line_index

            # Add line to current chunk (even if it makes it slightly exceed max size initially)
            current_chunk_lines.append(line)
            current_length += line_length

            # If the current chunk itself is now too large (e.g., one very long line)
            # And we haven't added it yet (avoids infinite loop)
            # This case needs careful handling, potentially splitting mid-line or just accepting large chunk
            # For simplicity now, we'll let it potentially exceed max_chunk_size if a single line causes it
            # or if the overlap logic couldn't prevent it.

        # Add the last chunk
        if current_chunk_lines:
            chunk_text = "\n".join(current_chunk_lines)
            chunks.append({
                "content": chunk_text,
                "metadata": self._create_chunk_metadata(
                    file_info, len(chunks), start_line_num, len(lines), "text"
                )
            })

        return chunks

    def _chunk_javascript(self, content: str, file_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk JavaScript/TypeScript by functions and classes (basic regex)."""
        logger.debug(f"Chunking JS/TS file: {file_info['path']}")
        # Basic implementation - similar structure to Python chunker
        # For robust JS/TS parsing, a dedicated library like 'tree-sitter' would be better

        chunks = []
        lines = content.splitlines()

        # Simple patterns for JS/TS functions and classes (can be improved)
        class_pattern = re.compile(r'^(export\s+)?(default\s+)?class\s+(\w+)')
        function_pattern = re.compile(r'^(export\s+)?(async\s+)?(function\*?|const|let|var)\s+(\w+)\s*(=|\()') # Basic check

        current_chunk_lines = []
        current_type = "module_level"
        start_line = 1
        in_block = False # Simplistic block tracking using braces
        brace_level = 0
        block_start_line = 0

        for line_num, line in enumerate(lines, 1):
            stripped_line = line.strip()

            # Track braces to approximate block end (very basic)
            if in_block:
                brace_level += line.count('{')
                brace_level -= line.count('}')
                if brace_level <= 0 and line_num > block_start_line:
                    # End of block detected (approximately)
                    current_chunk_lines.append(line) # Include the closing brace line
                    if len("\n".join(current_chunk_lines)) >= self.min_chunk_size:
                        chunk_content = "\n".join(current_chunk_lines)
                        chunks.append({
                            "content": chunk_content,
                            "metadata": self._create_chunk_metadata(
                                file_info, len(chunks), start_line, line_num, current_type
                            )
                        })
                    current_chunk_lines = []
                    in_block = False
                    current_type = "module_level"
                    start_line = line_num + 1 # Start next potential chunk after block
                    brace_level = 0
                    continue

            # Check for new class/function definition
            new_block = False
            if not in_block:
                class_match = class_pattern.match(stripped_line)
                func_match = function_pattern.match(stripped_line)

                potential_type = None
                if class_match:
                    potential_type = f"javascript_class_{class_match.group(3)}"
                    new_block = True
                elif func_match:
                    potential_type = f"javascript_function_{func_match.group(4)}"
                    new_block = True

                if new_block:
                     # Finalize previous module-level chunk if any
                    if current_chunk_lines:
                         chunk_content = "\n".join(current_chunk_lines)
                         chunks.append({
                             "content": chunk_content,
                             "metadata": self._create_chunk_metadata(
                                 file_info, len(chunks), start_line, line_num - 1, "module_level"
                             )
                         })

                    current_chunk_lines = [line]
                    current_type = potential_type
                    start_line = line_num
                    in_block = True
                    block_start_line = line_num
                    brace_level = line.count('{') - line.count('}') # Initial brace count
                    # If the definition line itself closes the block (unlikely but possible)
                    if brace_level <= 0 and line.count('{') > 0:
                        in_block = False # Treat as simple one-liner block
                    continue

            # Add line to the current chunk (either block or module-level)
            current_chunk_lines.append(line)

        # Add the last chunk
        if current_chunk_lines:
            chunk_content = "\n".join(current_chunk_lines)
            if len(chunk_content) >= self.min_chunk_size or len(chunks) == 0:
                chunks.append({
                    "content": chunk_content,
                    "metadata": self._create_chunk_metadata(
                        file_info, len(chunks), start_line, len(lines), current_type
                    )
                })

        # Fallback for JS/TS
        if not chunks and content:
            logger.debug(f"Falling back to text chunking for JS/TS file: {file_info['path']}")
            return self._chunk_text(content, file_info)

        return chunks 