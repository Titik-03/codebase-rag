import ast
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict, Any


def get_target_files(raw_dir: Path | str) -> List[Tuple[Path, str]]:
    """Recursively collect and tag eligible files."""
    target_files = []
    base = Path(raw_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"Source directory not found: {base}")

    for path in base.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix == ".py":
            target_files.append((path, "python"))
        elif path.suffix in [".md", ".txt", ".rst"]:
            target_files.append((path, "markdown"))
    return target_files


def calculate_char_offsets(source_code: str) -> List[int]:
    """Compute running 0-based character offsets for line starts."""
    running_offset = 0
    offsets = [0]
    for line in source_code.splitlines(keepends=True):
        running_offset += len(line)
        offsets.append(running_offset)
    return offsets


def parse_python_file(path: Path) -> Dict[str, Any] | None:
    """Read and parse a file safely without holding entire repos in memory."""
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        return {
            "path": str(path),
            "source": source,
            "tree": tree,
            "line_offsets": calculate_char_offsets(source),
        }
    except (SyntaxError, UnicodeDecodeError, OSError):
        return None


def get_node_char(node: ast.AST, offset: List[int]) -> Tuple[int, int]:
    """Map AST node lines to absolute character offsets."""
    start = offset[node.lineno - 1] + node.col_offset
    end = offset[node.end_lineno - 1] + node.end_col_offset
    return start, end


def chunk_markdown(raw_dir: Path | str,
                   max_chunk_size: int = 2000,
                   overlap: int = 120) -> List[Dict[str, Any]]:
    """Simple sliding-window chunking for markdown/txt/rst files."""
    chunks: List[Dict[str, Any]] = []

    for file_path, file_type in tqdm(
        get_target_files(raw_dir),
        desc="Chunking documentation",
        unit="file"
    ):
        if file_type != "markdown":
            continue

        try:
            source = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        pos = 0
        while pos < len(source):
            end = min(pos + max_chunk_size, len(source))
            chunks.append({
                "file_path": str(file_path),
                "first_character_index": pos,
                "last_character_index": end,
            })
            if end == len(source):
                break
            pos = end - min(overlap, max_chunk_size - 1)

    return chunks


def _slice_oversized(start: int, end: int, str_path: str, max_chunk_size: int,
                     chunks: List[Dict[str, Any]], overlap: int = 50) -> None:
    """Manually slice a node with no splittable children into fixed-size,
    overlapping windows so each chunk stays under max_chunk_size."""
    pos = start
    while pos < end:
        chunk_end = min(pos + max_chunk_size, end)
        chunks.append({
            "file_path": str_path,
            "first_character_index": pos,
            "last_character_index": chunk_end,
        })
        if chunk_end >= end:
            break
        pos = chunk_end - overlap


def _emit_node(node, line_offsets, str_path, max_chunk_size, chunks):
    start, end = get_node_char(node, line_offsets)
    node_length = end - start

    if node_length <= max_chunk_size:
        chunks.append({
            "file_path": str_path,
            "first_character_index": start,
            "last_character_index": end,
        })
        return

    children = []
    for attr in ("body", "orelse", "handlers", "finalbody"):
        children.extend(getattr(node, attr, None) or [])

    if not children:
        _slice_oversized(start, end, str_path, max_chunk_size, chunks)
        return

    for child in children:
        if not hasattr(child, "lineno"):
            continue
        _emit_node(child, line_offsets, str_path, max_chunk_size, chunks)


def parse_nodes(raw_dir: Path | str,
                max_chunk_size: int = 2000) -> List[Dict[str, Any]]:
    """Traverse Python files, extract node boundaries, and return chunks."""
    target_files = get_target_files(raw_dir)
    chunks: List[Dict[str, Any]] = []
    for file_path, file_type in tqdm(
        target_files,
        desc="Chunking Python files",
        unit="file"
    ):
        if file_type != "python":
            continue
        parsed = parse_python_file(file_path)
        if parsed is None:
            continue

        ast_tree = parsed["tree"]
        line_offsets = parsed["line_offsets"]
        str_path = parsed["path"]

        chunk_start = None
        chunk_end = None

        for node in ast_tree.body:
            if not hasattr(node, "lineno"):
                continue

            start, end = get_node_char(node, line_offsets)
            node_length = end - start

            if node_length > max_chunk_size:
                if chunk_start is not None:
                    chunks.append({
                        "file_path": str_path,
                        "first_character_index": chunk_start,
                        "last_character_index": chunk_end,
                    })
                    chunk_start = None
                _emit_node(node, line_offsets, str_path,
                           max_chunk_size, chunks)
                continue

            if (
                chunk_start is not None
                and (end - chunk_start) > max_chunk_size
            ):
                chunks.append({
                    "file_path": str_path,
                    "first_character_index": chunk_start,
                    "last_character_index": chunk_end,
                })
                chunk_start = None

            if chunk_start is None:
                chunk_start = start

            chunk_end = end

        if chunk_start is not None:
            chunks.append({
                "file_path": str_path,
                "first_character_index": chunk_start,
                "last_character_index": chunk_end,
            })
    return chunks


def add_source_to_chunks(chunks: List[Dict[str, Any]]) -> None:
    """Fill in the 'text' field for each chunk by slicing its source file.
    Reads each file once, no matter how many chunks reference it."""
    file_cache: Dict[str, str] = {}

    for c in chunks:
        path = c["file_path"]

        if path not in file_cache:
            try:
                file_cache[path] = Path(path).read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                file_cache[path] = ""

        source = file_cache[path]
        start, end = c["first_character_index"], c["last_character_index"]
        c["text"] = source[start:end]


def all_chunks():
    chunks = parse_nodes("vllm-0.10.1", max_chunk_size=2000) + \
        chunk_markdown("vllm-0.10.1", max_chunk_size=2000)
    add_source_to_chunks(chunks)
    return chunks
