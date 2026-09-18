import ast
from pathlib import Path
from typing import List, Tuple, Generator, Dict, Any


def get_target_files(raw_dir: Path | str) -> List[Tuple[Path, str]]:
    """Recursively collect and tag eligible files."""
    target_files = []
    base = Path(raw_dir)
    if not base.exists():
        return target_files

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
    """Read and parse a single file safely without holding entire repos in memory."""
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


def parse_nodes(raw_dir: Path | str, max_chunk_size: int = 2000) -> List[Dict[str, Any]]:
    """Traverse Python files, extract node boundaries, and return chunks."""
    target_files = get_target_files(raw_dir)
    chunks: List[Dict[str, Any]] = []

    for file_path, file_type in target_files:
        if file_type != "python":
            continue
        parsed = parse_python_file(file_path)

        ast_tree = parsed["tree"]
        line_offsets = parsed["line_offsets"]
        str_path = parsed["path"]

        for node in ast_tree.body:
            if not hasattr(node, "lineno"):
                continue

            start, end = get_node_char(node, line_offsets)
            chunk_length = end - start

            if 0 < chunk_length <= max_chunk_size:
                chunks.append({
                    "file_path": str_path,
                    "first_character_index": start,
                    "last_character_index": end,
                })

    return chunks