import hashlib
from .models import MinimalSearchResults
import json
from pydantic import ValidationError
from pathlib import Path


def make_cache_key(query: str, k: int, index_version: str) -> str:
    """Hash a query, result count, and index version into a cache key."""
    data = [query, k, index_version]

    serial = json.dumps(data, ensure_ascii=False)

    cache_key = hashlib.sha256(
        serial.encode("utf-8")
    ).hexdigest()

    return cache_key


def get_index_version(
    index_type: str,
    index_directory: str = "data/processed"
) -> str:

    """Fingerprint the sizes and modification times of index artifacts."""
    files = ["chunks", "index", "inverted_index", "metadata"]
    information = []

    for name in files:
        path = Path(index_directory) / f"{index_type}_{name}.pkl"
        stats = path.stat()

        information.append([
            path.name,
            stats.st_size,
            stats.st_mtime_ns
        ])

    serialized = json.dumps(information)

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


def load_cached_result(
    cache_key: str,
    cache_directory: str = "data/cache/queries"
) -> MinimalSearchResults | None:

    """Load a cached retrieval result, ignoring unreadable cache entries."""
    path = Path(cache_directory) / f"{cache_key}.json"

    if not path.exists():
        return None
    try:
        data = path.read_text(encoding="utf-8")
        return MinimalSearchResults.model_validate_json(data)
    except (ValidationError, OSError, UnicodeError):
        return None


def save_cached_result(
    cache_key: str,
    result: MinimalSearchResults,
    cache_directory: str = "data/cache/queries"
) -> None:
    """Persist a validated retrieval result in the query cache."""
    path = Path(cache_directory) / f"{cache_key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8"
    )
