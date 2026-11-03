from fastapi import FastAPI, Form
from .__main__ import search_both, query_answer, build_index
from pydantic import BaseModel, Field
from html import escape
from fastapi.responses import HTMLResponse
from pathlib import Path
from .models import MinimalSearchResults

app = FastAPI()


class SearchRequest(BaseModel):
    """Validate a query submitted to the HTTP API."""
    query: str
    k: int = Field(default=5, ge=1)


def indexes_exist(index_directory: str = "data/processed") -> bool:
    """Check whether the separate lexical index files are available."""
    for name in ("docs", "code"):
        for suffix in ("chunks", "index", "inverted_index", "metadata"):
            if not (Path(index_directory) / f"{name}_{suffix}.pkl").is_file():
                return False
    return True


@app.post("/ask", response_class=HTMLResponse, response_model=None)
def ask(query: str = Form(...)) -> str:
    """Render a generated answer in the HTML page."""
    if not indexes_exist():
        build_index()

    answer = query_answer(query)

    html = Path("templates/index.html").read_text(encoding="utf-8")

    html = html.replace(
        "<!-- answer -->",
        f"<p>{escape(answer)}</p>"
    )

    return html


@app.get("/", response_class=HTMLResponse, response_model=None)
def homepage() -> str:
    """Serve the local query form."""
    html = Path("templates/index.html").read_text(encoding="utf-8")
    return html


@app.post("/search", response_model=None)
def search(request: SearchRequest) -> MinimalSearchResults:
    """Return ranked source locations for a query."""
    return search_both(request.query, request.k)


@app.post("/answer", response_model=None)
def answer(request: SearchRequest) -> str:
    """Return a generated answer for an HTTP query."""
    return query_answer(request.query, request.k)
