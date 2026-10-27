from fastapi import FastAPI, Form
from .__main__ import search_both, query_answer, build_index
from pydantic import BaseModel, Field
from html import escape
from fastapi.responses import HTMLResponse
from pathlib import Path

app = FastAPI()

class SearchRequest(BaseModel):
    query: str
    k: int = Field(default=5, ge=1)


def indexes_exist(index_directory = "data/processed"):
    for name in ("docs", "code"):
        for suffix in ("chunks", "index", "inverted_index", "metadata"):
            if not (Path(index_directory) / f"{name}_{suffix}.pkl").is_file() :
                return False
    return True 


@app.post("/ask", response_class=HTMLResponse)
def ask(query: str = Form(...)):
    if not indexes_exist():
        build_index()

    answer = query_answer(query)

    html = Path("templates/index.html").read_text(encoding="utf-8")

    html = html.replace(
        "<!-- answer -->",
        f"<p>{escape(answer)}</p>"
    )

    return html
    
@app.get("/", response_class=HTMLResponse)
def homepage():
    html = Path("templates/index.html").read_text(encoding="utf-8")
    return html

@app.post("/search")
def search(request: SearchRequest):
    return search_both(request.query, request.k)

@app.post("/answer")
def answer(request: SearchRequest):
    return query_answer(request.query, request.k)