from fastapi import FastAPI, Form
from .__main__ import search_both, query_answer
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse
from pathlib import Path

app = FastAPI()

class SearchRequest(BaseModel):
    query: str
    k: int = Field(default=5, ge=1)

@app.post("/ask")
def ask(query: str = Form(...)):
    answer = query_answer(query)
    return {"recieved questions": answer}
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