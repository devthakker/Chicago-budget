#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from chicago_budget_rag.engine import RAGEngine

app = FastAPI(title="Chicago Budget RAG")
templates = Jinja2Templates(directory=str(ROOT / "templates"))
engine = RAGEngine(ROOT / "data/index")
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "query": "",
            "answer": None,
            "results": [],
            "error": None,
        },
    )


@app.post("/", response_class=HTMLResponse)
async def ask(request: Request, query: str = Form(...)) -> HTMLResponse:
    query = query.strip()
    if not query:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "query": query,
                "answer": None,
                "results": [],
                "error": "Enter a question to search the budget documents.",
            },
        )

    try:
        payload = engine.answer(query, top_k=6)
        answer = payload["answer"]
        results = payload["results"]
        error = None
    except FileNotFoundError:
        answer = None
        results = []
        error = "Index not found. Run `python3 build_index.py` first."

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "query": query,
            "answer": answer,
            "results": results,
            "error": error,
        },
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
